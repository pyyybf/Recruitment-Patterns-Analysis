import csv
import json
import numpy as np
import random
import shutil
import os

from utils import fs


# Create a dictionary to store the information
def get_employee(csv_file, output_file):
    data = {}

    # Read the CSV file
    with open(csv_file, 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            cik_number = int(row['file_name'].split('_')[0])  # Extracting CIK number from file name
            year = row['year']
            employee_num = int(row['employee_num'])
            file_name = row['file_name']

            # Create the inner dictionary for the year
            year_dict = {
                "employee_num": employee_num,
                "file_name": file_name
            }

            # Check if the CIK number already exists in the data dictionary
            if cik_number in data:
                # Create a new entry for the year
                data[cik_number][year] = year_dict
            else:
                # Create a new entry for the CIK number and year
                data[cik_number] = {year: year_dict}

    # Convert the data to JSON format
    json_data = json.dumps(data, indent=4)

    # Write the JSON data to a file
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json_file.write(json_data)


def get_change_rate(source_file, target_file, tmp_output_dir):
    with open(source_file, "r") as file:
        employee_num_data = json.load(file)

    time_num_headers = ["cik"]
    time_num_headers += [str(year) for year in range(2016, 2023)]
    time_num_headers += [f"{year}_file" for year in range(2016, 2023)]

    with open(f"{tmp_output_dir}/time_num.csv", "w") as time_num_csv:
        time_num_csv.write(f"{','.join(time_num_headers)}\n")
        for cik, data in employee_num_data.items():
            nums = []
            file_names = []
            pos_count = 0
            for year in range(2016, 2023):
                num = ""
                file_name = ""
                if str(year) in data:
                    pos_count += 1
                    num = str(data[str(year)]["employee_num"])
                    file_name = data[str(year)]["file_name"]
                nums.append(num)
                file_names.append(file_name)
            # Average annual employee number is larger 10, and employee number must be available for years 21 and 22
            if sum([int(num or 0) for num in nums]) / pos_count > 10 and nums[-1] != "" and nums[-2] != "":
                time_num_csv.write(f"{cik},{','.join(nums)},{','.join(file_names)}\n")

    data = []
    with open(f"{tmp_output_dir}/time_num.csv", "r") as fp:
        rows = csv.DictReader(fp)
        for row in rows:
            row_data = {
                "cik": row["cik"],
                "2016_num": row["2016"],
                "2016_file": row["2016_file"]
            }
            for year in range(2017, 2023):
                row_data[f"{year}_num"] = row[str(year)]
                row_data[f"{year}_file"] = row[f"{year}_file"]
                pre_num = row[str(year - 1)]
                cur_num = row[str(year)]
                if pre_num == "" or cur_num == "":
                    change_rate = ""
                else:
                    pre_num = int(pre_num)
                    cur_num = int(cur_num)
                    if cur_num == pre_num:
                        change_rate = 0
                    elif pre_num == 0:
                        change_rate = 1
                    else:
                        change_rate = (cur_num - pre_num) / pre_num
                row_data[str(year)] = change_rate
            data.append(row_data)

    fieldnames = ["cik"]
    fieldnames += [str(year) for year in range(2017, 2023)]
    fieldnames += [f"{year}_num" for year in range(2016, 2023)]
    fieldnames += [f"{year}_file" for year in range(2016, 2023)]
    with open(target_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def split_train_test(full_data_path, output_dir):
    data = []
    with open(full_data_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames
        for row in reader:
            data.append(row)

    # Split training and test sets
    train_cnt = int(len(data) * 0.8)
    random.seed(123)
    random.shuffle(data)
    train_set = data[:train_cnt]
    test_set = data[train_cnt:]

    with open(f"{output_dir}/train_change_rate.csv", "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(train_set)

    with open(f"{output_dir}/test_change_rate.csv", "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(test_set)


def split_data_file(source_dir, target_dir, index_dir):
    for prefix in ["train", "test"]:
        fs.clear_dir(f"{target_dir}/{prefix}_data")
        for year in range(2016, 2023):
            os.makedirs(f"{target_dir}/{prefix}_data/{year}")
        with open(f"{index_dir}/{prefix}_change_rate.csv", "r") as fp:
            rows = csv.DictReader(fp)
            for row in rows:
                for year in range(2016, 2023):
                    file_name = row[f"{year}_file"]
                    if file_name != "":
                        shutil.copy(f"{source_dir}/{year}/{file_name}",
                                    f"{target_dir}/{prefix}_data/{year}/{file_name}")


def split_row_filled(prefix="train"):
    with open(f"./{prefix}_change_rate.csv", "r") as source:
        rows = csv.DictReader(source)
        with open(f"./{prefix}_change_rate_filled.csv", "w") as target:
            writer = csv.DictWriter(target, fieldnames=["cik", "file_name", "year", "change_rate"])
            writer.writeheader()

            for row in rows:
                change_rates = []
                for year in range(2017, 2023):
                    if row[str(year)] != "":
                        change_rates.append(float(row[str(year)]))
                fill_val = np.mean(change_rates)

                for year in range(2017, 2023):
                    row_data = {
                        "cik": row["cik"],
                        "file_name": row[f"{year}_file"],
                        "year": year,
                        "change_rate": row[str(year)] if row[str(year)] != "" else fill_val,
                    }
                    writer.writerow(row_data)


def split_row_unfilled(output_dir, prefix="train"):
    with open(f"{output_dir}/{prefix}_change_rate.csv", "r") as source:
        rows = csv.DictReader(source)
        with open(f"{output_dir}/{prefix}_change_rate_unfilled.csv", "w") as target:
            writer = csv.DictWriter(target, fieldnames=["cik", "file_name", "year", "change_rate"])
            writer.writeheader()

            for row in rows:
                for year in range(2017, 2023):
                    if row[str(year)] != "":
                        row_data = {
                            "cik": row["cik"],
                            "file_name": row[f"{year}_file"],
                            "year": year,
                            "change_rate": row[str(year)],
                        }
                        writer.writerow(row_data)
