import json
import random
import shutil
import csv
import os

from utils import fs


def split_train_test():
    with open("./employee_num.json", "r") as file:
        employee_num_data = json.load(file)

    time_num_headers = ["cik"]
    time_num_headers += [str(year) for year in range(2016, 2023)]
    time_num_headers += [f"{year}_file" for year in range(2016, 2023)]

    with open("./time_num.csv", "w") as time_num_csv:
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
            if sum([int(num or 0) for num in nums]) / pos_count > 10 and nums[-1] != "" and nums[-2] != "":
                time_num_csv.write(f"{cik},{','.join(nums)},{','.join(file_names)}\n")

    data = []
    with open("./time_num.csv", "r") as fp:
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
    with open("./change_rate.csv", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    # 拆分train和test
    train_cnt = int(len(data) * 0.8)
    random.seed(123)
    random.shuffle(data)
    train_set = data[:train_cnt]
    test_set = data[train_cnt:]

    with open("./train_change_rate.csv", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(train_set)

    with open("./test_change_rate.csv", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(test_set)


def split_data_file():
    for prefix in ["train", "test"]:
        fs.clear_dir(f"./{prefix}_data")
        for year in range(2016, 2023):
            os.mkdir(f"./{prefix}_data/{year}")
        with open(f"./{prefix}_change_rate.csv", "r") as fp:
            rows = csv.DictReader(fp)
            for row in rows:
                for year in range(2016, 2023):
                    file_name = row[f"{year}_file"]
                    if file_name != "":
                        shutil.copy(f"./data_cleaned/{year}/{file_name}", f"./{prefix}_data/{year}/{file_name}")
