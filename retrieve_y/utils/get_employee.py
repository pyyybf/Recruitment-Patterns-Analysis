import csv
import json


# Create a dictionary to store the information
def get_employee(csv_file="./number_match.csv", output_file="./employee_num.json"):
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


def get_change_rate(source_file="./employee_num.json", target_file="./change_rate.csv"):
    with open(source_file, "r") as file:
        employee_num_data = json.load(file)

    with open(target_file, "w") as fp:
        fp.write("cik,from_year,from_num,to_year,to_num,change_rate,from_filename,to_filename\n")
        for cik, data in employee_num_data.items():
            years = sorted(data)
            for i in range(1, len(years)):
                from_file = data[years[i - 1]]["file_name"]
                to_file = data[years[i]]["file_name"]

                from_num = data[years[i - 1]]["employee_num"]
                to_num = data[years[i]]["employee_num"]
                change_rate = (to_num - from_num) / (from_num if from_num > 0 else 1)

                fp.write(f"{cik},{years[i - 1]},{from_num},{years[i]},{to_num},{change_rate},{from_file},{to_file}\n")


if __name__ == "__main__":
    csv_file = "C:/Alycia/USC/2023Fall/ISE540/Project/retrieve_y_result/test_csv/result.csv"
    output_file = "C:/Alycia/USC/2023Fall/ISE540/Project/retrieve_y_result/test_csv/results.json"
    get_employee(csv_file, output_file)
    print("All done!")
