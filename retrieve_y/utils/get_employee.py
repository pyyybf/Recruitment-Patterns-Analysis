import csv
import json


# Create a dictionary to store the information
def get_employee(csv_file, output_file):
    data = {}

    # Read the CSV file
    with open(csv_file, 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            cik_number = int(row['file_name'].split('_')[1])  # Extracting CIK number from file name
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
                if year in data[cik_number]:
                    # Append the data for the existing year
                    data[cik_number][year].append(year_dict)
                else:
                    # Create a new entry for the year
                    data[cik_number][year] = [year_dict]
            else:
                # Create a new entry for the CIK number and year
                data[cik_number] = {year: [year_dict]}

    # Convert the data to JSON format
    json_data = json.dumps(data, indent=4)

    # Write the JSON data to a file
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json_file.write(json_data)


if __name__ == "__main__":
    csv_file = "C:/Alycia/USC/2023Fall/ISE540/Project/retrieve_y_result/test_csv/result.csv"
    output_file = "C:/Alycia/USC/2023Fall/ISE540/Project/retrieve_y_result/test_csv/results.json"
    get_employee(csv_file, output_file)
    print("All done!")
