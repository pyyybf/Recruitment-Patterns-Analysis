import requests
from lxml import etree
import re
import pandas as pd

hearders = {'User-Agent': 'Mozilla/5.0'}  

def get_unique_ciks(input_csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Extract unique CIKs from the 'cik' column
    unique_ciks = df['cik'].unique()

    return unique_ciks

def get_sic_code(cik):
    try:
        url = f'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=&dateb=&owner=exclude&count=40'
        response = requests.get(url, headers=headers, timeout=20)
        if response.status_code != 200:
            print("Failed to retrieve data")
            return None

        element = etree.HTML(response.content)
        p = element.xpath("/html/body/div[4]/div[1]/div[3]/p")[0]
        line_text = ""
        for text in p.itertext():
            if text.strip() == "State location:":
                break
            line_text += text.strip()

        info = re.split(r":|- ", line_text)

        return info[1], info[2]

    except Exception as e:
        print(e)
        print(f"Error: {url}")
        return None, None

input_csv_file = "" # path to the csv file with source cik
ciks = get_get_unique_ciks
output_csv_file = 'sic_codes.csv'

sics = []

for cik in ciks:
    sic_code, name = get_sic_code(cik)
    if sic_code:
        sics.append([cik, sic_code, name])

# Write the data to a CSV file
with open(csv_filename, 'w', newline='') as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow(['CIK', 'SIC Code', 'Name'])
    # Write data rows
    writer.writerows(csv_filename)

print(f"Data has been written to '{csv_filename}'.")
        
