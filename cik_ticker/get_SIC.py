import requests
from bs4 import BeautifulSoup
import pandas as pd
import csv


def get_unique_ciks(input_csv_file):
    df = pd.read_csv(input_csv_file)
    unique_ciks = df['cik'].unique()
    return unique_ciks


def get_sic_code(cik):
    headers = {'User-Agent': 'Mozilla/5.0'}
    url = f'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=&dateb=&owner=exclude&count=40'
    try:
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return None
    except Exception as err:
        print(f"An error occurred: {err}")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')
    sic_code = None
    try:
        sic_code = soup.find('div', class_='companyInfo').find('a', href=lambda x: x and 'SIC' in x).text
    except AttributeError:
        print("SIC code not found")

    return sic_code


input_csv_file = 'change_rate.csv'  # Path to the CSV file with source CIK
ciks = list(get_unique_ciks(input_csv_file))
output_csv_file = 'sic_codes.csv'

sics = []

for cik in ciks:
    sic_code = get_sic_code(cik)
    if sic_code:
        sics.append([cik, sic_code])

# Write the data to a CSV file
with open(output_csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['CIK', 'SIC Code'])
    writer.writerows(sics)  # Write the list 'sics', not the filename

print(f"Data has been written to '{output_csv_file}'.")
