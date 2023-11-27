import requests
from bs4 import BeautifulSoup

def get_sic_code(cik):
    url = f'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=&dateb=&owner=exclude&count=40'
    response = requests.get(url, headers=headers, timeout=20)
    if response.status_code != 200:
        print("Failed to retrieve data")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')
    # This part might require adjustments depending on the actual page structure
    for row in soup.find_all('div', class_='companyInfo'):
        try:
            sic_code = row.find('a', {'href': lambda x: x and 'SIC' in x}).text
            return sic_code
        except AttributeError:
            continue

    print("SIC code not found")
    return None

# Example usage
cik = '320193'  # Example CIK for Apple Inc. No need to add zeros
sic_code = get_sic_code(cik)
if sic_code:
    print(f"SIC Code for CIK {cik}: {sic_code}")
