import requests
import yfinance as yf
from bs4 import BeautifulSoup

def cik_to_ticker(cik):
    # URL to SEC's company search using CIK
    url = f'https://www.sec.gov/cgi-bin/browse-edgar?CIK={cik}&Find=Search&owner=exclude&action=getcompany'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Try to find the ticker symbol from the SEC page
    for link in soup.find_all('a'):
        if 'ticker' in link.get('href', ''):
            return link.text.strip()

    return None

def get_industry_info(ticker):
    try:
        company = yf.Ticker(ticker)
        return company.info.get('industry')
    except Exception as e:
        print(f"Error retrieving industry info for {ticker}: {e}")
        return None

# Example usage
cik = '0000320193' # CIK for Apple Inc. as an example
ticker = cik_to_ticker(cik)

if ticker:
    print(f"Ticker for CIK {cik}: {ticker}")
    industry = get_industry_info(ticker)
    if industry:
        print(f"Industry for {ticker}: {industry}")
else:
    print(f"No ticker found for CIK {cik}")
