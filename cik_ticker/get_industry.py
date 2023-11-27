import yfinance as yf

def get_industry_info(ticker_symbol):
    # Create a Ticker object for the given ticker symbol
    stock = yf.Ticker(ticker_symbol)

    # Fetch and return the industry information
    try:
        industry = stock.info.get('industry')
        return industry
    except Exception as e:
        print(f"Error retrieving industry info for {ticker_symbol}: {e}")
        return None

# Example usage
ticker = 'AAPL'  # Apple Inc. as an example
industry = get_industry_info(ticker)
if industry:
    print(f"Industry for {ticker}: {industry}")
else:
    print("Industry information not found.")
