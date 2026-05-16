import yfinance as yf
print(yf.Ticker('INFY.NS').history(period='1mo').to_dict())
