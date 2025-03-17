import yfinance as yf

def get_stock_data(ticker):
    df = yf.download(ticker, start='2015-01-01', end='2025-01-01')[['Close']]
    return df
