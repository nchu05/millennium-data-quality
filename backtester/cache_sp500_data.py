import pandas as pd
import yfinance as yf
import pickle

def fetch_sp500_tickers():
    sp500_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    tickers = sp500_table['Symbol'].tolist()
    tickers = [ticker.replace('.', '-') for ticker in tickers]
    return tickers

def download_data(tickers, start_date, end_date):
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        group_by='ticker',
        auto_adjust=False,
        threads=True # this enables parallel downloading
    )
    return data

def calculate_vwap(data):
    vwap_data = {}
    for ticker in data.columns.levels[0]:
        adj_close = data[ticker]['Adj Close']
        volume = data[ticker]['Volume']
        vwap = (adj_close * volume).cumsum() / volume.cumsum()
        ticker_df = pd.DataFrame({
            'Adj Close': adj_close,
            'Volume': volume,
            'VWAP': vwap
        })
        vwap_data[ticker] = ticker_df
    return vwap_data

def save_data(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def main():
    tickers = fetch_sp500_tickers()
    start_date = '2010-01-01'
    end_date = '2024-11-20'
    data = download_data(tickers, start_date, end_date)
    vwap_data = calculate_vwap(data)
    save_data(vwap_data, 'sp500_data.pkl')
    print("Data has been cached and saved to sp500_data.pkl")

if __name__ == '__main__':
    main()