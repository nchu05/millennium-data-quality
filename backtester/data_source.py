from abc import ABC, abstractmethod
import pandas as pd
import yfinance as yf
from typing import List, Optional

class DataSource(ABC):
    """Interface for fetching historical market data."""
    
    @abstractmethod
    def get_historical_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical data for given tickers and date range."""
        pass

class YahooFinanceDataSource(DataSource):
    """Implementation of DataSource using Yahoo Finance."""
    
    def get_historical_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        data = yf.download(tickers, start=start_date, end=end_date)
        return data['Adj Close']
    
    def get_sp500_components(self) -> pd.DataFrame:
        """Fetch S&P 500 components and their market capitalizations."""
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        sp500_table = table[0]
        tickers = sp500_table['Symbol'].tolist()
        
        market_caps = {}
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            market_caps[ticker] = stock.info.get('marketCap', 0)
        
        return pd.DataFrame(list(market_caps.items()), columns=['Ticker', 'MarketCap'])
    
    def calculate_market_weights(self, components: pd.DataFrame) -> pd.DataFrame:
        """Calculate market weights of S&P 500 components."""
        total_market_cap = components['MarketCap'].sum()
        components['Weight'] = components['MarketCap'] / total_market_cap
        return components
    
    def create_market_weighted_portfolio(self, components: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        """Create a market-weighted portfolio from S&P 500 components."""
        tickers = components['Ticker'].tolist()
        data = self.get_historical_data(tickers, start_date, end_date)
        
        weighted_data = data.mul(components.set_index('Ticker')['Weight'], axis=1)
        portfolio = weighted_data.sum(axis=1)
        return portfolio
    
    def get_sp500_components_on_date(self, date: str, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch S&P 500 components and their market capitalizations on a specific date.
        Optionally limit the number of tickers fetched.
        """
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        sp500_table = table[0]
        tickers = sp500_table['Symbol'].tolist()
        if limit:
            tickers = tickers[:limit]
        market_caps = {}
        for ticker in tickers:
            try:
                market_cap = self.get_market_cap_on_date(ticker, date)
                if market_cap is not None:
                    market_caps[ticker] = market_cap
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")

        components = pd.DataFrame(list(market_caps.items()), columns=['Ticker', 'MarketCap'])
        return components

    def get_market_cap_on_date(self, ticker: str, date: str) -> Optional[float]:
        """
        Fetch the market capitalization of a ticker on a specific date.
        """
        data = yf.Ticker(ticker).history(start=date, end=date)
        if data.empty:
            return None
        close_price = data['Close'].iloc[0]
        shares_outstanding = yf.Ticker(ticker).get_shares_full(start=date, end=date)
        if shares_outstanding.empty:
            return None
        shares = shares_outstanding['Shares'].iloc[0]
        return close_price * shares

    def get_sp500_index_data(self, start_date: str, end_date: str) -> pd.Series:
        """
        Fetch historical data for the S&P 500 index (e.g., using the ^GSPC ticker).
        """
        index_data = yf.download('^GSPC', start=start_date, end=end_date)
        return index_data['Adj Close']