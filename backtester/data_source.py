from abc import ABC, abstractmethod
import pandas as pd
import yfinance as yf

class DataSource(ABC):
    """Interface for fetching historical market data."""
    
    @abstractmethod
    def get_historical_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical data for a given ticker and date range."""
        pass

class YahooFinanceDataSource(DataSource):
    """Implementation of DataSource using Yahoo Finance."""
    
    def get_historical_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        data = yf.download(ticker, start=start_date, end=end_date)
        return data[['Close']]