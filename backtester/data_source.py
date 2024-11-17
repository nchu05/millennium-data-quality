from abc import ABC, abstractmethod
import pandas as pd
import yfinance as yf
from typing import List

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
        return data['Close']