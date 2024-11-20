import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from backtester.data_source import YahooFinanceDataSource
from backtester.order_generator import MeanReversionOrderGenerator
from backtester.backtest_engine import SimpleBacktestEngine

class TestMarketWeightedPortfolio(unittest.TestCase):
    @patch('backtester.data_source.YahooFinanceDataSource.get_market_cap_on_date')
    @patch('backtester.data_source.YahooFinanceDataSource.get_historical_data')
    @patch('backtester.data_source.YahooFinanceDataSource.get_sp500_index_data')
    def test_sp500_equivalent_portfolio(self, mock_index_data, mock_historical_data, mock_market_cap_on_date):
        start_date = '2023-01-02'
        end_date = '2023-01-06'  # pick any short period where index is not rebalanced
        specific_date = '2023-01-02'

        # mock market caps
        tickers = ['AAPL', 'MSFT', 'AMZN']  
        market_caps = {
            'AAPL': 2e12,
            'MSFT': 1.8e12,
            'AMZN': 1.6e12
        }
        mock_market_cap_on_date.side_effect = lambda ticker, date: market_caps.get(ticker, None)

        # mock historical price data for components
        dates = pd.date_range(start=start_date, end=end_date)
        mock_prices = pd.DataFrame({
            'AAPL': [150, 152, 154, 153, 155],
            'MSFT': [250, 251, 253, 252, 254],
            'AMZN': [3300, 3320, 3340, 3330, 3350]
        }, index=dates)
        mock_historical_data.return_value = mock_prices
        
        # mock SPY data
        mock_index_prices = pd.Series([3800, 3820, 3840, 3830, 3850], index=dates, name='^GSPC')
        mock_index_data.return_value = mock_index_prices

        data_source = YahooFinanceDataSource()
        sp500_components = data_source.get_sp500_components_on_date(specific_date)
        sp500_components = data_source.calculate_market_weights(sp500_components)

        portfolio_values = data_source.create_market_weighted_portfolio(
            components=sp500_components,
            start_date=start_date,
            end_date=end_date
        )

        portfolio_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        index_return = (mock_index_prices.iloc[-1] / mock_index_prices.iloc[0]) - 1

        self.assertAlmostEqual(
            portfolio_return,
            index_return,
            delta=0.003, # at most 0.3% difference
            msg=f"Portfolio return {portfolio_return} does not match index return {index_return}"
        )

class TestBacktesterAndOrderGenerator(unittest.TestCase):

    @patch('backtester.data_source.YahooFinanceDataSource.get_historical_data')
    def test_mean_reversion_order_generation(self, mock_get_historical_data):
        # generate 100 random mock points for mean reversion 100 day window strat
        num_days = 150
        dates = pd.date_range(start='2023-01-01', periods=num_days)
        # create a price series that fluctuates around a mean to trigger buy/sell signals
        np.random.seed(0)  # for reproducibility
        prices = 100 + np.cumsum(np.random.normal(0, 1, size=num_days))
        mock_data = pd.DataFrame({
            'AAPL': prices
        }, index=dates)
        mock_get_historical_data.return_value = mock_data

        data_source = YahooFinanceDataSource()
        data = data_source.get_historical_data(['AAPL'], '2023-01-01', dates[-1].strftime('%Y-%m-%d'))
        order_generator = MeanReversionOrderGenerator()
        orders = order_generator.generate_orders(data)

        self.assertGreater(len(orders), 0)
        # check that orders only start after the 100th data point (since earlier ones can't compute the rolling average)
        orders_dates = [order['date'] for order in orders]
        self.assertTrue(all(order_date >= dates[99] for order_date in orders_dates))
        order_types = set(order['type'] for order in orders)
        self.assertTrue(order_types >= {'BUY', 'SELL'})
        self.assertTrue(all(order['ticker'] == 'AAPL' for order in orders))
        required_keys = {'date', 'type', 'ticker', 'quantity'}
        self.assertTrue(all(required_keys.issubset(order.keys()) for order in orders))

    def test_backtest_engine_insufficient_cash(self):
        backtest_engine = SimpleBacktestEngine(initial_cash=5000)
        orders = [{"date": '2023-01-01', "type": "BUY", "ticker": "AAPL", "quantity": 100}]
        data = pd.DataFrame({
            'AAPL': [150]
        }, index=[pd.Timestamp('2023-01-01')])

        results = backtest_engine.run_backtest(orders, data)
        portfolio_values = results['portfolio_values']
        self.assertEqual(portfolio_values.iloc[0]['Portfolio Value'], 5000)

    def test_backtest_engine_sell_without_holdings(self):
        backtest_engine = SimpleBacktestEngine(initial_cash=10000)
        orders = [{"date": '2023-01-01', "type": "SELL", "ticker": "AAPL", "quantity": 50}]
        data = pd.DataFrame({
            'AAPL': [150]
        }, index=[pd.Timestamp('2023-01-01')])

        results = backtest_engine.run_backtest(orders, data)
        portfolio_values = results['portfolio_values']
        self.assertEqual(portfolio_values.iloc[0]['Portfolio Value'], 10000)

    @patch('backtester.data_source.YahooFinanceDataSource.get_historical_data')
    def test_order_generator_with_single_data_point(self, mock_get_historical_data):
        dates = [pd.Timestamp('2023-01-01')]
        mock_data = pd.DataFrame({
            'AAPL': [150]
        }, index=dates)
        mock_get_historical_data.return_value = mock_data

        data_source = YahooFinanceDataSource()
        data = data_source.get_historical_data(['AAPL'], '2023-01-01', '2023-01-01')
        order_generator = MeanReversionOrderGenerator()
        orders = order_generator.generate_orders(data)

        self.assertEqual(len(orders), 0)


if __name__ == '__main__':
    unittest.main()