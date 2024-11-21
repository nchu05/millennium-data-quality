import pickle
import yfinance as yf
import pandas as pd
from data_source import YahooFinanceDataSource
from order_generator import BettingAgainstBetaOrderGenerator
from backtest_engine import EquityBacktestEngine
from metrics import ExtendedMetrics

def main():
    with open('sp500_data.pkl', 'rb') as f:
        sp500_data = pickle.load(f)
        
    spy_data = yf.download('SPY', start='2010-01-01', end='2023-10-01')
    sp500_data['SPY'] = spy_data[['Adj Close', 'Volume']]

    order_generator = BettingAgainstBetaOrderGenerator(lookback_period=60, rebalance_frequency='ME')
    orders = order_generator.generate_orders(sp500_data)

    combined_data = pd.DataFrame()
    for ticker, df in sp500_data.items():
        df = df[['Adj Close']].rename(columns={'Adj Close': ticker})
        if combined_data.empty:
            combined_data = df
        else:
            combined_data = combined_data.join(df, how='outer')

    combined_data = combined_data.sort_index()
    combined_data = combined_data.ffill()
    combined_data = combined_data.dropna(how='all')
    combined_data.index = pd.to_datetime(combined_data.index)

    for order in orders:
        order['date'] = pd.to_datetime(order['date'])

    backtest_engine = EquityBacktestEngine(initial_cash=50_000)
    backtest_results = backtest_engine.run_backtest(orders, combined_data)

    portfolio_values = backtest_results["portfolio_values"]["Portfolio Value"]
    returns = portfolio_values.pct_change().dropna()

    metrics_calculator = ExtendedMetrics()
    benchmark_data = sp500_data['SPY']['Adj Close']
    benchmark_returns = benchmark_data.pct_change().dropna()
    metrics = metrics_calculator.calculate(portfolio_values, returns, benchmark_returns)
    print("Backtest Metrics:", metrics)

    metrics_calculator.plot_returns(returns)
    
if __name__ == "__main__":
    main()