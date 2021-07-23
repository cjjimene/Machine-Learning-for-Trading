""""""  		  	   		   	 			  		 			 	 	 		 		 	
"""MC2-P1: Market simulator.  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		   	 			  		 			 	 	 		 		 	
Atlanta, Georgia 30332  		  	   		   	 			  		 			 	 	 		 		 	
All Rights Reserved  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
Template code for CS 4646/7646  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		   	 			  		 			 	 	 		 		 	
works, including solutions to the projects assigned in this course. Students  		  	   		   	 			  		 			 	 	 		 		 	
and other users of this template code are advised not to share it with others  		  	   		   	 			  		 			 	 	 		 		 	
or to make it available on publicly viewable websites including repositories  		  	   		   	 			  		 			 	 	 		 		 	
such as github and gitlab.  This copyright statement should not be removed  		  	   		   	 			  		 			 	 	 		 		 	
or edited.  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
We do grant permission to share solutions privately with non-students such  		  	   		   	 			  		 			 	 	 		 		 	
as potential employers. However, sharing with other current or future  		  	   		   	 			  		 			 	 	 		 		 	
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		   	 			  		 			 	 	 		 		 	
GT honor code violation.  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
-----do not edit anything above this line---  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
Student Name: Christopher Jimenez (replace with your name)  		  	   		   	 			  		 			 	 	 		 		 	
GT User ID: cjimenez36 (replace with your User ID)  		  	   		   	 			  		 			 	 	 		 		 	
GT ID: 903650706 (replace with your GT ID)  	  	   		   	 			  		 			 	 	 		 		 	
"""
import datetime as dt
import os

import numpy as np
import pandas as pd
from util import get_data, plot_data


def compute_portvals(
        orders_file="./orders/orders.csv",
        start_val=1000000,
        commission=9.95,
        impact=0.005,
):
    """
      Computes the portfolio values.

      :param orders_file: Path of the order file or the file object
      :type orders_file: str or file object
      :param start_val: The starting value of the portfolio
      :type start_val: int
      :param commission: The fixed amount in dollars charged for each transaction
      (both entry and exit)
      :type commission: float
      :param impact: The amount the price moves against the trader compared to the
      historical data at each transaction
      :type impact: float
      :return: the result (portvals) as a single-column dataframe, containing the
      value of the portfolio for each trading
      day in the first column from start_date to end_date, inclusive.
      :rtype: pandas.DataFrame
      """
    # Get orders, sort ascending so proper sequence is maintained.
    orders = pd.read_csv(orders_file, index_col="Date", parse_dates=True, na_values=["nan"])
    orders.sort_index(inplace=True, ascending=True)
    # Get list of unique symbols, and start and end dates contained within orders to get price data.
    symbols = list(orders.Symbol.unique())
    start_date = orders.index.min().to_pydatetime()
    end_date = orders.index.max().to_pydatetime()
    # Keep SPY so only valid trading days are kept.
    prices = get_data(symbols, pd.date_range(start_date, end_date))
    # Remove SPY now that it's no longer needed.
    prices.drop(columns=["SPY"], inplace=True)
    # Fill prices forward then backwards.
    prices.fillna(method="ffill", inplace=True)
    prices.fillna(method="bfill", inplace=True)
    prices["Cash"] = 1.0
    # Remove orders that did not occur on a valid trading day using `prices` dataframe.
    orders = orders[orders.index.isin(prices.index)]
    # Initialize `trades` dataframe.
    trades = prices.copy()
    # Set all values to zero using methodology found at StackOverflow:
    # https://stackoverflow.com/questions/42636765/how-to-set-all-the-values-of-an-existing-pandas-dataframe-to-zero
    for col in trades.columns:
        trades[col].values[:] = 0
    for i, sym, order, share in zip(orders.index, orders.Symbol, orders.Order, orders.Shares):
        trades.loc[i, sym] += share * (1 if order == "BUY" else -1)
        trades.loc[i, "Cash"] += share * (-(1 + impact) if order == "BUY" else (1 - impact)) * prices.loc[i, sym] - commission
    holdings = trades.copy()
    holdings.iloc[0, -1] += start_val
    for day in range(1, len(holdings)):
        holdings.iloc[day, :] += holdings.iloc[day - 1:day, :].sum()
    print(holdings)
    port_vals = np.sum(holdings * prices, axis=1)
    return port_vals


def test_code():
    """
      Helper function to test code
      """
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders2.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats.
    start_date = portvals.index.min().to_pydatetime()
    end_date = portvals.index.max().to_pydatetime()

    def daily_returns(p_value):
        daily_ret = (p_value / p_value.shift(1)) - 1
        return daily_ret[1:]
    
    def sharpe(daily_ret, daily_rf=0, k=252):
        return (np.mean(daily_ret - daily_rf) / np.std(daily_ret - daily_rf, ddof=1)) * (k ** .5)

    daily_ret = daily_returns(portvals)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [
        (portvals[-1] / portvals[0]) - 1,
        daily_ret.mean(),
        daily_ret.std(),
        sharpe(daily_ret),
    ]

    # Compare portfolio against $SPX
    print(f"Date Range: {start_date} to {end_date}")
    print()
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
    print()
    print(f"Cumulative Return of Fund: {cum_ret}")
    print()
    print(f"Standard Deviation of Fund: {std_daily_ret}")
    print()
    print(f"Average Daily Return of Fund: {avg_daily_ret}")
    print()
    print(f"Final Portfolio Value: {portvals[-1]}")


def author():
    """
      :return: The GT username of the student
      :rtype: str
      """
    return "cjimenez36"


if __name__ == "__main__":
    test_code()
    print(author())
