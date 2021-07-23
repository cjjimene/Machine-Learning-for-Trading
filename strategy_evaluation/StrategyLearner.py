""""""
"""  		  	   		   	 			  		 			 	 	 		 		 	
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
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

import BagLearner as bl
import RTLearner as rt
import indicators as ind
import pandas as pd
import util


class StrategyLearner(object):
    """
      A strategy learner that can learn a trading policy using the same indicators
      used in ManualStrategy.

      :param verbose: If “verbose” is True, your code can print out
      information for debugging.
          If verbose = False your code should not generate ANY output.
      :type verbose: bool
      :param impact: The market impact of each transaction, defaults to 0.0
      :type impact: float
      :param commission: The commission amount charged, defaults to 0.0
      :type commission: float
      """

    # constructor
    def __init__(self, verbose=False, impact=0.005, commission=9.95):
        """
            Constructor method
            """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": 5}, bags=40, boost=False, verbose=False)

    # this method should create a QLearner, and train it for trading
    def add_evidence(
            self,
            symbol="JPM",
            sd=dt.datetime(2008, 1, 1),
            ed=dt.datetime(2009, 12, 31),
            sv=100000,
    ):
        """
            Trains your strategy learner over a given time frame.

            :param symbol: The stock symbol to train on
            :type symbol: str
            :param sd: A datetime object that represents the start date, defaults to
            1/1/2008
            :type sd: datetime
            :param ed: A datetime object that represents the end date, defaults to
            1/1/2009
            :type ed: datetime
            :param sv: The starting value of the portfolio
            :type sv: int
            """

        # add your code to do learning here
        # Lookback windows for SMA based indicators.
        lookback = 20
        # Lookback well before start date so data is not lost due to rolling means.
        sd1 = sd - dt.timedelta(days=lookback * 10)
        ed1 = ed + dt.timedelta(days=lookback * 2)
        dates = pd.date_range(sd1, ed1)
        prices = util.get_data([symbol], dates)
        # Remove SPY because it is not needed.
        prices.drop(columns=["SPY"], inplace=True)
        # Sort index and normalize prices.
        prices.sort_index(inplace=True)
        # Get indicators, including lagged versions
        prices["PSMA"] = ind.get_price_sma(prices[symbol], lookback)
        prices["PSMA_1"] = prices["PSMA"].shift(1)
        prices["BBP"] = ind.get_bollinger_band_percentage(prices[symbol], lookback)
        prices["CROSS"] = ind.get_golden_cross_ratio(prices[symbol], lookback)
        prices["CROSS_1"] = prices["CROSS"].shift(1)
        # Get future 5 day returns
        prices["RETURNS"] = prices.loc[:, symbol].shift(-5) / prices.loc[:, symbol] - 1
        # Limit dataframe back to originally desired timeframe and normalize.
        prices = prices.loc[(prices.index >= sd) & (prices.index <= ed)]
        prices.drop(columns=[symbol], inplace=True)
        # Drop nulls in case there is no additional data to fetch.
        prices.dropna(inplace=True)
        # Set up trade rules.
        prices["BUY"] = prices["RETURNS"] >= (.019 + self.impact)
        prices["SELL"] = prices["RETURNS"] < (-.019 - self.impact)
        prices["ORDER"] = 0
        prices.loc[prices["BUY"], "ORDER"] = 1
        prices.loc[prices["SELL"], "ORDER"] = -1
        self.X_train = prices[["PSMA", "PSMA_1", "BBP", "CROSS", "CROSS_1"]].to_numpy()
        self.y_train = prices[["ORDER"]].to_numpy()
        self.learner.add_evidence(self.X_train, self.y_train)

    # this method should use the existing policy and test it against new data
    def testPolicy(
            self,
            symbol="JPM",
            sd=dt.datetime(2010, 1, 1),
            ed=dt.datetime(2011, 12, 31),
            sv=100000,
    ):
        """
            Tests your learner using data outside of the training data

            :param symbol: The stock symbol that you trained on on
            :type symbol: str
            :param sd: A datetime object that represents the start date, defaults to
            1/1/2008
            :type sd: datetime
            :param ed: A datetime object that represents the end date, defaults to
            1/1/2009
            :type ed: datetime
            :param sv: The starting value of the portfolio
            :type sv: int
            :return: A DataFrame with values representing trades for each day. Legal
            values are +1000.0 indicating
                a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and
                0.0 indicating NOTHING.
                Values of +2000 and -2000 for trades are also legal when switching
                from long to short or short to
                long so long as net holdings are constrained to -1000, 0, and 1000.
            :rtype: pandas.DataFrame
            """

        # here we build a fake set of trades
        # Lookback windows for SMA based indicators.
        lookback = 20
        # Lookback well before start date so data is not lost due to rolling means.
        sd1 = sd - dt.timedelta(days=lookback * 10)
        dates = pd.date_range(sd1, ed)
        prices = util.get_data([symbol], dates)
        # Remove SPY because it is not needed.
        prices.drop(columns=["SPY"], inplace=True)
        # Sort index and normalize prices.
        prices.sort_index(inplace=True)
        # Get indicators, including lagged versions
        prices["PSMA"] = ind.get_price_sma(prices[symbol], lookback)
        prices["PSMA_1"] = prices["PSMA"].shift(1)
        prices["BBP"] = ind.get_bollinger_band_percentage(prices[symbol], lookback)
        prices["CROSS"] = ind.get_golden_cross_ratio(prices[symbol], lookback)
        prices["CROSS_1"] = prices["CROSS"].shift(1)
        # Limit dataframe back to originally desired timeframe and normalize.
        prices = prices.loc[(prices.index >= sd) & (prices.index <= ed)]
        prices.drop(columns=[symbol], inplace=True)
        # Query Random Forest.
        df_trades = pd.DataFrame(index=prices.index)
        df_trades["Order"] = self.learner.query(prices.to_numpy())[0]

        # Dict containing Orders to perform based on position.
        position = 0
        buy = {0: 1000, 1000: 0, -1000: 2000}
        sell = {0: -1000, 1000: -2000, -1000: 0}
        for i, Order in zip(df_trades.index, df_trades.Order):
            if df_trades.loc[i]["Order"] == 1:
                df_trades.loc[i, "Order"] = buy[position]
            elif df_trades.loc[i]["Order"] == -1:
                df_trades.loc[i, "Order"] = sell[position]
            position += df_trades["Order"][i]
            df_trades.loc[i, "Position"] = position
        return df_trades[["Order"]]


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "cjimenez36"  # replace tb34 with your Georgia Tech username.


if __name__ == "__main__":
    pass
