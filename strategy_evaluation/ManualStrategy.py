import datetime as dt

import indicators as ind
import marketsimcode as ms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import util


def author():
    """
      :return: The GT username of the student
      :rtype: str
      """
    return "cjimenez36"


def testPolicy(symbol="JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
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
    # Get indicators
    prices["PSMA"] = ind.get_price_sma(prices[symbol], lookback)
    prices["PSMA_1"] = prices["PSMA"].shift(1)
    prices["BBP"] = ind.get_bollinger_band_percentage(prices[symbol], lookback)
    prices["CROSS"] = ind.get_golden_cross_ratio(prices[symbol], lookback)
    prices["CROSS_1"] = prices["CROSS"].shift(1)
    # Limit dataframe back to originally desired timeframe and normalize.
    prices = prices.loc[prices.index >= sd]
    prices[symbol] = prices[symbol] / prices[symbol].iloc[0]
    # Set up trade rules.
    df_trades = pd.DataFrame()
    df_trades["Buy"] = ((prices.PSMA >= .85) & (prices.PSMA_1 < .85)) | (prices.BBP < 0) | (
            (prices.CROSS >= 1) & (prices.CROSS_1 < 1))
    df_trades["Sell"] = ((prices.PSMA < .85) & (prices.PSMA_1 >= .85)) | (prices.BBP > 1) | (
            (prices.CROSS < 1) & (prices.CROSS_1 >= 1))
    df_trades["Order"] = 0
    df_trades.loc[df_trades["Buy"], "Order"] = 1
    df_trades.loc[df_trades["Sell"], "Order"] = -1
    df_trades["Position"] = 0
    position = 0

    # Dict containing Orders to perform based on position.
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


if __name__ == "__main__":
    pass
