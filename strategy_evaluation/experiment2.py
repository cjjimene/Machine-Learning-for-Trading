import datetime as dt

import ManualStrategy as ml
import StrategyLearner as sl
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


def run_experiment_two():
    # Setup
    symbol = "JPM"
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sv = 100000
    commission = 0.0
    impact0 = 0.0
    impact005 = 0.005
    impact010 = 0.10

    # 0 Impact
    learner = sl.StrategyLearner(verbose=False, impact=impact0, commission=commission)
    learner.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
    strategy = learner.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
    strategy_zero = ms.compute_portvals(orders_df=strategy, start_val=sv, commission=commission, impact=impact0)

    # .005 Impact
    learner = sl.StrategyLearner(verbose=False, impact=impact005, commission=commission)
    learner.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
    strategy = learner.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
    strategy_five = ms.compute_portvals(orders_df=strategy, start_val=sv, commission=commission, impact=impact005)

    # .010 Impact
    learner = sl.StrategyLearner(verbose=False, impact=impact010, commission=commission)
    learner.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
    strategy = learner.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
    strategy_one = ms.compute_portvals(orders_df=strategy, start_val=sv, commission=commission, impact=impact010)

    # Normalize
    strategy_zero = strategy_zero / strategy_zero[0]
    strategy_five = strategy_five / strategy_five[0]
    strategy_one = strategy_one / strategy_one[0]

    # Plot portfolio value changes
    strategy_zero.plot(label="0.000 Impact", legend=True)
    strategy_five.plot(label="0.005 Impact", legend=True)
    strategy_one.plot(label="0.010 Impact", legend=True)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.title("In-sample Strategy Learner with Varying Impact: Portfolio Values")
    plt.savefig("experiment2_portval.png")
    plt.close()

    # Plot portfolio rolling stdev changes
    strategy_zero.rolling(window=20, center=False).std().plot(label="0.000 Impact", legend=True)
    strategy_five.rolling(window=20, center=False).std().plot(label="0.005 Impact", legend=True)
    strategy_one.rolling(window=20, center=False).std().plot(label="0.010 Impact", legend=True)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value Standard Deviation (Moving 20 Weeks)")
    plt.title("In-sample Strategy Learner with Varying Impact: Portfolio Value STDEV")
    plt.tight_layout()
    plt.savefig("experiment2_portval_std.png")
    plt.close()

if __name__ == "__main__":
    pass
