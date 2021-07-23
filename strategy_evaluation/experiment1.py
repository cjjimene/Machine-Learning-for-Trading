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


def run_experiment_one():
    # Setup
    symbol = "JPM"
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sv = 100000
    commission = 9.95
    impact = 0.005

    # Manual Strategy
    manual = ml.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
    manual_pv = ms.compute_portvals(
        manual,
        symbol=symbol,
        start_val=sv,
        commission=commission,
        impact=impact,
    )

    # Benchmark
    benchmark = manual.copy()
    benchmark["Order"] = 0
    benchmark["Order"][0] = 1000
    benchmark_pv = ms.compute_portvals(
        benchmark,
        symbol=symbol,
        start_val=sv,
        commission=commission,
        impact=impact,
    )

    # Strategy Learner
    learner = sl.StrategyLearner()
    learner.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
    strategy = learner.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
    strategy_pv = ms.compute_portvals(orders_df=strategy, start_val=sv, commission=commission, impact=impact)

    # Normalize
    strategy_pv = strategy_pv / strategy_pv[0]
    manual_pv = manual_pv / manual_pv[0]
    benchmark_pv = benchmark_pv / benchmark_pv[0]

    # Experiment 1 Plot
    benchmark_pv.plot(label="Benchmark", legend=True, color="green")
    manual_pv.plot(label="Manual Strategy", legend=True, color="red")
    strategy_pv.plot(label="Strategy Learner", legend=True)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.title("In Sample Portfolio Values: Strategy Comparison")
    plt.savefig("experiment1.png")
    plt.close()

    # Manual Strategy Plots
    # Manual vs Benchmark
    benchmark_pv.plot(label="Benchmark", legend=True, color="green")
    manual_pv.plot(label="Manual Strategy", legend=True, color="red")
    short = manual.loc[manual.Order < 0].index.to_list()
    long = manual.loc[manual.Order > 0].index.to_list()
    xmin, xmax, ymin, ymax = plt.axis()
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.title("Benchmark vs Manual Strategy Portfolio Values: In-Sample")
    plt.vlines(short, ymin=ymin, ymax=ymax, colors="black")
    plt.vlines(long, ymin=ymin, ymax=ymax, colors="blue")
    plt.savefig("manualis.png")
    plt.close()


    # Manual Strategy Out of Sample
    sd = dt.datetime(2010, 1, 1)
    ed = dt.datetime(2011, 12, 31)
    manual = ml.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
    manual_pv = ms.compute_portvals(
        manual,
        symbol=symbol,
        start_val=sv,
        commission=commission,
        impact=impact,
    )

    # Benchmark
    benchmark = manual.copy()
    benchmark["Order"] = 0
    benchmark["Order"][0] = 1000
    benchmark_pv = ms.compute_portvals(
        benchmark,
        symbol=symbol,
        start_val=sv,
        commission=commission,
        impact=impact,
    )
    # Normalize
    manual_pv = manual_pv / manual_pv[0]
    benchmark_pv = benchmark_pv / benchmark_pv[0]

    # Experiment 1 Plot
    benchmark_pv.plot(label="Benchmark", legend=True, color="green")
    manual_pv.plot(label="Manual Strategy", legend=True, color="red")
    short = manual.loc[manual.Order < 0].index.to_list()
    long = manual.loc[manual.Order > 0].index.to_list()
    xmin, xmax, ymin, ymax = plt.axis()
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.title("Benchmark vs Manual Strategy Portfolio Values: Out of Sample")
    plt.vlines(short, ymin=ymin, ymax=ymax, colors="black")
    plt.vlines(long, ymin=ymin, ymax=ymax, colors="blue")
    plt.savefig("manualoos.png")
    plt.close()


if __name__ == "__main__":
    pass
