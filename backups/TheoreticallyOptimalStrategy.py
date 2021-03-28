import datetime as dt
import os
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import marketsim as ms
from util import get_data, plot_data


def author():
	return 'nriojas3'


def normalize_df(df):
	return df / df.iloc[0, :]


def generate_plot(optimal, benchmark):
	fig, ax = plt.subplots(figsize=(8, 6))
	ax.plot(normalize_df(optimal), color='red')
	ax.plot(normalize_df(benchmark), color='green')
	ax.tick_params(axis='x', rotation=20)
	ax.grid()
	plt.xlabel("Date")
	plt.ylabel("Normalized Portfolio Value")
	plt.legend(['Optimal', "Benchmark"])
	plt.title("JPM Theoretically Optimal Trading Strategy vs Benchmark Strategy")
	fig.text(0.5, 0.5, 'Property of Nathan Riojas',
			 fontsize=30, color='gray',
			 ha='center', va='center', rotation='30', alpha=0.36)
	plt.savefig("optimal.png")


def testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
	commission, impact = 0, 0
	dates = pd.date_range(sd, ed)
	dfPrice = get_data([symbol], dates, True, colname='Adj Close').drop(columns=['SPY'])
	dfPrice.sort_index()
	dfTrades = dfPrice.copy()
	dfTrades = ((dfTrades.shift(-1) - dfTrades) / abs(dfTrades.shift(-1) - dfTrades) * 1000).fillna(0)
	dfTradesPast = (dfTrades.shift(1)).fillna(0)
	optimized_orders = dfTrades - dfTradesPast
	benchmark_orders = optimized_orders * 0
	benchmark_orders.iloc[0][0] = 1000
	optimal_portfolio = ms.compute_portvals(optimized_orders, sd, ed, symbol, sv, commission, impact)
	benchmark_portfolio = ms.compute_portvals(benchmark_orders, sd, ed, symbol, sv, commission, impact)
	generate_plot(optimal_portfolio, benchmark_portfolio)
	return optimized_orders


if __name__ == "__main__":

	df_trades = testPolicy()

