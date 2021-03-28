import datetime as dt
import os
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from util import get_data, plot_data


def author():
    return 'nriojas3'


def calculate_std(df, lookback):
    return df.rolling(window=lookback, min_periods=lookback).std()


def calculate_sma(df, lookback):
    sma = df.rolling(window=lookback, min_periods=lookback).mean()
    return sma


def calculate_price_per_sma(df, sma):
    return df / sma


def calculate_momentum(df, lookback):
    return df.pct_change(lookback)


def calculate_ema(df, lookback):
    return df.ewm(span=lookback).mean()


def calculate_BB_data(df, lookback, sma, std):
    top_band = sma + (2 * std)
    bottom_band = sma - (2 * std)
    return (df - bottom_band) / (top_band - bottom_band), top_band, bottom_band


def plot_std(df, std):
    fig, ax = plt.subplots(2, figsize=(8, 6))
    fig.suptitle('JPM 14 Day Rolling Standard Deviation vs Date')
    plt.subplot(2, 1, 1)
    plt.ylabel("Normalized Price")
    plt.plot(df, color='blue', label="price")
    plt.grid()
    plt.legend(["Price"])

    plt.subplot(2, 1, 2)
    plt.ylabel("Standard Deviation")
    plt.plot(std, color='maroon', label="std")
    plt.grid()
    plt.tick_params(axis='x', rotation=20)
    plt.legend(["Std"])

    for ax in fig.get_axes():
        ax.label_outer()
    plt.xlabel("Date")

    fig.text(0.5, 0.5, 'Property of Nathan Riojas',
             fontsize=30, color='gray',
             ha='center', va='center', rotation='30', alpha=0.36)
    plt.savefig("std.png")


def plot_BB(df, top, bottom):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(top, color='red')
    ax.plot(bottom, color='red')
    ax.plot(df, color='teal')
    ax.grid()
    ax.tick_params(axis='x', rotation=20)
    plt.xlabel("Date")
    plt.ylabel("Normalized Price")
    plt.legend(["BB Upper", "BB Lower", "Price"])
    plt.title("JPM Normalized Stock Price vs Date with Bollinger Bands")
    # add watermark
    fig.text(0.5, 0.5, 'Property of Nathan Riojas',
             fontsize=30, color='gray',
             ha='center', va='center', rotation='30', alpha=0.36)
    plt.savefig("bollinger.png")


def plot_BBP(bbp):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(bbp, color ='darkgreen')
    ax.grid()
    ax.tick_params(axis='x', rotation=20)
    plt.xlabel("Date")
    plt.ylabel("Bollinger Band Percentage")
    plt.legend(["BBP"])
    plt.title("JPM Bollinger Band Percentage vs Date")
    # add watermark
    fig.text(0.5, 0.5, 'Property of Nathan Riojas',
             fontsize=30, color='gray',
             ha='center', va='center', rotation='30', alpha=0.36)
    plt.savefig("bbp.png")


def plot_sma(df, sma, sma_per_price):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df, color='mediumpurple', alpha=0.75)
    ax.plot(sma, color='red')
    ax.plot(sma_per_price, color='darkgreen', alpha=0.7)
    ax.grid()
    ax.tick_params(axis='x', rotation=20)
    plt.xlabel("Date")
    plt.ylabel("Normalized Price")
    plt.legend(["Price", "SMA", "Price / SMA"])
    plt.title("JPM Rolling 14 Day Simple Moving Average vs Date")
    fig.text(0.5, 0.5, 'Property of Nathan Riojas',
             fontsize=30, color='gray',
             ha='center', va='center', rotation='30', alpha=0.36)
    plt.savefig("sma.png")


def plot_momentum(df, momentum):
    fig, ax = plt.subplots(2, figsize=(8, 6))
    fig.suptitle('JPM Rolling 14 Day Momentum vs Date')
    plt.subplot(2, 1, 1)
    plt.ylabel("JPM Normalized Price")
    plt.plot(df, color='green', label="price")
    plt.grid()
    plt.legend(["Price"])

    plt.subplot(2, 1, 2)
    plt.ylabel("Momentum")
    plt.plot(momentum, color='teal', label="std")
    plt.grid()
    plt.tick_params(axis='x', rotation=20)
    plt.legend(["Momentum"])

    for ax in fig.get_axes():
        ax.label_outer()
    plt.xlabel("Date")

    fig.text(0.5, 0.5, 'Property of Nathan Riojas',
             fontsize=30, color='gray',
             ha='center', va='center', rotation='30', alpha=0.36)
    plt.savefig("momentum.png")


def plot_ema(df, ema, sma):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(ema, color='navy')
    ax.plot(df, color='r', alpha=0.4)
    ax.plot(sma, color='g', alpha=.7)
    ax.tick_params(axis='x', rotation=20)
    ax.grid()
    plt.xlabel("Date")
    plt.ylabel("Exponential Moving Average")
    plt.legend(["EMA", 'SMA', 'Price'])
    plt.title("JPM Rolling 14 Day Exponential Moving Average vs Date")
    fig.text(0.5, 0.5, 'Property of Nathan Riojas',
             fontsize=30, color='gray',
             ha='center', va='center', rotation='30', alpha=0.36)
    plt.savefig("ema.png")


def generate_indicators(stock='JPM', lookback=14, start_date=dt.datetime(2008, 1, 1), end_date=dt.datetime(2009, 12, 31)):

    dates = pd.date_range(start_date, end_date)
    df = get_data([stock], dates, True, colname='Adj Close').drop(columns=['SPY'])
    df_norm = df / df.iloc[0, :]

    # generate dataframes from technical indicators
    std = calculate_std(df_norm, lookback)
    sma = calculate_sma(df_norm, lookback)
    price_per_sma = calculate_price_per_sma(df_norm, sma)
    momentum = calculate_momentum(df_norm, lookback)
    ema = calculate_ema(df_norm, lookback)
    bbp, top_band, bottom_band = calculate_BB_data(df_norm, lookback, sma, std)

    # generate plots
    plot_BB(df_norm, top_band, bottom_band)
    plot_BBP(bbp)
    plot_std(df_norm, std)
    plot_sma(df_norm, sma, price_per_sma)
    plot_momentum(df_norm, momentum)
    plot_ema(df_norm, ema, sma)


if __name__ == "__main__":
    generate_indicators()

