# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 17:38:20 2023

@author: HP
"""

import yfinance as yf
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from stocktrends import Renko


def ATR(DF,n=14):
    df = DF.copy()
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = df['High'] - df['Adj Close'].shift(1)
    df['L-PC'] = df['Low'] - df['Adj Close'].shift(1)
    df['TR'] = df[['H-L','H-PC','L-PC']].max(axis=1, skipna = False)
    df['ATR'] = df['TR'].ewm(com = n,min_periods = n).mean(skipna=True)
    return df['ATR']

def CAGR(DF,n0=78*252):    
    df = DF.copy()
    if 'return' not in df.columns.to_list():
        df['return'] = df['Adj Close'].pct_change()
    df['cum_return'] = (1+df['return']).cumprod()   
    n = len(df)/n0
    CAGR = (df['cum_return'].tolist())[-1]**(1/n)-1
    return CAGR

def volatility(DF,n0=78*252):
    df = DF.copy()
    if 'return' not in df.columns.to_list():
        df['return'] = df['Adj Close'].pct_change()
    return df['return'].std()*np.sqrt(n0)

def Sharpe(DF,rf=.03):
    return (CAGR(DF)-rf)/volatility(DF)

def Sortino(DF,rf=.03):
    df = DF.copy()
    if 'return' not in df.columns.to_list():
        df['return'] = df['Adj Close'].pct_change()
    neg_return = np.where(df['return']>0,0,df['return'])
    neg_vol = pd.Series(neg_return[neg_return!=0]).std() * np.sqrt(252)
    return (CAGR(DF)-rf)/neg_vol
    
def max_drawdown(DF):
    df = DF.copy()
    if 'return' not in df.columns.to_list():
        df['return'] = df['Adj Close'].pct_change()
    df['cum_return'] = (1+df['return']).cumprod()
    df['cum_roll_max'] = df['cum_return'].cummax()
    df['drawdown'] = df['cum_roll_max'] - df['cum_return']
    return (df['drawdown']/df['cum_roll_max']).max()

def Calmar(DF):
    return (CAGR(DF)/max_drawdown(DF))
    
tickers = ["MSFT","AAPL","META","AMZN","INTC", "CSCO","VZ","IBM","TSLA","AMD"]

ohlc = {}

ticker_signal = {}
ticker_ret = {}

for ticker in tickers:
    ohlc[ticker] = yf.download(ticker,period='1mo',interval='5m')
    ohlc[ticker].dropna(inplace=True,how="all") 
    ohlc[ticker]['ATR'] = ATR(ohlc[ticker],20)
    ohlc[ticker]['roll_max_cp'] = ohlc[ticker]['High'].rolling(20).max()
    ohlc[ticker]['roll_min_cp'] = ohlc[ticker]['Low'].rolling(20).min()
    ohlc[ticker]['roll_max_vol'] = ohlc[ticker]['Volume'].rolling(20).max()
    ticker_signal[ticker] = ''
    ticker_ret[ticker] = []
    
    
for ticker in tickers:
    print(ticker)
    for i in range(len(ohlc[ticker])):
        if ticker_signal[ticker] == '':
            ticker_ret[ticker].append(0)
            if ohlc[ticker]['High'][i]>=ohlc[ticker]['roll_max_cp'][i] and \
                ohlc[ticker]['Volume'][i]>=1.5*ohlc[ticker]['roll_max_vol'][i-1]:
                ticker_signal[ticker] = 'Buy'
            elif ohlc[ticker]['Low'][i]<=ohlc[ticker]['roll_min_cp'][i] and \
                ohlc[ticker]['Volume'][i]>=1.5*ohlc[ticker]['roll_max_vol'][i-1]:
                ticker_signal[ticker] = 'Sell'
                
        elif ticker_signal[ticker] == 'Buy':
            if ohlc[ticker]['Low'][i] < ohlc[ticker]['Close'][i-1] - ohlc[ticker]['ATR'][i-1]:
                ticker_signal[ticker] = ""
                ticker_ret[ticker].append(((ohlc[ticker]['Close'][i-1] - ohlc[ticker]['ATR'][i-1])/ohlc[ticker]['Close'][i-1])-1)
            elif ohlc[ticker]['Low'][i]<=ohlc[ticker]['roll_min_cp'][i] and \
                ohlc[ticker]['Volume'][i]>=1.5*ohlc[ticker]['roll_max_vol'][i-1]:
                ticker_signal[ticker] = "Sell"
                ticker_ret[ticker].append(((ohlc[ticker]['Close'][i])/ohlc[ticker]['Close'][i-1])-1)    
            else:
                ticker_ret[ticker].append(((ohlc[ticker]['Close'][i])/ohlc[ticker]['Close'][i-1])-1)    
                
        elif ticker_signal[ticker] == 'Sell':
            if ohlc[ticker]['High'][i] > ohlc[ticker]['Close'][i-1] + ohlc[ticker]['ATR'][i-1]:
                ticker_signal[ticker] = ""
                ticker_ret[ticker].append(((ohlc[ticker]['Close'][i-1] + ohlc[ticker]['ATR'][i-1])/ohlc[ticker]['Close'][i-1])-1)
            elif ohlc[ticker]['High'][i]>=ohlc[ticker]['roll_max_cp'][i] and \
                ohlc[ticker]['Volume'][i]>=1.5*ohlc[ticker]['roll_max_vol'][i-1]:
                ticker_signal[ticker] = 'Buy'
                ticker_ret[ticker].append(ohlc[ticker]['Close'][i-1]/ohlc[ticker]['Close'][i]-1)
            else:
                ticker_ret[ticker].append(ohlc[ticker]['Close'][i-1]/ohlc[ticker]['Close'][i]-1)
                
    ohlc[ticker]['return'] = np.array(ticker_ret[ticker])
    
strategy_df = pd.DataFrame()
for ticker in tickers:
    strategy_df[ticker] = ohlc[ticker]["return"]
strategy_df["return"] = strategy_df.mean(axis=1)  

CAGR(strategy_df)
Sharpe(strategy_df,0.025)
max_drawdown(strategy_df)  

# vizualization of strategy return
(1+strategy_df["return"]).cumprod().plot()