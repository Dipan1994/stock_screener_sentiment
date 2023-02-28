# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 19:01:25 2023

@author: HP
"""

import yfinance as yf
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from stocktrends import Renko
import statsmodels.api as sm

def OBV(DF):
    df = DF.copy()
    if 'return' not in df.columns:
        df['return'] = df['Adj Close'].pct_change()
    df['direction'] = np.where(df['return']>=0,1,-1)
    df['direction'][0] = 0
    df['vol_adj'] = df['Volume'] * df['direction']
    df['obv'] = df['vol_adj'].cumsum()
    return df['obv']


def ATR(DF,n=14):
    df = DF.copy()
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = df['High'] - df['Adj Close'].shift(1)
    df['L-PC'] = df['Low'] - df['Adj Close'].shift(1)
    df['TR'] = df[['H-L','H-PC','L-PC']].max(axis=1, skipna = False)
    df['ATR'] = df['TR'].ewm(com = n,min_periods = n).mean(skipna=True)
    df2 = df.drop(['H-L','H-PC','L-PC'],axis=1)
    return df2

def slope(ser,n):
    slopes = [i*0 for i in range(n-1)]
    for i in range(n,len(ser)+1):
        y = ser[i-n:i]
        x = np.array(range(n))
        y_scaled = (y - y.min())/(y.max() - y.min())
        x_scaled = (x - x.min())/(x.max() - x.min())
        x_scaled = sm.add_constant(x_scaled)
        model = sm.OLS(y_scaled,x_scaled)
        results = model.fit()
        slopes.append(results.params[-1])
    slope_angle = (np.rad2deg(np.arctan(np.array(slopes))))
    return np.array(slope_angle)

def renko_DF(DF):
    df = DF.copy()
    df.reset_index(inplace=True)
    df = df.iloc[:,[0,1,2,3,4,5]]
    df.columns = ["date","open","high","low","close","volume"]
    df2 = Renko(df)
    df2.brick_size = max(0.5,round(ATR(DF,120)["ATR"][-1],0))
    renko_df = df2.get_ohlc_data()
    renko_df["bar_num"] = np.where(renko_df["uptrend"]==True,1,np.where(renko_df["uptrend"]==False,-1,0))
    for i in range(1,len(renko_df["bar_num"])):
        if renko_df["bar_num"][i]>0 and renko_df["bar_num"][i-1]>0:
            renko_df["bar_num"][i]+=renko_df["bar_num"][i-1]
        elif renko_df["bar_num"][i]<0 and renko_df["bar_num"][i-1]<0:
            renko_df["bar_num"][i]+=renko_df["bar_num"][i-1]
    renko_df.drop_duplicates(subset="date",keep="last",inplace=True)
    return renko_df


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
ohlc_renko = {}

ticker_signal = {}
ticker_ret = {}

for ticker in tickers:
    ohlc[ticker] = yf.download(ticker,period='1mo',interval='5m')
    ohlc[ticker].dropna(inplace=True,how="all") 
    renko = renko_DF(ohlc[ticker])
    renko.columns = ["Date","open","high","low","close","uptrend","bar_num"]
    ohlc[ticker]["Date"] = ohlc[ticker].index
    ohlc_renko[ticker] = ohlc[ticker].merge(renko.loc[:,["Date","bar_num"]],how="outer",on="Date")
    ohlc_renko[ticker]["bar_num"].fillna(method='ffill',inplace=True)
    ohlc_renko[ticker]["obv"]= OBV(ohlc_renko[ticker])
    ohlc_renko[ticker]["obv_slope"]= slope(ohlc_renko[ticker]["obv"],5)
    ticker_signal[ticker] = ""
    ticker_ret[ticker] = []
    
    
for ticker in tickers:
    print("calculating daily returns for ",ticker)
    for i in range(len(ohlc[ticker])):
        if ticker_signal[ticker] == "":
            ticker_ret[ticker].append(0)
            if ohlc_renko[ticker]["bar_num"][i]>=2 and ohlc_renko[ticker]["obv_slope"][i]>30:
                ticker_signal[ticker] = "Buy"
            elif ohlc_renko[ticker]["bar_num"][i]<=-2 and ohlc_renko[ticker]["obv_slope"][i]<-30:
                ticker_signal[ticker] = "Sell"
        
        elif ticker_signal[ticker] == "Buy":
            ticker_ret[ticker].append((ohlc_renko[ticker]["Adj Close"][i]/ohlc_renko[ticker]["Adj Close"][i-1])-1)
            if ohlc_renko[ticker]["bar_num"][i]<=-2 and ohlc_renko[ticker]["obv_slope"][i]<-30:
                ticker_signal[ticker] = "Sell"
            elif ohlc_renko[ticker]["bar_num"][i]<2:
                ticker_signal[ticker] = ""
                
        elif ticker_signal[ticker] == "Sell":
            ticker_ret[ticker].append((ohlc_renko[ticker]["Adj Close"][i-1]/ohlc_renko[ticker]["Adj Close"][i])-1)
            if ohlc_renko[ticker]["bar_num"][i]>=2 and ohlc_renko[ticker]["obv_slope"][i]>30:
                ticker_signal[ticker] = "Buy"
            elif ohlc_renko[ticker]["bar_num"][i]>-2:
                ticker_signal[ticker] = ""
    ohlc_renko[ticker]["return"] = np.array(ticker_ret[ticker])


#calculating overall strategy's KPIs
strategy_df = pd.DataFrame()
for ticker in tickers:
    strategy_df[ticker] = ohlc_renko[ticker]["return"]
strategy_df["return"] = strategy_df.mean(axis=1)
CAGR(strategy_df)
Sharpe(strategy_df,0.025)
max_drawdown(strategy_df)  

#visualizing strategy returns
(1+strategy_df["return"]).cumprod().plot()
