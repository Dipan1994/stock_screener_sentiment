# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 16:59:42 2023

@author: HP
"""

import yfinance as yf
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from stocktrends import Renko

tickers = ["MMM","AXP","T","BA","CAT","CSCO","KO", "XOM","GE","GS","HD",
           "IBM","INTC","JNJ","JPM","MCD","MRK","MSFT","NKE","PFE","PG","TRV",
           "UNH","VZ","V","WMT","DIS"]

ohlc_mon = {} # directory with ohlc value for each stock            
start = dt.datetime.today()-dt.timedelta(3650)
end = dt.datetime.today()

for ticker in tickers:
    ohlc_mon[ticker] = yf.download(ticker,start,end,interval='1mo')
    ohlc_mon[ticker].dropna(inplace=True,how="all")
 
DJI = yf.download("^DJI",dt.date.today()-dt.timedelta(3650),dt.date.today(),interval='1mo')

    
def CAGR(DF,n0=12):    
    df = DF.copy()
    if 'return' not in df.columns.to_list():
        df['return'] = df['Adj Close'].pct_change()
    df['cum_return'] = (1+df['return']).cumprod()   
    n = len(df)/n0
    CAGR = (df['cum_return'].tolist())[-1]**(1/n)-1
    return CAGR

def volatility(DF,n0=12):
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
    


#Buy and hold of index
DJI['return'] = DJI['Adj Close'].pct_change()
CAGR(DJI)
Sharpe(DJI)
max_drawdown(DJI)

#Rebalancing strategy
return_df = pd.DataFrame()

for ticker in tickers:
    print(ticker)
    ohlc_mon[ticker]['return'] = ohlc_mon[ticker]['Adj Close'].pct_change()
    return_df[ticker] = ohlc_mon[ticker]['return']

def portfolio(DF,m,x):
    df = DF.copy()
    portfolio = []
    monthly_ret = [0]

    for i in range(1,len(df)):
        if len(portfolio)>0:
            monthly_ret.append(df[portfolio].iloc[i,:].mean())
            bad_stocks = df[portfolio].iloc[i,:].sort_values(ascending=True)[:x].index.to_list()
            portfolio = [t for t in portfolio if t not in bad_stocks]
            
        fill = m - len(portfolio)
        #new_picks = df[[t for t in tickers if t not in portfolio]].iloc[i,:].sort_values(ascending=False)[:fill].index.to_list()
        new_picks = df.iloc[i,:].sort_values(ascending=False)[:fill].index.to_list()
        
        portfolio = portfolio + new_picks
        print(portfolio)
    monthly_ret_df = pd.DataFrame(np.array(monthly_ret),columns = ['return'])
    return monthly_ret_df

CAGR(portfolio(return_df, 6, 3))
Sharpe(portfolio(return_df, 6, 3))
max_drawdown(portfolio(return_df, 6, 3))


fig, ax = plt.subplots()
plt.plot((1+portfolio(return_df, 6, 3)).cumprod())
plt.plot((1+DJI['return'].reset_index(drop=True)).cumprod())
plt.title("Index Return vs Strategy Return")
plt.ylabel("cumulative return")
plt.xlabel("months")
ax.legend(["Strategy Return","Index Return"])