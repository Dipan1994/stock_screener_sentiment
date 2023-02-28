# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 02:31:51 2023

@author: HP
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup
import pandas as pd

url_list = []
date_time = []
news_text = []
headlines = []

for i in range(1,3):
    url = 'https://oilprice.com/Energy/Crude-Oil/Page-{}.html'.format(i)
    request = requests.get(url)
    soup = BeautifulSoup(request.text,'html.parser')
    for links in soup.find_all('div',{'class':'categoryArticle__content'}):
        for info in links.find_all('a'):
            if info.get('href') not in url_list:
                url_list.append(info.get('href'))
                print(info.get('href'))
                
                
for www in url_list:
    
    headlines.append(www.split("/")[-1].replace('-',' '))
    request = requests.get(www)
    soup = BeautifulSoup(request.text,'html.parser')
    
    for dates in soup.find_all('span',{'class':'article_byline'}):
        date_time.append(dates.text.split('_')[-1])
        
    temp = []
    for news in soup.find_all('p'):
        temp.append(news.text)
        
    
    for last in reversed(temp):
        if last.split(" ")[0]=="By" and last.split(" ")[-1] == "Oilprice.com":
            break
        elif last.split(" ")[0]=="By":
            break
    
    joined_text = ' '.join(temp[temp.index("More Info")+1: temp.index(last)])
    
    news_text.append(joined_text)
        
        
        
news_df = pd.DataFrame({'Date':date_time,
                        'Headline':headlines,
                        'News':news_text})

analyser = SentimentIntensityAnalyzer()

def comp_score(text):
    return analyser.polarity_scores(text)['compound']

news_df['sentiment'] = news_df['News'].apply(comp_score)


        
        
    