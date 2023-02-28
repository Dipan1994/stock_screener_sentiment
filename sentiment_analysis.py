# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 01:49:23 2023

@author: HP
"""

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

text = 'I am not good at sentiment analysis. I just started learning.'

#tokenization
from nltk.tokenize import word_tokenize
tokens = word_tokenize(text)
print(tokens)

#lemmatization
from nltk.stem import WordNetLemmatizer
lem = WordNetLemmatizer()
tokens = [lem.lemmatize(word) for word in tokens]

#Steming
from nltk.stem import PorterStemmer
tokens = word_tokenize(text.lower())
ps = PorterStemmer()
tokens = [ps.stem(word) for word in tokens]

#Stop Words
import nltk
stopwords = nltk.corpus.stopwords.words("English")
print(stopwords)
tokens_new = [j for j in tokens if j not in stopwords]

########################
#Lexicon based approach

#Vader
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
analyser.polarity_scores("This is great.")
analyser.polarity_scores("Piss off.")

#textblob
from textblob import TextBlob
TextBlob('His').sentiment
TextBlob('average').sentiment






