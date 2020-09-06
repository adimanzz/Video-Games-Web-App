# -*- coding: utf-8 -*-
"""
Created on Sat May 30 19:11:39 2020

@author: aditya
"""

import pandas as pd
import nltk
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import WordNetLemmatizer
import re
from bs4 import BeautifulSoup
import unicodedata

# Keeping '!' and '?' because they help emphasize a given sentiment 
punctuation = punctuation.replace('!','').replace('?','')


class CleanText():
    def __init__(self):
        self.stop_words = stopwords.words('english')
        self.custom = self.stop_words + list(punctuation)
        self.wordnet_lemmatizer = WordNetLemmatizer()
        self.contraction_mapping = eval(open('contraction_mapping.txt','r', encoding = 'utf8').read())
        self.emoticons = eval(open('emoticons.txt','r', encoding = 'utf8').read())  
        
    def tokenize(self, token):
        self.token = str(token)
        #self.token = self.token.lower()
        tokens = nltk.tokenize.word_tokenize(self.token)
        tokens = [t for t in tokens if len(t) > 2] #Remove single characters
        tokens = [self.wordnet_lemmatizer.lemmatize(t) for t in tokens] #Lemmatize words
        tokens = [t for t in tokens if t not in self.custom] #Remove Stopwords and Punctuation
        tokens = [t for t in tokens if not any(c.isdigit() for c in t)] #Remove digits
        return tokens
    
    def convert_emoticons(self, word):
        self.word = word
        for emot in self.emoticons:
            self.word = re.sub(u'('+emot+')', "_".join(self.emoticons[emot].replace(",","").split()), self.word)
        return self.word

    def clean(self, text):
        self.text = text
        text = self.text.apply(lambda x : BeautifulSoup(x, 'lxml').get_text()) #remove html encodings
        text = text.str.lower() #convert into lower case
        text = text.apply(lambda x : re.sub(r'rt @[A-Za-z0-9]+','',x)) #rt stands for retweet remove rt @anytext
        text = text.apply(lambda x : re.sub(r'@[A-Za-z0-9]+','',x)) #get rid of all the @tags
        text = text.apply(lambda x : re.sub(r'{link}','',x)) #removing {link}
        text = text.apply(lambda x : re.sub(r'b/c','because',x)) #replacing b/c with because
        text = text.apply(lambda x : re.sub(r'w/','with',x)) #replacing w/ with 'with'
        text = text.apply(lambda x : unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore')) #removing accented characters
        text = text.apply(lambda x :' '.join([self.contraction_mapping[t] if t in self.contraction_mapping else t for t in x.split(" ")]))#contraction mapping
        text = text.apply(lambda x : re.sub(r'http?://[A-Za-z0-9./]+','',x)) #removing all the unneccesary links
        text = text.apply(lambda x : re.sub(r'https?://[A-Za-z0-9./]+','',x)) #removing all the unneccesary links
        text = text.apply(lambda x : re.sub(r'bit.ly/?[A-Za-z0-9./]+','',x)) #removing all the unneccesary links
        text = text.apply(lambda x : re.sub("www\.\S+",'',x)) #removing all the unneccesary links
        text = text.apply(lambda x : re.sub("\.com",' ',x)) 
        text = text.apply(lambda x : re.sub("\)",' ',x) if re.search("[A-Z a-z 0-9]\)",x) else x)
        #few ')' were unnecessarily treated as emoticons hence removing them
        text = text.apply(lambda x: self.convert_emoticons(str(x))) #converting emoticons to text
        text = text.str.lower()
        text = text.apply(lambda x : re.sub("#[A-Za-z0-9]+", " ",x)) #remove hash
        return text
    
    


