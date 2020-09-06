# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 19:32:06 2020

@author: aditya
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from clean_text import CleanText
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import keras
from keras.models import Model, Sequential, load_model
from keras.layers import LSTM
from keras.layers import Flatten, Dense, Dropout, Activation, Input ,BatchNormalization
from keras.optimizers import Adam
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from attention import Attention
from model import SentimentNet


tweets = pd.read_csv('data/tweets_1600000.csv', encoding = 'latin')
tweets.columns = ['sentiment','id','time','query','name','text']
tweets = tweets[['text','sentiment']]
tweets = tweets.sample(frac=0.1, random_state=8)
'''tweets_df = pd.read_csv('data/tweets_1600000.csv', encoding = 'latin')
tweets_df.columns = ['sentiment','id','time','query','name','tweet']
tweets_df = tweets_df[['tweet','sentiment']]
tweets_df['clean_tweet'] = clean.clean(tweets_df['tweet'])
tweets_df['clean_tweet'] = tweets_df['clean_tweet'].apply(lambda x: clean.tokenize(x))
docs2 = tweets_df['clean_tweet']
t2 = Tokenizer()
t2.fit_on_texts(docs2)
vocab_size2 = len(t2.word_index) + 1
#encode the documents
encoded_docs2 = t2.texts_to_sequences(docs2)'''

clean = CleanText()

#clean() removes urls, emoticons and hashtags
tweets['text'] = clean.clean(tweets['text'])
#remove punctuations, stopwords, lemmatize and splits the sentences into tokens
tweets['text'] = tweets['text'].apply(lambda x: clean.tokenize(x))

docs = tweets['text']
labels = tweets['sentiment']
le = LabelEncoder()
labels_en = le.fit_transform(labels) #Negative: 0, Positive: 1
labels_en = keras.utils.to_categorical(np.asarray(labels_en))

#tokenizer
t = Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1
#encode the documents
encoded_docs = t.texts_to_sequences(docs)

# Function to find length of the Longest Sentence
def maxLength(sentence):
    max_length = 0
    for i in sentence:
        length = len(i)
        if length > max_length:
            max_length = length
    return max_length

max_length = 40

#pad docs to max length
padded_docs = pad_sequences(encoded_docs, maxlen = max_length, padding = 'post')

# Train/Dev/Test set: 80/10/10 split

xtrain, xdev, ytrain, ydev = train_test_split(padded_docs, labels_en, test_size = 0.0125, random_state = 8)

# xtest and ytest is our Unseen Data which will be used to get an Unbiased Evaluation of the Model
xdev, xtest, ydev, ytest = train_test_split(xdev, ydev, test_size = 0.5, random_state = 8)


#load embedding into memory
embeddings_index = dict()
f = open('D:/Downloads/Data Science/Glove/glove.6B.300d.txt', encoding = 'utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype = 'float32')
    embeddings_index[word] = coefs
f.close()    

embedding_dim = 300

# Weight matrix for words in training
embedding_matrix = np.zeros((vocab_size , embedding_dim))    
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Model
model = SentimentNet(max_length, vocab_size, embedding_dim, num_classes = 2)
model = model.forward()

model.layers[1].set_weights([embedding_matrix])
model.layers[1].trainable = False

opt = Adam(learning_rate = 0.01, beta_1 = 0.9, beta_2 = 0.999, decay = 0.005)
model.compile(loss = 'categorical_crossentropy',
              optimizer = opt,
              metrics = ['accuracy'])

checkpoint = ModelCheckpoint('Models/model_mil-{epoch:03d}-{accuracy:03f}-{val_accuracy:03f}.h5', verbose=1, monitor='val_accuracy',save_best_only=True, mode='auto') 

history = model.fit(xtrain, ytrain,
                    batch_size = 32,
                    epochs = 20,
                    shuffle = True,
                    workers = 12,
                    callbacks=[checkpoint],
                    validation_data = (xdev, ydev))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#Save 

#bs=32, epoch=20, lr=0.01, d=0.005, rd=0.25, d=0.5

model_test = load_model('Models/model_sentiment.h5',custom_objects = {'Attention':Attention})
