# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 18:03:59 2020

@author: aditya

"""


import keras
from keras import Input
from keras.layers.merge import add
from keras.models import Model 
from keras.layers import Bidirectional, LSTM, Dense, Dropout, Embedding
from attention import Attention


class SentimentNet():
    def __init__(self, max_length, vocab_size, embedding_dims, num_classes):
        self.MAX_LENGTH = max_length
        self.VOCAB_SIZE = vocab_size
        self.embedding_dims = embedding_dims
        self.num_classes = num_classes
        
    def forward(self):
        
        input_seq = Input(shape = (self.MAX_LENGTH,))
        seq = Embedding(self.VOCAB_SIZE, self.embedding_dims, mask_zero = True)(input_seq)
        seq = Dropout(0.3)(seq)
        seq = Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout = 0.4))(seq)
        seq = Attention()(seq)
        seq = Dense(200, activation = 'relu')(seq)
        outputs = Dense(self.num_classes, activation = 'softmax')(seq)
        
        model = Model(inputs = input_seq, outputs = outputs)
         
        return model
    

     
                    
        