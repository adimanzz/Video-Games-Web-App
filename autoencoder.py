# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 10:21:01 2020

@author: aditya
"""

import pandas as pd
pd.options.display.max_columns = 500
pd.options.display.max_rows = 500
import numpy as np
import pickle

import keras
import keras.backend as K
from keras import Input
from keras.models import Model, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

users_df = pd.read_pickle('user_df.pkl')
users_df.head()

games = users_df.columns.tolist()
gametoix = {}
ixtogame = {}

ix = 0
for game in games:
    gametoix[game] = ix
    ixtogame[ix] = game
    ix += 1

#pickle.dump(gametoix,open('game_index_dictionaries/gametoix.pkl','wb'))
#pickle.dump(ixtogame,open('game_index_dictionaries/ixtogame.pkl','wb'))

NUM_GAMES = len(users_df.columns)
MAX_LENGTH = len(users_df.columns)


class AE_Model():
    def __init__(self, NUM_GAMES):
        self.num_games = NUM_GAMES
        
    def forward(self):
        input_vec = Input(shape = (self.num_games, ))
        encoder = Dense(256, activation = 'sigmoid')(input_vec)
        
        decoder = Dense(256, activation = 'sigmoid')(encoder)
        output = Dense(self.num_games, activation = 'softmax')(decoder)
        
        model = Model(inputs = input_vec, outputs = output)
        return model

#Data for training

xtrain = users_df.iloc[:9044]
xtest = users_df.iloc[9044:]

#Model

model = AE_Model(NUM_GAMES)
model = model.forward()

optim = Adam(learning_rate = 0.01, decay = 0.005)

model.compile(optimizer = optim, loss='mse', metrics = ['accuracy'])

#model.fit(xtrain, xtrain,
          batch_size = 64,
          epochs = 200,
          shuffle = True,
          validation_data = (xtest,xtest))


#Test
model = load_model('Models/model2')
pred_games = []
vec = np.zeros((1,NUM_GAMES))
fav_games = ['Dota 2','Warframe','TERA','Grand Theft Auto V']    
seq = [gametoix[game] for game in fav_games if game in gametoix]
for i in seq:
    vec[0][i] = 1
 
pred = model.predict(vec)
scores = np.argsort(-pred)
for i in scores[0][:13]:
    pred_games.append(ixtogame[i])
pred_games = [x for x in pred_games if x not in fav_games]

model.fit(vec, vec,  batch_size = 1, epochs = 10, shuffle = True)    
    
# Save Model
#bs=64, epochs=100, lr=0.01, decay=0.005
#model.save('Models/model1') val_acc = 77%

#bs=64, epochs=200, lr=0.01, decay=0.005
#model.save('Models/model2') val_acc = 80.6%
