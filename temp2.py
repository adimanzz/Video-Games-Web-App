# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 12:10:11 2020

@author: adity
"""
import streamlit as st
import pandas as pd
import numpy as np

#import tensorflow as tf
from keras.models import load_model
import keras.backend.tensorflow_backend as tb
#tb._SYMBOLIC_SCOPE.value = True

@st.cache(persist=True)
def load_data():
    gametoix = pd.read_pickle('game_index_dictionaries/gametoix.pkl')
    
    ixtogame = pd.read_pickle('game_index_dictionaries/ixtogame.pkl')
    
    model = load_model('Models/model2')
    
    similarity_matrix = pd.read_pickle('similarity_matrix.pkl')
    
    return gametoix, ixtogame, model, similarity_matrix
    
gametoix, ixtogame, model, similarity_matrix = load_data()   

#@tf.function
def predict(model, vec):
    return model.predict(vec)

def similar_games(similarity_matrix, fav_games):
    for i,game in enumerate(fav_games):
        try:        
            if i < 1:
                similar_items = pd.DataFrame(similarity_matrix.loc[game])
                similar_items.columns = ['similarity_score'+str(i)]
            else:
                similar_items['similarity_score'+str(i)] = list(similarity_matrix.loc[game].values)
        except:
            continue

    similar_items['similarity_score'] = similar_items.mean(axis = 1)
    similar_items = similar_items.sort_values('similarity_score', ascending = False)
    similar_items = similar_items.iloc[1:11]
    similar_items.reset_index(inplace=True)
    similar_items = similar_items.rename(index=str, columns={"index":"game"})
    similar_items = similar_items[['game','similarity_score']]
    return similar_items

NUM_GAMES = 1088

st.title('Video Game Dashboard')
st.subheader('Enter your favourite games')

recc_games = []
vec = np.zeros((1,NUM_GAMES))
fav_games = st.multiselect('Enter games: ', list(gametoix.keys()))   
seq = [gametoix[game] for game in fav_games if game in gametoix]
for i in seq:
    vec[0][i] = 1
    
st.write(fav_games)

if st.button('Recommend'):
    pred = model.predict(vec)
    scores = np.argsort(-pred)
    for i in scores[0][:13]:
        recc_games.append(ixtogame[i])
    recc_games = [x for x in recc_games if x not in fav_games]
    recc_games2 = similar_games(similarity_matrix, fav_games)
    recc_games2 = list(recc_games2.game)
    
    st.subheader('Recommended Games based on Users')
    st.write(recc_games)
    
    st.subheader('Recommended Games based on what you like')
    st.write(recc_games2)






