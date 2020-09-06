# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 16:11:23 2020

@author: adity
"""

import streamlit as st
import pandas as pd
import numpy as np

#import tensorflow as tf
from tensorflow.keras.models import load_model
import keras.backend.tensorflow_backend as tb
#tb._SYMBOLIC_SCOPE.value = True

@st.cache(persist=False)
def load_data():
    gametoix = pd.read_pickle('game_index_dictionaries/gametoix.pkl')
    
    ixtogame = pd.read_pickle('game_index_dictionaries/ixtogame.pkl')
    
    model = load_model('Models/model2')
    
    similarity_matrix = pd.read_pickle('similarity_matrix.pkl')
    
    vg_url = pd.read_csv('vgsales_url.csv')
    
    return gametoix, ixtogame, model, similarity_matrix,  vg_url
    
gametoix, ixtogame, model, similarity_matrix,  vg_url = load_data() 

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
images1 = []
images2 = []

vec = np.zeros((1,NUM_GAMES))
fav_games = st.multiselect('Enter games: ', list(gametoix.keys()))   
seq = [gametoix[game] for game in fav_games if game in gametoix]
for i in seq:
    vec[0][i] = 1
    
#st.write(fav_games)

st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        max-width: {100}px;
        max-height: {50}px;
        padding-top: {0}rem;
        padding-right: {1}rem;
        padding-left: {1}rem;
        padding-bottom: {0}rem;
    }}
    .reportview-container .main {{
        color: {'black'};
        background-color: {'white'};
    }}
</style>
""",
        unsafe_allow_html=True,
    )

fig = make_subplots(cols=2,shared_xaxes=False, subplot_titles=('Total Games Sold from 1970 - 2020','Popularity of Game Genres from 1970 - 2020'))
        fig1 = px.bar(sales_graph, y='Region',x='Total Sales(In Millions)',color='Region',animation_frame='Year', range_x=[0,360], orientation='h')
        fig2 = px.bar(sales_graph2, y='Number of Games',x='Genre',color='Genre',animation_frame='Year', range_y=[0,270])
        trace1 = fig1['data'][0]
        trace2 = fig2['data'][0]
        fig.add_trace(fig1['data'][0],col=1,row=1)
        fig.add_trace(fig1['data'][1],col=1,row=1)
        fig.add_trace(fig1['data'][2],col=1,row=1)
        fig.add_trace(fig1['data'][3],col=1,row=1)
        
        fig.add_trace(fig2['data'][0],col=2,row=1)
        fig.add_trace(fig2['data'][1],col=2,row=1)
        fig.add_trace(fig2['data'][2],col=2,row=1)
        fig.add_trace(fig2['data'][3],col=2,row=1)
        #fig.add_trace(go.Figure(fig1),row=1,col=1)
        #fig.add_trace(go.Figure(fig2),row=1,col=2)
        
        st.plotly_chart(fig)

if st.button('Recommend'):
    pred = model.predict(vec)
    scores = np.argsort(-pred)
    for i in scores[0][:13]:
        recc_games.append(ixtogame[i])
    recc_games = [x for x in recc_games if x not in fav_games]
    recc_games2 = similar_games(similarity_matrix, fav_games)
    recc_games2 = list(recc_games2.game)
    
    for i in recc_games:
        images1.append('http://www.vgchartz.com' + str(vg_url[vg_url['Name'] == i]['img_url'].iloc[0]))
        
        
    for i in recc_games2:
        images2.append('http://www.vgchartz.com' + str(vg_url[vg_url['Name'] == i]['img_url'].iloc[0]))
        
        
    st.subheader('Recommended Games based on Users:')
    #st.write(recc_games)
    st.image(images1, width = 200, caption = recc_games)
    
    st.subheader('Recommended Games similar to what you like:')
    #st.write(recc_games2)
    st.image(images2, width = 200, caption = recc_games2)
    