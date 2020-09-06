# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 20:42:24 2020

@author: adity
"""

import streamlit as st
import pandas as pd

@st.cache(persist=True)
def load_data():
    sm_games = pd.read_csv('steamgames.csv')
    
    vg_games = pd.read_csv('vgsales.csv')
    
    vg_url = pd.read_csv('vgsales_url.csv')
    
    return sm_games, vg_games, vg_url
    
sm_games, vg_games, vg_url = load_data()    


st.title('Video Game Dashboard')
st.subheader('Enter your favourite games')

images = []
game = st.multiselect('Enter games: ', vg_games['Name'][:1000])     
st.write(game)
for i in game:
    images.append('http://www.vgchartz.com' + str(vg_url[vg_url['Name'] == i]['img_url'].iloc[0]))
st.image(images,
         width = 200)

