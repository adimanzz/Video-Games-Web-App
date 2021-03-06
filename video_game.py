# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 13:11:16 2020

@author: aditya
"""
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
pd.options.display.max_columns = 500
import numpy as np
from sqlalchemy import create_engine

import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly
from statsmodels.tsa.arima_model import ARIMA

import tensorflow.compat.v1 as tf
from tensorflow import Graph
#from tensorflow import Session
from keras.models import load_model
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
#config = ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.2
#config.gpu_options.allow_growth = True
#session = InteractiveSession(config=config)


from twitterscraper import query_tweets
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from attention import Attention
from clean_text import CleanText
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from wordcloud import WordCloud

st.beta_set_page_config(page_icon='Images/icon4.png', layout='wide')



@st.cache(persist=False, allow_output_mutation = True, show_spinner = True)
def load_data():
    
    engine = create_engine('sqlite:///database/videogames.db', echo=True)
    sqlite_connection = engine.connect()
    
    
    gametoix = pd.read_pickle('game_index_dictionaries/gametoix.pkl')
    
    ixtogame = pd.read_pickle('game_index_dictionaries/ixtogame.pkl')
        
    similarity_matrix = pd.read_pickle('similarity_matrix.pkl')
    
    url = pd.read_sql("select * from vgsales_url", sqlite_connection)
    #vg_url['Year'] = vg_url['Year'].apply(lambda x: str(x).replace('.0',''))
    
    sales = pd.read_sql("select * from vgsales", sqlite_connection)
    sales = sales.dropna()
    
    sales_graph = pd.read_pickle('sales_graph.pkl')
    
    sales_graph2 = pd.read_pickle('sales_graph2.pkl')
    
    ratings = pd.read_pickle('ratings.pkl')
    
    prices_graph = pd.read_pickle('prices_graph.pkl')
    
    playtime_graph = pd.read_pickle('playtime_graph.pkl') 
    
    user = pd.read_sql("select * from user_games", sqlite_connection)
    user.columns = ['user','game','activity','playtime','0']
    user.drop(['0'], 1, inplace = True)    
    user = user.loc[user['activity'] != 'purchase']    
    user.drop(['activity','user'], 1, inplace = True)
    
    steam = pd.read_sql("select * from steamgames", sqlite_connection)
   
    
    return gametoix, ixtogame, similarity_matrix,  url, sales, sales_graph, sales_graph2, ratings,prices_graph, playtime_graph, user, steam
    
gametoix, ixtogame, similarity_matrix,  url, sales, sales_graph, sales_graph2, ratings, prices_graph, playtime_graph, user, steam = load_data() 

#tf.enable_eager_execution()
#tf.disable_v2_behavior()


model = load_model('Models/model2')
model_sentiment = load_model('Models/model_mil_77.h5' ,custom_objects = {'Attention':Attention})
            

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
    similar_items = similar_items.iloc[1:11] #original [1:11]
    similar_items.reset_index(inplace=True)
    similar_items = similar_items.rename(index=str, columns={"index":"game"})
    similar_items = similar_items[['game','similarity_score']]
    return similar_items

def plot_graph(df, features):
    plt.figure(figsize=(16,6))
    colors = ['red','green']
    for i,x in enumerate(features):
        plt.subplot(1,2,i+1).set_title('Sentiment-wise '+x.capitalize()+' ratio', fontsize = 20)
        df.groupby(['sentiment'])[x].mean().plot(color=colors[i], linestyle='dashed', marker='o',
                                                                markerfacecolor=colors[i], markersize=10).set_xticklabels(['','Negative','','','','','Positive'])
        plt.xlabel('Sentiment', fontsize = 10)
        plt.ylabel('Average'+x.capitalize(), fontsize = 10)
        plt.xticks(fontsize = 15)
    plt.show()
    st.pyplot()
    
def label_to_sentiment(label):
    if label == 0:
        return 'Negative'
    else:
        return 'Positive'
  
def plot_wordcloud():
        plt.figure(figsize=(16,6))
        dfs = [df['clean_text'][df['sentiment'] == 'Positive'],df['clean_text'][df['sentiment'] == 'Negative']]
        sentiment = ['Positive','Negative']
        for i, data in enumerate(dfs):
            words = []
            for sent in data:
                for word in sent:
                    words.append(word.lower()) 
            skip = ['fuck','video','game','video game', 'play','fucking', 'twitter']  
            words = [x for x in words if x not in skip]
            words = pd.Series(words).str.cat(sep=' ')                  
            wordcloud = WordCloud(width=700, height=400,max_font_size=80).generate(words)
            plt.subplot(1,2,i+1).set_title(sentiment[i], fontsize = 20)
            plt.plot()
            plt.imshow(wordcloud, interpolation="bilinear")
            #ax[i].set_title('Positive', fontsize = 25)
            plt.axis("off")
        plt.show()
            #st.subheader(title)
        st.pyplot(use_container_width=True)    

NUM_GAMES = 1088

#Graph df
ratings_plot = ratings.sort_values(by='website_rating', ascending = False)[['game','website_rating','release']]


st.image('Images/background2_logo.jpg', use_container_width=True)

option = st.sidebar.selectbox(
    'Select Page',
     ['Dashboard',  'Recommendation System', 'Sentiment Analysis'])


if option == 'Dashboard':     
    
    
    st.title('Video Game Dashboard')
    content = st.sidebar.selectbox('Select Content',['Overview','Demographics','Top Tens'])
    if content == 'Overview':
        radio = st.sidebar.radio('Select Graph: ',options = ['Sales (Total)','Price & Duration','Region (Sales)','Genre'])
        if radio == 'Sales (Total)':
            components.html(f"""
    <link href="https://unpkg.com/tailwindcss@^1.0/dist/tailwind.min.css" rel="stylesheet">
    <div class="rounded-lg shadow-lg overflow-hidden rounded-md bg-gray-500 px-4 py-6">
    <div class="flex flex-wrap gap-x-3 gap-y-2">
    
        <div class="rounded-md bg-white overflow-hidden shadow-lg">
            <div class="flex px-6 pt-2">
                <div class="text-1 text-2xl text-center"><span>Average Playtime</span></div>
            </div>
            <div class="px-6 font-bold text-lg text-x1 pb-2"><span>48 Hours</span></div>
            </div>
            
        <div class="rounded-md bg-white overflow-hidden shadow-lg">
            <div class="px-6 pt-2">
                <div class="text-1 text-2xl text-center"><span>Most Popular Platform </span></div>
            </div>
            <div class="px-6 font-bold text-lg text-x1 pb-2"><span>PS4</span></div>
            </div>    
            
        <div class="rounded-md bg-white overflow-hidden shadow-lg">
            <div class="px-6 pt-2">
                <div class="text-1 text-2xl text-center"><span>Most Popular Genre</span></div>
            </div>
            <div class="px-6 font-bold text-lg text-x1 pb-2"><span>Action</span></div>
           </div>
           
       <div class="rounded-md bg-white overflow-hidden shadow-lg">  
            <div class="px-6 pt-2">
                <div class="text-1 text-2xl text-center"><span>Average Increase in Price After Covid</span></div>
            </div>
            <div class="px-6 font-bold text-lg text-x1 pb-2"><span>$9</span></div>
           </div>
                 
        
    </div>
    </div>
            """)
            
            graph = url.groupby(['Year'])['Name'].count() 
            graph = graph.iloc[:-2]
            #graph.index = index   
            model = ARIMA(graph, order=(5,1,1))
            model_fit = model.fit()
            forecast, stderr, conf = model_fit.forecast(steps=12, alpha = 0.35)
            forecast = pd.Series(forecast)
            conf = pd.DataFrame(conf)
            forecast.index = np.arange(2019,2031,1)
            conf.index = np.arange(2019,2031,1)
    
            ax = graph.plot(label='observed', figsize=(20, 7))
            forecast.plot(ax=ax, label='Forecast')
            # conf.iloc[:, 0].plot(ax=ax, color='k', alpha=.30)
            # conf.iloc[:, 1].plot(ax=ax, color='k', alpha=.30)
            ax.fill_between(graph.index, graph.values, color='k', alpha=.15)
            ax.fill_between(conf.index,  conf.iloc[:, 0],  conf.iloc[:, 1], color='k', alpha=.15)
            plt.xlabel('Years',fontsize=20)
            plt.ylabel('Number of Games Made',fontsize=17)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.title('Total Games made from 1970 - 2020 (Forecast till 2030)',fontsize=25)
            plt.box(False)
            
            #st.subheader('Number of Games produced from 1970 - 2020')
            st.pyplot( use_container_width=True)    
         
        elif radio == 'Price & Duration':    
            fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]],
                        subplot_titles = ('Most Popular Price ranges','Most Popular Game Lengths'),horizontal_spacing=0,
                        column_widths=[0.5,0.5])
    
            fig.add_trace(go.Pie(labels=prices_graph['average_price'], values=prices_graph['steamid'], name='Most Popular Price ranges',
                        marker=dict(colors=plotly.colors.sequential.Mint_r,line=dict(color='#FFF', width=2)),pull=[0.1,0,0,0,0,0,0]),
                  1, 1)
            fig.add_trace(go.Pie(labels=playtime_graph['main_story_duration_average'], values=playtime_graph['name'], name='Most Popular Game Lengths',
                  marker=dict(colors=plotly.colors.sequential.PuBu_r, line=dict(color='#FFF', width=2)),pull=[0.1,0,0,0,0,0,0]), 
                  1, 2)
            fig.update_traces(hole=.5, hoverinfo="label+percent",showlegend=False, textinfo='label+percent')
            fig.update_layout( width = 1500, height = 700)
            st.plotly_chart(fig, use_container_width=True) 
        
        elif radio == 'Region (Sales)':
            fig = px.bar(sales_graph, y='Region',x='Total Sales(In Millions)',color='Region',
                animation_frame='Year', range_x=[0,360], title = 'Total Games Sold from 1970 - 2020', 
                orientation='h', width = 1500,height=500)
           
            st.plotly_chart(fig, use_container_width=True)      
                    
        else:
            fig = px.bar(sales_graph2, y='Number of Games',x='Genre',color='Genre',
                animation_frame='Year', range_y=[0,270], title = 'Popularity of Game Genres from 1970 - 2020',
                width = 1500,height=500)
            
            st.plotly_chart(fig, use_container_width=True)
        
        #st.plotly_chart(fig2)
        
    elif content == 'Demographics':
        radio2 = st.sidebar.radio('Select Graph: ', ['Age','Gender','Income','Global Revenue'])
        
        if radio2 == 'Age':
            age_df = pd.DataFrame({'age':['18 - 24 years','25 - 34 years','35 - 44 years','45 - 54 years'],
                          'percentage':[29.9, 42.2, 21.5, 6.5]})        
            colors = ['lightslategray',] * 4
            colors[1] = 'crimson'        
            fig2 = go.Figure(data = [go.Bar(x = age_df['age'], y = age_df['percentage'], marker_color=colors,
                                          text=age_df['percentage'], textposition='outside')])
            fig2.update_layout(width = 1500, height = 500,title = 'Distribution of Users by Age', 
                              xaxis_title = 'Age', yaxis_title = 'Percentage')
            st.plotly_chart(fig2, use_container_width=True)
        
        elif radio2 == 'Gender':
            gender_df = pd.DataFrame({'gender': ['male','female'],
                             'percentage': [77.6, 22.4]})
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                y=['Gender'],
                x=[77.6],
                name='Male',
                orientation='h',
                marker=dict(
                    color='rgba(16, 107, 170, 0.6)',
                    line=dict(color='rgba(0, 0,0, 1.0)', width=1)
                )  , text = ['77.6 %'], textposition='inside'
            ))
            fig2.add_trace(go.Bar(
                y=['Gender'],
                x=[22.4],
                name='Female',
                orientation='h',
                marker=dict(
                    color='rgba(150,150,150, 0.6)',
                    line=dict(color='rgba(58, 71, 80, 1.0)', width=1)
                )  , text = ['22.4 %'], textposition='inside', 
            ))        
            fig2.update_layout(width = 1500, height = 500,barmode="stack", 
                              title = 'Distribution of Users by Gender', xaxis_title = 'Percentage')
            st.plotly_chart(fig2, use_container_width=True)
            
        elif radio2 == 'Income':    
            income_df = pd.DataFrame({'income':['-','Low Income','Medium Income','High Income'],
                             'percentage': [100, 42.6, 30.3, 27.1]})
            fig2 = go.Figure(go.Pie(labels=income_df['income'], values=income_df['percentage'],
                  marker=dict(colors= [
                'rgb(255, 255, 255)',
                plotly.colors.sequential.Teal_r[0],
                plotly.colors.sequential.Teal_r[1],
                plotly.colors.sequential.Teal_r[2]], line=dict(color='#FFF', width=2)),pull=[0.1,0,0], domain={"x": [1, 1]},
                          textfont=dict(color='white')))  
            fig2.update_traces(hole=.5, hoverinfo="label+value",showlegend=True, textinfo='label+value', rotation = 90)
            fig2.update_layout( width = 1500, height = 800, title = 'Distribution of Users by Income')       
            st.plotly_chart(fig2, use_container_width=True)
            
        else:
            df_countries = pd.read_pickle('data/countries_revenue.pkl')
            fig2 = px.choropleth(df_countries, locations='iso',
                                color='Revenue', 
                                hover_name="Countries",
                                color_continuous_scale=px.colors.sequential.Teal)
            fig2.update_layout(width = 1500, height = 700,title='Global comparison of Revenue (in millions) generated in 2019')
            st.plotly_chart(fig2, use_container_width=True)
        
    
    else:      
        
        owner = pd.DataFrame(steam.sort_values(by='owner_base',ascending=False).drop_duplicates(subset=['steamid']).head()[['steamid','owner_base']])
        owner['game'] = ['GTA 5', "Tom Clancy's Rainbow Six® Siege", 'Borderlands 2', 'Half-Life 2', 'The Tiny Bang Story']        
        foll = pd.DataFrame(steam.groupby(['steamid'])['new_followers'].sum().sort_values(ascending=False).head())
        foll['game'] = ['GTA 5', 'Red Dead Redemption 2', 'Wolcen: Lords of Mayhem', 'TemTem', 'Human: Fall Flat']
        
        years = np.arange(1970,2021,1)
        begin = st.sidebar.selectbox('From:', years,0)
        end = st.sidebar.selectbox('Till:', years,50)
        
        radio3 = st.sidebar.radio('Select Graph:', ['Sales','Rating','Owned & Followed','Playtime'])
        
        if radio3 == 'Sales':
            graph = url[(url['Year'] >= begin) & (url['Year'] <= end)].sort_values(by='Total_Shipped', ascending =  False)[['Name','Total_Shipped']].head(10)
            fig3 = plt.figure(figsize = (22,7))
            sns.set_context('poster')
            sns.barplot(y="Name", x="Total_Shipped", data=graph,
                             palette="Blues_d")
            sns.despine(left=True, bottom=True)
            plt.title('Top Ten Most Sold Games from: '+str(begin)+' - '+str(end),fontsize=30)
            plt.ylabel('Games',fontsize=20)
            plt.xlabel('Games Sold (In Millions)',fontsize=20)  
            plt.yticks(fontsize = 10)
            plt.tight_layout()
            st.pyplot()
        
        elif radio3 == 'Rating':
            years2 = np.arange(1995,2020,1)
            if begin and end not in years2:
                begin = 1995
                end = 2019
            fig32 = plt.figure(figsize=(20,7))
            plt.subplot(1,2,1).set_title('Top Ten Highest Rated Games', fontsize = 20)
            plt.plot()
            sns.barplot(x="game", y="website_rating", data=ratings_plot[(ratings_plot['release'].dt.year>=begin)&(ratings_plot['release'].dt.year<=end)].head(10), palette='ch:.25_r') #mako
            plt.xticks(rotation=45,fontsize=10)
            plt.ylabel('Rating',fontsize=20)
            plt.xlabel('Games',fontsize=20)
            sns.despine(left=True, bottom=True)
            sns.set_context('poster')
            plt.subplot(1,2,2).set_title('Top Ten Lowest Rated Games', fontsize = 20)
            plt.plot()
            sns.barplot(x="game", y="website_rating", data=ratings_plot[(ratings_plot['release'].dt.year>=begin)&(ratings_plot['release'].dt.year<=end)].tail(10), palette='ch:2.5,-.2,dark=.3')
            plt.xticks(rotation=45,fontsize=10)
            plt.ylabel('Rating',fontsize=20)
            plt.xlabel('Games',fontsize=20)
            sns.despine(left=True, bottom=True)
            sns.set_context('poster')
            plt.tight_layout()
            st.pyplot()

        elif radio3 == 'Owned & Followed':
            fig3 = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]],
                        subplot_titles = ('Top 5 Most Owned Games on Steam','Top 5 Most Followed Games on Steam'),horizontal_spacing=0,
                        column_widths=[0.5,0.5])
            
    
            fig3.add_trace(go.Pie(labels=owner['game'], values=owner['owner_base'], name='Top 5 Most Owned Games on Steam',
                                marker=dict(colors=plotly.colors.sequential.Mint_r,line=dict(color='#FFF', width=2)),pull=[0.1,0,0,0,0,0,0]),
                          1, 1)
            fig3.add_trace(go.Pie(labels=foll['game'], values=foll['new_followers'], name='Most Popular Game Lengths',
                      marker=dict(colors=plotly.colors.sequential.PuBu_r, line=dict(color='#FFF', width=2)),pull=[0.1,0,0,0,0,0,0]), 
                          1, 2)
            
            fig3.update_traces(hole=.5, hoverinfo="label+percent",showlegend=False, textinfo='label+percent')
            fig3.update_layout( width = 1500, height = 700)
            st.plotly_chart(fig3, use_container_width=True)
        
        else:
            highest_playtime = user.sort_values(by='playtime', ascending = False).drop_duplicates(subset=['game']).head(10)   
            user_graph = user.groupby(['game']).agg({'playtime':['mean']}).reset_index()
            user_graph.columns = ['game','average playtime']
            avg_playtime = user_graph.sort_values(by='average playtime', ascending = False)
            highest_playtime['average_playtime'] = highest_playtime['game'].apply(lambda x: avg_playtime[avg_playtime['game'] == x]['average playtime'].iloc[0])
            highest_playtime = highest_playtime.sort_values(by='playtime')
            
            fig3 = go.Figure()    
            fig3.add_trace(go.Bar(
                y=highest_playtime['game'],
                x=highest_playtime['average_playtime'],
                orientation = 'h',
                name='Average Playtimes',
                marker_color='lightsalmon'
            ))
            fig3.add_trace(go.Bar(
                y=highest_playtime['game'],
                x=highest_playtime['playtime'],
                orientation = 'h',
                name='Highest Playtimes',
                marker_color='indianred'
            ))
            
            fig3.update_layout(barmode='group', xaxis_tickangle=-45, title = 'Top Ten Games with Highest Playtimes by Users',
                              xaxis_title = 'Hours', yaxis_title = 'Games', width = 1500, height = 550)
            st.plotly_chart(fig3, use_container_width=True)
    
elif option == 'Recommendation System':
    st.title('Video Game Recommendation System')
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
    
    if st.button('Recommend'):
        pred = model.predict(vec)
        scores = np.argsort(-pred)
        for i in scores[0][:15]:   #orignal 13
            recc_games.append(ixtogame[i])
        recc_games = [x for x in recc_games if x not in fav_games]
        recc_games = recc_games[:10]
        recc_games2 = similar_games(similarity_matrix, fav_games)
        recc_games2 = list(recc_games2.game)
        
        for i in recc_games:
            images1.append('http://www.vgchartz.com' + str(url[url['Name'] == i]['img_url'].iloc[0]))
            
            
        for i in recc_games2:
            images2.append('http://www.vgchartz.com' + str(url[url['Name'] == i]['img_url'].iloc[0]))
            
            
        st.subheader('Recommended Games based on Users:')
        #st.write(recc_games)
        st.image(images1, width = 250, caption = recc_games)
        
        st.subheader('Recommended Games similar to what you like:')
        #st.write(recc_games2)
        st.image(images2, width = 250, caption = recc_games2, use_container_width=True)
        
        
        #Real-Time training
        model.fit(vec, vec,  batch_size = 1, epochs = 10, shuffle = True) 
        
        
else:       
    
    
    

    st.title("Sentiment Analysis of Tweets")    
    date = st.sidebar.date_input('Enter Date Range:',[datetime.date(2019, 7, 6), datetime.date(2019, 7, 8)])
    limit = st.sidebar.slider('Enter number of Tweets to scrape:',0,1000)
    lang = 'english'
    
    
    if st.button('Scrape Tweets'):
        with st.spinner('Scraping Tweets...'):
            tweets = query_tweets('videogames', begindate = date[0], enddate = date[1], limit = limit, lang = lang)
        
        
        df = pd.DataFrame(t.__dict__ for t in tweets)
        df = df[['timestamp','text','likes','retweets']]
        df = df.drop_duplicates(subset=['likes'])
        clean = CleanText()
        df['clean_text'] = clean.clean(df['text']) 
        df['clean_text'] = df['clean_text'].apply(lambda x: clean.tokenize(x)) 
        
        docs = df['clean_text']
        
        #tokenizer
        t = Tokenizer()
        t.fit_on_texts(docs)
        vocab_size = len(t.word_index) + 1
        
        #encode the documents
        encoded_docs = t.texts_to_sequences(docs)
        
        #pad docs to max length
        padded_docs = pad_sequences(encoded_docs, maxlen = 40, padding = 'post') 
        labels_categorical = model_sentiment.predict(padded_docs)
        df['labels'] = np.argmax(labels_categorical, axis = 1)
        df['sentiment'] = df['labels'].apply(lambda x: label_to_sentiment(x))
        #df = pd.read_pickle('tweets_sentiment.pkl')
        
        fig = go.Figure(data = [go.Bar(x = df['sentiment'].value_counts().index, y = df['sentiment'].value_counts()/len(df)*100,
                        text=round(df['sentiment'].value_counts()/len(df)*100,1), textposition='outside', marker_color=['crimson',plotly.colors.sequential.Mint_r[4]])])
        fig.update_layout(title = 'Distribution of Tweets by Sentiment', xaxis_title='Sentiment',
                          yaxis_title='Percentage',height=700,width=1500)
        st.plotly_chart(fig, use_container_width=True)
        
             
        plot_wordcloud() 
        
        plot_graph(df, ['likes','retweets'])
        
        st.subheader('Most Liked Tweets')
        st.table(df[df['likes'] < 1000].sort_values(by = 'likes', ascending = False).iloc[:3][['text','likes']])
        
        st.subheader('Most Retweetd Tweets')        
        st.table(df[df['retweets'] < 1000].sort_values(by = 'retweets', ascending = False).iloc[:3][['text','retweets']])
    
    