# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 11:25:00 2020

@author: aditya
"""
!pip install sqlalchemy
!pip install MySQLdb
!pip install mysqlclient

import pandas as pd
from sqlalchemy import create_engine, types
import MySQLdb
import time

start = time.time()
user = pd.read_csv('user_games.csv')
end = time.time()
print(end - start)


engine = create_engine('mysql://root:root@localhost/webapp') # enter your password and database names here


#df = pd.read_csv("Excel_file_name.csv",sep=',',quotechar='\'',encoding='utf8') # Replace Excel_file_name with your excel sheet name
user.to_sql('user',con=engine,index=False,if_exists='append') # Replace Table_name with your sql table name




dbConnection = engine.connect()

start = time.time()
user2 = pd.read_sql("select * from user", dbConnection)
end = time.time()
print(end - start)
#pd.set_option('display.expand_frame_repr', False)

dbConnection.close()


gametoix = pd.read_pickle('game_index_dictionaries/gametoix.pkl')
    
ixtogame = pd.read_pickle('game_index_dictionaries/ixtogame.pkl')
    
similarity_matrix = pd.read_pickle('similarity_matrix.pkl')

url = pd.read_csv('vgsales_url.csv')
#vg_url['Year'] = vg_url['Year'].apply(lambda x: str(x).replace('.0',''))

sales = pd.read_csv('vgsales.csv')
sales = sales.dropna()

sales_graph = pd.read_pickle('sales_graph.pkl')

sales_graph2 = pd.read_pickle('sales_graph2.pkl')

ratings = pd.read_pickle('ratings.pkl')

prices_graph = pd.read_pickle('prices_graph.pkl')

playtime_graph = pd.read_pickle('playtime_graph.pkl') 

user = pd.read_csv('user_games.csv')
user.columns = ['user','game','activity','playtime','0']
user.drop(['0'], 1, inplace = True)    
user = user.loc[user['activity'] != 'purchase']    
user.drop(['activity','user'], 1, inplace = True)

steam = pd.read_csv('steamgames.csv')
   
