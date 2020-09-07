# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 12:44:46 2020

@author: aditya
"""
import pandas as pd 
from sqlalchemy import create_engine
import time

engine = create_engine('sqlite:///database/videogames.db', echo=True)
sqlite_connection = engine.connect()


url = pd.read_csv('vgsales_url.csv')
url.to_sql('vgsales_url', sqlite_connection, if_exists='fail', index = False)

sales = pd.read_csv('vgsales.csv')
sales.to_sql('vgsales', sqlite_connection, if_exists='fail', index = False)


user_df = pd.read_csv('user_games.csv')
user_df.to_sql('user_games', sqlite_connection, if_exists='fail', index=False)

steam = pd.read_csv('steamgames.csv')
steam.to_sql('steamgames', sqlite_connection, if_exists='fail', index = False)




user2 = pd.read_sql("select * from steamgames", sqlite_connection)
user2 = pd.DataFrame(user2)


sqlite_connection.close()
