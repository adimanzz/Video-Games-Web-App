# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 18:35:58 2020

@author: adity
"""

import pandas as pd
pd.options.display.max_columns = 500
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

df = pd.read_csv('user_games.csv')
df.columns = ['user_id', 'game', 'state', 'playtime', 'zero']
df.drop(['zero'], 1, inplace = True)
df.head()

df_games = df.iloc[:, :2]
df_games.head()
df_games.drop_duplicates(inplace = True)
df_games['one'] = 1

num_users = df_games.iloc[:,0].nunique()
num_games = df_games.iloc[:,1].nunique()

users = df_games.iloc[:,0].unique().tolist()
games = df_games.iloc[:,1].unique().tolist() 
    
#Pivot
df_games = df_games.pivot(index='user_id', columns='game', values='one')
df_games.fillna(0, inplace = True)

#Model
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.fc1 = nn.Linear(num_games, 256)
        self.fc2 = nn.Linear(256, 100)
        self.fc3 = nn.Linear(100, 256)
        self.fc4 = nn.Linear(256, num_games)
        self.activation = nn.Relu()
        
   def forward(self, x):
       x = self.activation(self.fc1(x))
       x = self.activation(self.fc2(x))
       x = self.activation(self.fc3(x))
       x = self.fc4(x)
       return x
   
ae = AE()
criterion = nn.MSELoss()
optimizer = optim.Adam(ae.parameters(), lr = 0.01, weight_decay = 0.5)

class train():
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


