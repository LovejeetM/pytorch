import os
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.simplefilter('ignore', FutureWarning)

filepath='concrete_data.csv'
concrete_data = pd.read_csv(filepath)

concrete_data.head()

concrete_data.shape
concrete_data.isnull().sum()

concrete_data_columns = concrete_data.columns
predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']]
target = concrete_data['Strength']

predictors.head()
target.head()

predictors_norm = (predictors - predictors.mean()) / predictors.std()
predictors_norm.head()

n_cols = predictors_norm.shape[1]

def regression_model():
    model = Sequential()
    model.add(input(shape= (n_cols,)))
    model.add(Dense(50, activation= 'relu'))
    model.add(Dense(50, activation= 'relu'))
    model.add(Dense= 1)

    model.compile(optimizer= "adam", loss= 'mean_squared_error')
    return model

model = regression_model()

model.fit(predictors_norm, target, valdation_split= 0.25, epochs= 100, verbose= 2)

def regression_model():
    input_colm = predictors_norm.shape[1] 
    model = Sequential()
    model.add(Input(shape=(input_colm,)))  
    model.add(Dense(50, activation='relu'))  
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu')) 
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))  
    model.add(Dense(1))  
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

model = regression_model()
model.fit(predictors_norm, target, validation_split=0.1, epochs=100, verbose=1)
