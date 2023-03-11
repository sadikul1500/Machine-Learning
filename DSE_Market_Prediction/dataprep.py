import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from config import *

def prepared_data(data_file):
    train_df = pd.read_csv(f"{data_dir}/{data_file}")
    train_df = train_df.drop(['Scrip', 'Unnamed: 0', 'DateEpoch'], axis=1)
    train_df['date'] = pd.to_datetime(train_df['DateString'])
    train_df.set_index('date', inplace=True)
    train_df = train_df.sort_index()
    train_df = train_df.drop_duplicates()

    train_df['closing_price'] = train_df['Close']
    train_df['volume'] = train_df['Volume']
    train_df = train_df.drop(['Close', 'Volume', 'DateString'], axis=1)
    
    dataset = train_df.filter(class_names)
    scaled_data = scaler.fit_transform(dataset)
    return scaled_data

def multiVariant_timeseries_data_XY(data):
     X = []
     Y = []
     end = len(data)
     for i in range(window, end):
         X.append(data[i-window : i, :])
         Y.append(data[i, :])
     return np.array(X), np.array(Y)

def generate_train_dataset(data_file):
    dataset = prepared_data(data_file)
    X, Y = multiVariant_timeseries_data_XY(dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=train_test_ratio, shuffle=False)
    return (X_train, y_train), (X_test, y_test)

def generate_test_dataset(data_file):
    dataset = prepared_data(data_file)
    X, Y = multiVariant_timeseries_data_XY(dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=train_test_ratio, shuffle=False)
    return (X_test, y_test)
