# -*- coding: utf-8 -*-
"""prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1t2bVtrHNGY1rKUaQh-W6MEuXLInscXgD

**Yusuf's code to get last 100 days data of a company**
"""

import requests
import pandas as pd
import json
import os
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from datetime import date, timedelta

PATH = "/content/drive/MyDrive/SPM_model_train_eval/results/CNN"

def getCompanyJSON(scrip):
  response = requests.get("https://www.amarstock.com/data/afe01cd8b512070a/?scrip=" + scrip + "&cycle=Day1&dtFrom=2000-07-20T05%3A02%3A13.318Z")
  response.raise_for_status()
  if response.status_code == 200:
    todos = json.loads(response.text)
    return todos
  return None


"""# **Load the model.... - Khalil**"""

def get_pretrained_model(model_save_path):
  try:
    model = tf.keras.models.load_model(model_save_path)
  except:
    model = None
  return model



"""**Prediction Code goes here**"""

def get_OBV(cp, volume):
    OBV = []
    OBV.append(0)
    for i in range(1, len(cp)):
        if cp[i] > cp[i-1]: #If the closing price is above the prior close price 
              OBV.append(OBV[-1] + volume[i]) #then: Current OBV = Previous OBV + Current Volume
        elif cp[i] < cp[i-1]:
              OBV.append( OBV[-1] - volume[i])
        else:
              OBV.append(OBV[-1])
    return OBV


def buy_sell(closing_price, obv, obv_ema):
  sigPriceBuy = []
  sigPriceSell = []
  flag = -1 #A flag for the trend upward/downward
  #Loop through the length of the data set
  for i in range(0, len(obv)):
    #if OBV > OBV_EMA  and flag != 1 then buy else sell
      if obv[i] > obv_ema[i] and flag != 1:
          sigPriceBuy.append(closing_price[i])
          sigPriceSell.append(np.nan)
          flag = 1
      #else  if OBV < OBV_EMA  and flag != 0 then sell else buy
      elif obv[i] < obv_ema[i] and flag != 0:    
          sigPriceSell.append(closing_price[i])
          sigPriceBuy.append(np.nan)
          flag = 0
      #else   OBV == OBV_EMA  so append NaN 
      else: 
        sigPriceBuy.append(np.nan)
        sigPriceSell.append(np.nan)
  
  return sigPriceBuy, sigPriceSell

"""# **Sami..... This is prediction is for cnn_lstm check it and complete it after kahlil does his work**
"""

def jsonTOArray(last_100_days, scaler):
  l = []
  
  for days in last_100_days:
    l += [[days['Close'], days['Volume']]]
  
  # scaler = MinMaxScaler(feature_range=(0,1))
  scaled_data = scaler.fit_transform(l)
  # scaled_data
  np_array = np.array(scaled_data)
  np_array = np.reshape(np_array, (1, 25, 4, 2))

  return np_array, scaled_data

def load_model(scrip):
  # company_name = input()
  files = os.listdir(PATH)

  if "CNN_1.0_"+scrip not in files:
    return None
  else:
    model_save_path = f"{PATH}/CNN_1.0_{scrip}"
    model = get_pretrained_model(model_save_path)
    # model.summary()
  
  return model

def getFutureDates(n):
  one_day_delta = timedelta(1)
  day = date.today()
  days = []
  for i in range(n):
    day += one_day_delta
    days.append(day)
  
  return days

def getPredictions(model, np_array, scaler, scaled_data, n):
  predictions = []
  # n = 30 # n can be 30 or 60......
  for i in range(n):
      prediction = np.reshape((model.predict(np_array)), (1,2)) #replace model name...

      scaled_data = np.append(scaled_data, prediction)

      scaled_data = np.reshape(scaled_data, (scaled_data.shape[0]//2, 2))

      np_array = np.array(scaled_data[-100:,:])
      np_array = np.reshape(np_array, (1, 25, 4, 2))
      
      prediction = scaler.inverse_transform(prediction)
      predictions.append(prediction)
  
  predictions = np.array(predictions)
  predictions = np.reshape(predictions, (predictions.shape[0], predictions.shape[2]))

  days = getFutureDates(n)

  closing_price = predictions[:, 0]
  volume = predictions[:, 1]

  obv = get_OBV(closing_price, volume)
  obv_ema = pd.Series(obv).ewm(com=20).mean()

  sigPriceBuy, sigPriceSell = buy_sell(closing_price, obv, obv_ema)

  return closing_price, obv, days, sigPriceBuy, sigPriceSell

def prediction(scrip, n):
  last_100_days = getCompanyJSON(scrip)[-100:]
  
  if len(last_100_days) < 100:
    return 0

  scaler = MinMaxScaler(feature_range=(0,1))

  np_array, scaled_data = jsonTOArray(last_100_days, scaler)
  
  model = load_model(scrip)

  if model == None:
    print('here in model')
    return 0
  
  closing_price, obv, days, sigPriceBuy, sigPriceSell = getPredictions(model, np_array, scaler, scaled_data, n)
  
  # print(closing_price, obv, days)

  return closing_price, obv, days, sigPriceBuy, sigPriceSell

print(prediction('1JANATAMF', 30))