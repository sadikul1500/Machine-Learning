import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Dropout, TimeDistributed, Flatten, Conv1D, MaxPooling1D

import config

class BiLSTM(Model):
  def __init__(self, input_shape, num_classes):
    super(BiLSTM, self).__init__(name='BiLSTM')
    self.input_layer = Bidirectional(LSTM(50, return_sequences=True, input_shape = input_shape))
    self.bi_lstm = Bidirectional(LSTM(50, return_sequences=False))
    self.hidden_layer = Dense(25, activation= 'relu')
    self.output_layer = Dense(num_classes, activation= 'relu')
  
  def call(self, input_tensor, training=False):
    x = self.input_layer(input_tensor)
    x = Dropout(0.5)
    x = self.bi_lstm(x)
    x = Dropout(0.5)
    x = self.hidden_layer(x)
    x = self.output_layer(x)
    return x

def cnn_lstm_model():
  model_lstm = Sequential()
  model_lstm.add(TimeDistributed(Conv1D(32, 1), input_shape=(None, 4 , 2)))
  model_lstm.add(TimeDistributed(MaxPooling1D()))
  model_lstm.add(TimeDistributed(Flatten()))
  model_lstm.add(LSTM(32))
  model_lstm.add(Dense(2))

  return model_lstm

def bi_lstm_model():
  model_lstm = Sequential()
  model_lstm.add(Bidirectional(LSTM(50, return_sequences=True, input_shape=(config.window, config.num_classes))))
  model_lstm.add(Dropout(0.5))
  model_lstm.add(Bidirectional(LSTM(50, return_sequences=False)))
  model_lstm.add(Dropout(0.5))
  model_lstm.add(Dense(25))
  model_lstm.add(Dense(2))


  return model_lstm