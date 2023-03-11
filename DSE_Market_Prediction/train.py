import os
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
import numpy as np

import config
from models import bi_lstm_model, cnn_lstm_model
from dataprep import generate_train_dataset
from utils import evaluate

def get_model():
    if config.model_name == "LSTM":
        model = bi_lstm_model()
    if config.model_name == "CNN":
        model = cnn_lstm_model()

    # model.summary()
    if config.optimizer_fn == 'sgd':
        optimizer = tf.keras.optimizers.SGD(config.learning_rate, config.momentum)
    elif config.optimizer_fn == 'adam':
        optimizer = tf.keras.optimizers.Adam(config.learning_rate, config.momentum)
    elif config.optimizer_fn == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(config.learning_rate, config.momentum)
    else:
        print("add another optimizer")
    
    model.compile(optimizer= optimizer,
                    loss=config.loss_fn,
                    metrics=['accuracy'])
  

    return model

if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU found: {gpu}")

    for data_file in config.data_files:
      # load configs and resolve paths
      if not os.path.exists(config.result_dir):
          os.mkdir(config.result_dir)
      result_save_path = os.path.join(config.result_dir, config.model_name)
      if not os.path.exists(result_save_path):
          os.mkdir(result_save_path)
      log_dir = os.path.join(result_save_path, "logs/logs_{}_{}".format(config.version, data_file.split('.')[0]))
      if not os.path.exists(log_dir):
          os.mkdir(log_dir)
      log_train = os.path.join(log_dir, 'train')
      if not os.path.exists(log_train):
          os.mkdir(log_train)
      log_valid = os.path.join(log_dir, 'valid')
      if not os.path.exists(log_valid):
          os.mkdir(log_valid)

      # get the original_dataset
      train_dataset, valid_dataset = generate_train_dataset(data_file)
      model = get_model()
      # set the callbacks
      tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
      rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)
      callback_list = [rlrop, tensorboard_callback]

      if config.model_name == "CNN":
        cnn_train = np.reshape(train_dataset[0], (train_dataset[0].shape[0], 25, 4, 2)) 
        cnn_valid = np.reshape(valid_dataset[0], (valid_dataset[0].shape[0], 25, 4, 2)) 
        #  print(cnn_train.shape, cnn_valid.shape)
      # start training
      model.fit(cnn_train, train_dataset[1],
                  epochs= config.EPOCHS,
                  validation_data=(cnn_valid, valid_dataset[1]),
                  batch_size = config.BATCH_SIZE,
                  callbacks=callback_list,
                  verbose=1)

      # save model
      model_name="{}_{}_{}".format(config.model_name, config.version, data_file.split('.')[0])
      model_save_path = os.path.join(result_save_path, model_name)
      model.save(model_save_path, save_format='tf')

      rmse = evaluate(data_file)

      print(rmse)