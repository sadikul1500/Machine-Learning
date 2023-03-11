import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_squared_error
from config import *
from dataprep import generate_test_dataset

def evaluate(data_file):
    # get the test_dataset
    X_test, Y_test = generate_test_dataset(data_file)
    X_test = np.reshape(X_test, (X_test.shape[0], 25, 4, 2)) 
    if X_test.shape[0] == 0:
      return 0
    result_save_path = os.path.join(result_dir, model_name)
    model = "{}_{}_{}".format(model_name, version, data_file.split('.')[0])
    model_save_path = os.path.join(result_save_path, model)
   
    loaded_model = tf.keras.models.load_model(model_save_path)
    
    predictions = loaded_model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    Y_test = scaler.inverse_transform(Y_test)
    rmse = np.sqrt(mean_squared_error(predictions, Y_test))
    return rmse