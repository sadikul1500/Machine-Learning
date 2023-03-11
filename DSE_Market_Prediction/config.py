import os
from sklearn.preprocessing import MinMaxScaler
from unittest import result

data_dir = '/content/drive/MyDrive/SPM_model_train_eval/DSECompanyData'

################ Just chnage start and end index ##############
start_index = 149
end_index = 150
##########################################

data_files = os.listdir(data_dir)[start_index:end_index] #'ACI.csv'

# print(type(data_files))
# print(data_files[:2])
print('data files: ', data_files)

num_classes = 2
class_names = ['closing_price', 'volume']
train_test_ratio = .10
window = 100

EPOCHS = 100
BATCH_SIZE = 32

# optimizer = "sgd"
optimizer_fn = "adam"
# optimizer = "rmsprop"
loss_fn = 'mse'

learning_rate = 0.01
momentum = 0.9
model_name = "CNN"

# paths and directories
result_dir = '/content/drive/MyDrive/SPM_model_train_eval/results'


# train_dir = "/content/Dog_vs_Cat/train"
# valid_dir = "/content/Dog_vs_Cat/valid"
# test_dir = "/content/Dog_vs_Cat/test"
scaler = MinMaxScaler(feature_range=(0,1))

version = '1.0'
