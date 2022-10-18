# Caricare le librerie
import tensorflow as tf
from keras import callbacks
from keras import optimizers
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model, clone_model
from tensorflow.keras.layers import Dropout, Flatten, Dense, Conv2D, BatchNormalization, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.optimizers import SGD
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


import numpy as np
import cv2
import h5py
import os

import mlflow
import mlflow.keras

# #ML FLOW PARAMS
# get_ipython().system_raw("mlflow ui --port 5000 &")
# # from getpass import getpass

# #os.environ['MLFLOW_TRACKING_USERNAME'] = input('Enter your DAGsHub username: ')
# os.environ['MLFLOW_TRACKING_USERNAME'] = "GaetanoDibenedetto"
# #os.environ['MLFLOW_TRACKING_PASSWORD'] = getpass('Enter your DAGsHub access token: ')

# os.environ['MLFLOW_TRACKING_PASSWORD'] = "ddec1d9afd9f6c362203803b1cee472f02892972"
# #os.environ['MLFLOW_TRACKING_PROJECTNAME'] = input('Enter your DAGsHub project name: ')
# os.environ['MLFLOW_TRACKING_PROJECTNAME'] = "glassDetection"
# mlflow.set_tracking_uri(f'https://dagshub.com/' + os.environ['MLFLOW_TRACKING_USERNAME'] + '/' + os.environ['MLFLOW_TRACKING_PROJECTNAME'] + '.mlflow')

# mlflow.start_run()
# #Params MLFLOW for datasets
# trainingSetSize = 20000
# validationSetSize = trainingSetSize + 2500
# testSize = 2500
# percentageGlasses = 0.5
# percentageNoGlasses = 0.5
# mlflow.log_param("trainingSetSize", trainingSetSize)
# mlflow.log_param("validationSetSize", validationSetSize)
# mlflow.log_param("testSize", testSize)
# mlflow.log_param("percentageGlasses", percentageGlasses)
# mlflow.log_param("percentageNoGlasses", percentageNoGlasses)

# IMPORTA DATASET
dataset_used = "selfie"
mlflow.log_param("dataset_used", dataset_used)
random_state=0
mlflow.log_param("random_state", random_state)

with h5py.File("./data/Selfie_reduced/selfie_reduced.h5",'r') as data_aug:
  
  X = data_aug["img"][...] 
  aug_wearing_glasses = data_aug["data_label1"][...] 
  aug_wearing_sunglasses = data_aug["data_label2"][...] 

y = []
for i in range(len(aug_wearing_glasses)):
    if str(aug_wearing_glasses[i]) == '1' or str(aug_wearing_sunglasses[i]) == '1':
        y.append('1')
    else : y.append('0')


X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    train_size=0.8,
    test_size=0.2,
    random_state=random_state,
)
print(len(X))
print(len(X_train))
print(len(X_valid))

print(y_train[0])
print(type(X_train[0]))
print(X_train[0].shape)

cv2.imshow('img',X_train[2])
cv2.waitKey()

