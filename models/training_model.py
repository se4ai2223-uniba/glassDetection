# Caricare le librerie
from sysconfig import get_python_version
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

#ML FLOW PARAMS
# get_python_version().system_raw("mlflow ui --port 5000 &")
# from getpass import getpass

#os.environ['MLFLOW_TRACKING_USERNAME'] = input('Enter your DAGsHub username: ')
os.environ['MLFLOW_TRACKING_USERNAME'] = "GaetanoDibenedetto"
#os.environ['MLFLOW_TRACKING_PASSWORD'] = getpass('Enter your DAGsHub access token: ')

os.environ['MLFLOW_TRACKING_PASSWORD'] = "ddec1d9afd9f6c362203803b1cee472f02892972"
#os.environ['MLFLOW_TRACKING_PROJECTNAME'] = input('Enter your DAGsHub project name: ')
os.environ['MLFLOW_TRACKING_PROJECTNAME'] = "glassDetection"
mlflow.set_tracking_uri(f'https://dagshub.com/' + os.environ['MLFLOW_TRACKING_USERNAME'] + '/' + os.environ['MLFLOW_TRACKING_PROJECTNAME'] + '.mlflow')

mlflow.start_run()


# IMPORT DATASET
dataset_used = "selfie"
mlflow.log_param("dataset_used", dataset_used)
random_state=1
mlflow.log_param("random_state", random_state)

with h5py.File("./data/Selfie_reduced/selfie_reduced.h5",'r') as data_aug:
  
  X = data_aug["img"][...] 
  aug_wearing_glasses = data_aug["wearing_glasses"][...] 
  aug_wearing_sunglasses = data_aug["wearing_sunglasses"][...] 

y = []
for i in range(len(aug_wearing_glasses)):
    if str(aug_wearing_glasses[i]) == '1' or str(aug_wearing_sunglasses[i]) == '1':
        y.append(1)
    else : y.append(0)


X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    train_size=0.8,
    test_size=0.2,
    random_state=random_state,
)

X_train = np.array(X_train)
X_valid = np.array(X_valid)
y_train = np.array(y_train)
y_valid = np.array(y_valid)

#Params MLFLOW for datasets
trainingSetSize = len(X_train)
validationSetSize = len(X_valid)

mlflow.log_param("trainingSetSize", trainingSetSize)
mlflow.log_param("validationSetSize", validationSetSize)


# DEFINING THE MODEL

#creation of the model 
import keras.utils
from keras import utils as np_utils

glasses_model = Sequential()

glasses_model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape = X_train[0].shape))
glasses_model.add(MaxPooling2D(pool_size=(2, 2)))
glasses_model.add(BatchNormalization())
glasses_model.add(Dropout(0.2))

glasses_model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
glasses_model.add(MaxPooling2D(pool_size=(2, 2)))
glasses_model.add(BatchNormalization())
glasses_model.add(Dropout(0.2))

glasses_model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
glasses_model.add(MaxPooling2D(pool_size=(2, 2)))
glasses_model.add(BatchNormalization())
glasses_model.add(Dropout(0.2))

glasses_model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
glasses_model.add(MaxPooling2D(pool_size=(2, 2)))
glasses_model.add(BatchNormalization())
glasses_model.add(Dropout(0.2))

glasses_model.add(Flatten())
glasses_model.add(Dense(128, activation='relu'))
glasses_model.add(Dropout(0.5))
glasses_model.add(Dense(64, activation='relu'))
glasses_model.add(Dropout(0.5))
glasses_model.add(Dense(1, activation='sigmoid'))


# Fit the model
glasses_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)

checkpoint_filepath_glasses = "./models/CNN/"
#checkpoint_filepath_glasses = "/content/drive/My Drive/AndrettaDibenedetto/Consegna/models/finalModelGlassDetection"


model_checkpoint_callback_glasses = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath_glasses, monitor='val_loss', mode='min', save_best_only=True)


glasses_model.summary() 
mlflow.tensorflow.autolog()

#FIT THE MODEL
glasses_model.fit(x=X_train, y=y_train, batch_size=32, epochs=1, verbose=1, validation_data=(X_valid, y_valid), callbacks=[callback, model_checkpoint_callback_glasses])


