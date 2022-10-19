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

# #ML FLOW PARAMS
# # get_python_version().system_raw("mlflow ui --port 5000 &")
# # from getpass import getpass

# #os.environ['MLFLOW_TRACKING_USERNAME'] = input('Enter your DAGsHub username: ')
# os.environ['MLFLOW_TRACKING_USERNAME'] = "GaetanoDibenedetto"
# #os.environ['MLFLOW_TRACKING_PASSWORD'] = getpass('Enter your DAGsHub access token: ')

# os.environ['MLFLOW_TRACKING_PASSWORD'] = "ddec1d9afd9f6c362203803b1cee472f02892972"
# #os.environ['MLFLOW_TRACKING_PROJECTNAME'] = input('Enter your DAGsHub project name: ')
# os.environ['MLFLOW_TRACKING_PROJECTNAME'] = "glassDetection"
# mlflow.set_tracking_uri(f'https://dagshub.com/' + os.environ['MLFLOW_TRACKING_USERNAME'] + '/' + os.environ['MLFLOW_TRACKING_PROJECTNAME'] + '.mlflow')

# mlflow.start_run()


# IMPORT TEST SET
dataset_used = "selfie"
mlflow.log_param("dataset_used", dataset_used)
random_state=1
mlflow.log_param("random_state", random_state)

with h5py.File("./data/Selfie_reduced/processed/selfie_reduced.h5",'r') as data_aug:
  
  X_test = data_aug["img"][...] 
  aug_wearing_glasses = data_aug["wearing_glasses"][...] 
  aug_wearing_sunglasses = data_aug["wearing_sunglasses"][...] 

y_test = []
for i in range(len(aug_wearing_glasses)):
    if str(aug_wearing_glasses[i]) == '1' or str(aug_wearing_sunglasses[i]) == '1':
        y_test.append(1)
    else : y_test.append(0)


X_test = np.array(X_test)
y_test = np.array(y_test)

#Params MLFLOW for datasets
testSetSize = len(X_test)

mlflow.log_param("testSetSize", testSetSize)

# # DEFINING THE MODEL

# #creation of the model 
# import keras.utils
# from keras import utils as np_utils

# glasses_model = Sequential()

# glasses_model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape = X_train[0].shape))
# glasses_model.add(MaxPooling2D(pool_size=(2, 2)))
# glasses_model.add(BatchNormalization())
# glasses_model.add(Dropout(0.2))

# glasses_model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
# glasses_model.add(MaxPooling2D(pool_size=(2, 2)))
# glasses_model.add(BatchNormalization())
# glasses_model.add(Dropout(0.2))

# glasses_model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
# glasses_model.add(MaxPooling2D(pool_size=(2, 2)))
# glasses_model.add(BatchNormalization())
# glasses_model.add(Dropout(0.2))

# glasses_model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
# glasses_model.add(MaxPooling2D(pool_size=(2, 2)))
# glasses_model.add(BatchNormalization())
# glasses_model.add(Dropout(0.2))

# glasses_model.add(Flatten())
# glasses_model.add(Dense(128, activation='relu'))
# glasses_model.add(Dropout(0.5))
# glasses_model.add(Dense(64, activation='relu'))
# glasses_model.add(Dropout(0.5))
# glasses_model.add(Dense(1, activation='sigmoid'))


# # Fit the model
# glasses_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)

# checkpoint_filepath_glasses = "./models/CNN/"
# #checkpoint_filepath_glasses = "/content/drive/My Drive/AndrettaDibenedetto/Consegna/models/finalModelGlassDetection"


# model_checkpoint_callback_glasses = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath_glasses, monitor='val_loss', mode='min', save_best_only=True)


# glasses_model.summary() 
# mlflow.tensorflow.autolog()

checkpoint_filepath_glasses = "./models/CNN/"

# Load best model from checkpoint
best_model_glasses = load_model(checkpoint_filepath_glasses)

# Get predictions
model_predictions = best_model_glasses.predict(X_test)

# Get int values from predictions
model_predictions = model_predictions.round()

# Print confusion matrix
conf_matrix_glasses = confusion_matrix(y_test, model_predictions)
print("glasses confusion matrix: ")
print(conf_matrix_glasses)

#Print the accuracy
accuracy_glasses = accuracy_score(y_test, model_predictions)
print("accuracy")
print(accuracy_glasses)
mlflow.log_metric("accuracy", accuracy_glasses)

mlflow.end_run()
