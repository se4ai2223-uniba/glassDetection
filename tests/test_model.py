import os
import h5py
import sys
import numpy as np
import tensorflow as tf
from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout, Flatten,
                          MaxPooling2D)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.models import load_model

dir = os.path.dirname(__file__)
sys.path.insert(1, os.path.join(dir, "..", "src", "models"))

from predict_model import create_test_set

from train_model import create_train_val_sets, model_creation, model_training

#Create train and validation set
X_train, y_train, X_valid, y_valid = create_train_val_sets()

# Create the test set
X_test, y_test = create_test_set()

#creatinf the model -- model_creation(X_train, loss, optimizer)
glasses_model = model_creation(X_train, "binary_crossentropy", "adam")

#fitting the model -- model_training(model, X, y, batch, epochs, verbose, X_val, y_val)
model_training(glasses_model, X_train, y_train, 32, 1, 1, X_valid, y_valid)

# Loading the CNN
CHECKPOINT_FILEPATH_GLASSES = "./models/CNN/"

# Load best model from checkpoint
best_model_glasses = load_model(CHECKPOINT_FILEPATH_GLASSES)




#=================================
    # TESTING
#=================================

# ------ Testing the function for the test set -----
def test_create_test_set():
    
    X_test, y_test = create_test_set()

    assert len(X_test) > 0
    assert len(y_test) > 0
    assert len(y_test) == len(X_test)


# ------ Testing the function for the train and validation set -----
def test_create_test_set():
    
    X_train, y_train, X_val, y_val = create_train_val_sets()

    assert len(X_train) > 0
    assert len(y_train) > 0
    assert len(X_val) > 0
    assert len(y_val) > 0
    assert len(y_train) == len(X_train)
    assert len(y_val) == len(X_val)



# def test_model_return_vals():
#     """
#     Tests for the returned values of the modeling function
#     """

#     model_predictions = best_model_glasses.predict(X_test)

#     # Get int values from predictions
#     model_predictions = model_predictions.round()

#     # Print the accuracy
#     accuracy_glasses = accuracy_score(y_test, model_predictions)

#     #=================================
#     # TEST SUITE
#     #=================================
#     # Check returned scores' type
#     assert isinstance(accuracy_glasses, float)
#     # Check returned scores' range
#     assert accuracy_glasses >= 0
#     assert accuracy_glasses <= 1

