import os
import h5py
import numpy as np
import tensorflow as tf
from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout, Flatten,
                          MaxPooling2D)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.models import load_model


with h5py.File("./data/Selfie_reduced/processed/selfie_reduced.h5", "r") as data_aug:

    X = data_aug["img"][...]
    aug_wearing_glasses = data_aug["wearing_glasses"][...]
    aug_wearing_sunglasses = data_aug["wearing_sunglasses"][...]

y = []
for i, _ in enumerate(aug_wearing_glasses):
    if str(aug_wearing_glasses[i]) == "1" or str(aug_wearing_sunglasses[i]) == "1":
        y.append(1)
    else:
        y.append(0)


X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    train_size=0.8,
    test_size=0.2,
    random_state=0,
)

X_train = np.array(X_train)
X_valid = np.array(X_valid)
y_train = np.array(y_train)
y_valid = np.array(y_valid)


with h5py.File("./data/Selfie_reduced/processed/selfie_reduced.h5", "r") as data_aug:

    X_test = data_aug["img"][...]
    aug_wearing_glasses = data_aug["wearing_glasses"][...]
    aug_wearing_sunglasses = data_aug["wearing_sunglasses"][...]

y_test = []
for i, _ in enumerate(aug_wearing_glasses):
    if str(aug_wearing_glasses[i]) == "1" or str(aug_wearing_sunglasses[i]) == "1":
        y_test.append(1)
    else:
        y_test.append(0)


X_test = np.array(X_test)
y_test = np.array(y_test)

#---- Loading the CNN----

CHECKPOINT_FILEPATH_GLASSES = "./models/CNN/"

# Load best model from checkpoint
best_model_glasses = load_model(CHECKPOINT_FILEPATH_GLASSES)


#=================================
    # TESTING
#=================================

def test_model_return_vals():
    """
    Tests for the returned values of the modeling function
    """

    model_predictions = best_model_glasses.predict(X_test)

    # Get int values from predictions
    model_predictions = model_predictions.round()

    # Print the accuracy
    accuracy_glasses = accuracy_score(y_test, model_predictions)

    #=================================
    # TEST SUITE
    #=================================
    # Check returned scores' type
    assert isinstance(accuracy_glasses, float)
    # Check returned scores' range
    assert accuracy_glasses >= 0
    assert accuracy_glasses <= 1

def test_wrong_input_raises_assertion():
    """
    Tests for various assertion cheks written in the modeling function
    """
    filename = 'testing'
    scores = train_linear_model(X,y, filename=filename)

    #=================================
    # TEST SUITE
    #=================================
    # Test that it handles the case of: X is a string
    msg = train_linear_model('X',y)
    assert isinstance(msg, AssertionError)
    assert msg.args[0] == "X must be a Numpy array"
    # Test that it handles the case of: y is a string
    msg = train_linear_model(X,'y')
    assert isinstance(msg, AssertionError)
    assert msg.args[0] == "y must be a Numpy array"
