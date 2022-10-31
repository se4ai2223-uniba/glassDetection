# pylint: disable=invalid-name
# pylint: disable=wrong-import-position
# pylint: disable=redefined-builtin
# pylint: disable=import-error
# pylint: disable=missing-function-docstring

import os
import sys

import numpy as np
import pytest
from keras.models import load_model
from sklearn.metrics import accuracy_score

dir = os.path.dirname(__file__)
sys.path.insert(1, os.path.join(dir, "..", "src", "models"))

from predict_model import compute_model_accuracy, compute_predictions, create_test_set
from train_model import create_train_val_sets, model_creation, model_training

sys.path.insert(1, os.path.join(dir, "..", "src", "data"))
from make_dataset import (
    _blur_pass,
    _brightness_shift_pass,
    _horizontal_flip_pass,
    _noise_pass,
)

# =================================
# TESTING
# =================================

# ------ Testing the function for the test set -----
def test_create_test_set():

    img_set, label_set = create_test_set()

    assert len(img_set) > 0
    assert len(label_set) > 0
    assert len(label_set) == len(img_set), "Test"


# ------ Testing the function for the train and validation set -----
def test_create_train_val_sets():

    img_train, label_train, img_val, label_val = create_train_val_sets()

    assert len(img_train) > 0
    assert len(label_train) > 0
    assert len(img_val) > 0
    assert len(label_val) > 0
    assert len(label_train) == len(img_train)
    assert len(label_val) == len(img_val), "Train"


# Testing the input given at the model
def test_input_shape():

    shape1 = X_train.shape
    shape2 = best_model_glasses.input_shape

    assert shape1[1:] == shape2[1:], "Input Shape"


# test for checking the reduction of the val_loss at each epoch
def test_reduction_val_loss():
    assert all(earlier >= later for earlier, later in zip(epoch_loss, epoch_loss[1:]))


# testing the overfit on a batch
def test_overfit_batch():

    X_batch = X_train[:32]
    y_batch = y_train[:32]

    trained, _ = model_training(
        glasses_model, X_batch, y_batch, 8, 1, 1, X_valid, y_valid
    )

    accuracy = trained.history["val_accuracy"]

    assert not accuracy[0] == pytest.approx(0.95, abs=0.05)


# testing the aspect values by the model
def test_model_return_vals():
    """
    Tests for the returned values of the modeling function
    """

    # Print the accuracy
    accuracy_glasses = accuracy_score(y_test, model_predictions)

    # Check returned scores' type
    assert isinstance(accuracy_glasses, float)
    # Check returned scores' range
    assert accuracy_glasses >= 0
    assert accuracy_glasses <= 1


# Directional testing for the training
def test_noise_impact_train():

    accuracy = train_history.history["val_accuracy"]
    accuracy_noise = train_history_noise.history["val_accuracy"]

    # Check that the accuracy on the val_set is less in the case without noise
    assert accuracy[-1] >= accuracy_noise[-1]


# Directional testing for the test


def test_noise_impact_test():

    predicts = compute_predictions(model, X_test)
    predicts_noise = compute_predictions(model_noise, X_test)

    # compute the model accuracy
    accuracy = compute_model_accuracy(y_test, predicts)
    accuracy_noise = compute_model_accuracy(y_test, predicts_noise)

    # Check that the accuracy on the test_set is less in the case without noise
    assert accuracy >= accuracy_noise


def create_noise_sets(train, val, test):

    train_noise = []
    val_noise = []
    test_noise = []

    for _, img in enumerate(train):
        gaussian_noise_imgs = _blur_pass(img)
        train_noise.append(gaussian_noise_imgs)

    for _, img in enumerate(val):
        gaussian_noise_imgs = _blur_pass(img)
        val_noise.append(gaussian_noise_imgs)

    for _, img in enumerate(test):
        gaussian_noise_imgs = _blur_pass(img)
        test_noise.append(gaussian_noise_imgs)

    train_noise = np.array(train_noise)
    val_noise = np.array(val_noise)
    test_noise = np.array(test_noise)

    return train_noise, val_noise, test_noise


def test_invariance_testing():
    new_test = []
    new_test.append(X_test[0])
    new_test.append(_noise_pass(X_test[0]))
    new_test.append(_blur_pass(X_test[0]))
    new_test.append(_brightness_shift_pass(X_test[0]))
    new_test.append(_horizontal_flip_pass(X_test[0]))
    new_test = np.array(new_test)
    results = best_model_glasses.predict(new_test)
    results = results.round()
    for _, prediction in enumerate(results):
        assert results[0] == prediction


# Create train and validation set
X_train, y_train, X_valid, y_valid = create_train_val_sets()

# Create the test set
X_test, y_test = create_test_set()

# Create a noise train and val set
X_train_noise, X_valid_noise, X_test_noise = create_noise_sets(X_train, X_valid, X_test)

# creatinf the model -- model_creation(X_train, loss, optimizer)
glasses_model = model_creation(X_train, "binary_crossentropy", "adam")

glasses_model_noise = model_creation(X_train, "binary_crossentropy", "adam")

# fitting the model -- model_training(model, X, y, batch, epochs, verbose, X_val, y_val)
history, _ = model_training(glasses_model, X_train, y_train, 32, 2, 1, X_valid, y_valid)

epoch_loss = history.history["val_loss"]
# Loading the CNN
CHECKPOINT_FILEPATH_GLASSES = "./models/CNN/"

# Load best model from checkpoint
best_model_glasses = load_model(CHECKPOINT_FILEPATH_GLASSES)

# loading the best model
model_predictions = best_model_glasses.predict(X_test)

# Get int values from predictions
model_predictions = model_predictions.round()

train_history, model = model_training(
    glasses_model, X_train, y_train, 32, 2, 1, X_valid, y_valid
)
train_history_noise, model_noise = model_training(
    glasses_model, X_train_noise, y_train, 32, 2, 1, X_valid, y_valid
)
