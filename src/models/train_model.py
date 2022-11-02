"""Code used to create and train the model
"""
# pylint: disable=invalid-name
# Caricare le librerie

import os

import h5py
import mlflow
import mlflow.keras
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from sklearn.model_selection import train_test_split

dir = os.path.dirname(__file__)
filename_processed = os.path.join(
    dir, "..", "..", "data", "Selfie_reduced", "processed"
)


def create_train_val_sets():
    """This is a function used to create the training and validation set

    Returns:
        _type_: array
    """
    h5_path = os.path.join(filename_processed, "selfie_reduced.h5")

    with h5py.File(h5_path, "r") as data_aug:

        X = data_aug["img"][...]
        aug_wearing_glasses = data_aug["wearing_glasses"][...]
        aug_wearing_sunglasses = data_aug["wearing_sunglasses"][...]

    y = []
    for i, _ in enumerate(aug_wearing_glasses):
        if str(aug_wearing_glasses[i]) == "1" or str(aug_wearing_sunglasses[i]) == "1":
            y.append(1)
        else:
            y.append(0)

    random_state = 1

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

    return X_train, y_train, X_valid, y_valid


# DEFINING THE MODEL
def model_creation(X_train, loss, optimizer):
    """Function used to create a CNN

    Args:
        X_train (array): arrays of images
        loss (string): the loss used by the model
        optimizer (string): the optimizer used by the model

    Returns:
        Sequential: return the model created
    """
    glasses_model = Sequential()

    glasses_model.add(
        Conv2D(
            filters=16,
            kernel_size=(5, 5),
            activation="relu",
            input_shape=X_train[0].shape,
        )
    )
    glasses_model.add(MaxPooling2D(pool_size=(2, 2)))
    glasses_model.add(BatchNormalization())
    glasses_model.add(Dropout(0.2))

    glasses_model.add(Conv2D(filters=32, kernel_size=(5, 5), activation="relu"))
    glasses_model.add(MaxPooling2D(pool_size=(2, 2)))
    glasses_model.add(BatchNormalization())
    glasses_model.add(Dropout(0.2))

    glasses_model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
    glasses_model.add(MaxPooling2D(pool_size=(2, 2)))
    glasses_model.add(BatchNormalization())
    glasses_model.add(Dropout(0.2))

    glasses_model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
    glasses_model.add(MaxPooling2D(pool_size=(2, 2)))
    glasses_model.add(BatchNormalization())
    glasses_model.add(Dropout(0.2))

    glasses_model.add(Flatten())
    glasses_model.add(Dense(128, activation="relu"))
    glasses_model.add(Dropout(0.5))
    glasses_model.add(Dense(64, activation="relu"))
    glasses_model.add(Dropout(0.5))
    glasses_model.add(Dense(1, activation="sigmoid"))

    # Fit the model
    glasses_model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

    return glasses_model


# Function for training the model
def model_training(model, X, y, batch, epochs, verbose, X_val, y_val):
    """Function used to train a CNN

    Args:
        model (Sequential): The model that we want to train
        X (array): The training set used for the training
        y (array): The label of the train set
        batch (integer): the size of the batch
        epochs (integer): number of epochs used for the train
        verbose (integer): the number used for the early stopping
        X_val (array): the validation set
        y_val (array): labels of the validation set

    Returns:
        history, Sequential: return the history and model after the train
    """
    callback_train = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5, verbose=1
    )

    checkpoint_filepath_glasses = "./models/CNN/"

    model_checkpoint_callback_glasses = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath_glasses,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    )

    history = model.fit(
        x=X,
        y=y,
        batch_size=batch,
        epochs=epochs,
        verbose=verbose,
        validation_data=(X_val, y_val),
        callbacks=[callback_train, model_checkpoint_callback_glasses],
    )
    return history, model


def main():
    """The main of the code"""
    # ML FLOW PARAMS
    # from getpass import getpass

    # os.environ['MLFLOW_TRACKING_USERNAME'] = input('Enter your DAGsHub username: ')
    os.environ["MLFLOW_TRACKING_USERNAME"] = "GaetanoDibenedetto"
    # os.environ['MLFLOW_TRACKING_PASSWORD'] = getpass('Enter your DAGsHub access token: ')

    os.environ["MLFLOW_TRACKING_PASSWORD"] = "ddec1d9afd9f6c362203803b1cee472f02892972"
    # os.environ['MLFLOW_TRACKING_PROJECTNAME'] = input('Enter your DAGsHub project name: ')
    os.environ["MLFLOW_TRACKING_PROJECTNAME"] = "glassDetection"
    mlflow.set_tracking_uri(
        "https://dagshub.com/"
        + os.environ["MLFLOW_TRACKING_USERNAME"]
        + "/"
        + os.environ["MLFLOW_TRACKING_PROJECTNAME"]
        + ".mlflow"
    )

    mlflow.start_run()

    # IMPORT DATASET
    dataset_used = "selfie"
    mlflow.log_param("dataset_used", dataset_used)
    random_state = 1
    mlflow.log_param("random_state", random_state)

    # Creation of the sets used for the training of the model
    X_train, y_train, X_valid, y_valid = create_train_val_sets()

    # Params MLFLOW for datasets
    trainingsetsize = len(X_train)
    trainingsetsize = len(X_valid)

    mlflow.log_param("trainingSetSize", trainingsetsize)
    mlflow.log_param("validationSetSize", trainingsetsize)

    # Instantiation of the model
    glasses_model = model_creation(X_train, "binary_crossentropy", "adam")

    glasses_model.summary()

    mlflow.tensorflow.autolog(
        log_models=True,
        registered_model_name="GlassDect",
        disable=False,
        exclusive=False,
        disable_for_unsupported_versions=False,
        silent=False,
        log_input_examples=False,
        log_model_signatures=False,
    )

    _, _ = model_training(glasses_model, X_train, y_train, 32, 2, 1, X_valid, y_valid)

    mlflow.end_run()
    print("End procedure")


if __name__ == "__main__":
    main()
