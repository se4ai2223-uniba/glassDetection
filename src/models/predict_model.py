"""Code used to predict and evaluate the model
"""
# pylint: disable=invalid-name

# Caricare le librerie
import os
import h5py
import mlflow
import mlflow.keras
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.models import load_model

dir = os.path.dirname(__file__)
filename_processed = os.path.join(
    dir, "..", "..", "data", "Selfie_reduced", "processed"
)


def create_test_set():
    """Function used to create the test set

    Returns:
        array: test set
    """
    h5_path = os.path.join(filename_processed, "selfie_reduced.h5")

    with h5py.File(h5_path, "r") as data_aug:

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

    return X_test, y_test


# Function for computing the predictions of the models
def compute_predictions(best_model_glasses, X_test):
    """Function used to compute the predictions of a model

    Args:
        best_model_glasses (Sequential): Model used to make the predictions
        X_test (array): set used for the predictions

    Returns:
        array: the array of predictions done by the model
    """
    # Get predictions
    model_predictions = best_model_glasses.predict(X_test)

    # Get int values from predictions
    model_predictions = model_predictions.round()

    return model_predictions


def print_confusion_matrix(y_test, model_predictions):
    """Function used to print a confusion matrix of the predictions

    Args:
        y_test (array): labels of the test set
        model_predictions (array): labels of the predictions done by the mode
    """
    # Print confusion matrix
    conf_matrix_glasses = confusion_matrix(y_test, model_predictions)
    print("glasses confusion matrix: ")
    print(conf_matrix_glasses)


def compute_model_accuracy(y_test, model_predictions):
    """Function used to compute the accuracy of the model

    Args:
        y_test (array): Label of the test set
        model_predictions (array): Predictions made by the model

    Returns:
        float: final accuracy of the model
    """
    return accuracy_score(y_test, model_predictions)


def main():
    """main function of the code"""
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

    X_test, y_test = create_test_set()

    # Params MLFLOW for datasets
    testsetsize = len(X_test)

    mlflow.log_param("testSetSize", testsetsize)

    mlflow.tensorflow.autolog()

    checkpoint_filepath_glasses = "./models/CNN/"

    # Load best model from checkpoint
    best_model_glasses = load_model(checkpoint_filepath_glasses)

    # compute the predictions
    model_predictions = compute_predictions(best_model_glasses, X_test)

    # printing the confusion matrix
    print_confusion_matrix(y_test, model_predictions)

    print("1")
    print("1")
    # compute the model accuracy
    accuracy_glasses = compute_model_accuracy(y_test, model_predictions)

    print("2")
    print("2")
    mlflow.log_metric("testset_accuracy", accuracy_glasses)

    print("3")
    print("3")
    artifacts = mlflow.artifacts.download_artifacts(checkpoint_filepath_glasses)

    print("4")
    print("4")
    mlflow.sklearn.log_model(
        best_model_glasses, artifacts, registered_model_name="GlassDect"
    )
    print("5")
    print("5")
    mlflow.end_run()


if __name__ == "__main__":
    main()
