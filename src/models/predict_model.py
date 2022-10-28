# pylint: disable=missing-module-docstring
# Caricare le librerie
import os
import h5py
import mlflow
import mlflow.keras
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.models import load_model


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

# Params MLFLOW for datasets
TESTSETSIZE = len(X_test)

mlflow.log_param("testSetSize", TESTSETSIZE)

mlflow.tensorflow.autolog()

CHECKPOINT_FILEPATH_GLASSES = "./models/CNN/"

# Load best model from checkpoint
best_model_glasses = load_model(CHECKPOINT_FILEPATH_GLASSES)

# Get predictions
model_predictions = best_model_glasses.predict(X_test)

# Get int values from predictions
model_predictions = model_predictions.round()

# Print confusion matrix
conf_matrix_glasses = confusion_matrix(y_test, model_predictions)
print("glasses confusion matrix: ")
print(conf_matrix_glasses)

# Print the accuracy
accuracy_glasses = accuracy_score(y_test, model_predictions)
print("accuracy")
print(accuracy_glasses)
mlflow.log_metric("testset_accuracy", accuracy_glasses)

run_id = mlflow.last_active_run().info.run_id
artifacts = mlflow.artifacts.download_artifacts(CHECKPOINT_FILEPATH_GLASSES)

mlflow.sklearn.log_model(
    best_model_glasses, artifacts, registered_model_name="GlassDect"
)

mlflow.end_run()
