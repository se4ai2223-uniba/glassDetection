"""
    Scripts for testing the api
"""
# pylint: disable=protected-access
# pylint: disable=redefined-builtin
# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=wrong-import-order

from fastapi.testclient import TestClient
from http import HTTPStatus
from keras.models import load_model
from api import app
import numpy as np
import cv2
import os
import sys
import json

client = TestClient(app)

dir = os.path.dirname(__file__)

sys.path.insert(1, os.path.join(dir, "..", "src", "data"))
from make_dataset import _face_alignment

sys.path.insert(1, os.path.join(dir, "..", "src", "models"))
from predict_model import create_test_set

checkpoint_filepath_glasses = os.path.join(
    dir, "..", "models", "finalModelGlassDetection255"
)
model = load_model(checkpoint_filepath_glasses)


def test_image():

    """
    Function for testing the api
    """
    HOST_URL = os.environ.get("SSH_HOST")
    url = HOST_URL + ":8000/predict"
    path_image = os.path.join(dir, "test_img.jpg")
    data_tag = "maybeImage"
    data = {data_tag: open(path_image, "rb")}
    response = client.post(url=url, files=data)

    img = cv2.imread(path_image)
    img = _face_alignment(img)
    img_list = []
    img_list.append(img)
    img_list = np.array(img_list)

    prediction = model.predict(img_list)
    prediction = prediction.round()

    assert response.request.method == "POST"
    assert response.status_code == HTTPStatus.OK

    json_response = response.text
    json_response = json.loads(json_response)

    if prediction[0] == 1:
        assert json_response["message"] == "Glasses detected!"
    else:
        assert json_response["message"] == "Glasses NOT detected!"


def test_not_image():
    """Function for testing the input of the user"""

    HOST_URL = os.environ.get("SSH_HOST")
    url = HOST_URL + ":8000/predict"
    file = os.path.join(dir, "..", "requirements.txt")
    data_tag = "maybeImage"
    data = {data_tag: open(file, "rb")}
    response = client.post(url=url, files=data)

    assert response.request.method == "POST"
    assert response.status_code == HTTPStatus.NOT_ACCEPTABLE
    json_response = response.text
    json_response = json.loads(json_response)
    assert json_response["message"] == "Image needed!"
