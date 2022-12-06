from fastapi.testclient import TestClient
from http import HTTPStatus
from PIL import Image
from keras.models import load_model
from api import app
import base64
import numpy as np
import cv2
import os
import io
import sys
import json
from make_dataset import _face_alignment

client = TestClient(app)

dir = os.path.dirname(__file__)
sys.path.insert(1, os.path.join(dir))


sys.path.insert(1, os.path.join(dir, "..", "src", "models"))
from predict_model import create_test_set

checkpoint_filepath_glasses = os.path.join(
    dir, "..", "models", "finalModelGlassDetection255"
)
model = load_model(checkpoint_filepath_glasses)

def test_image():

    url = 'http://127.0.0.1:8000/predict'
    path_image = os.path.join(dir,'test_img.jpg')
    dataTag = 'maybeImage'
    data = {dataTag: open(path_image, 'rb')}
    response = client.post(url=url, files=data)    

    img = cv2.imread(path_image)
    img = _face_alignment(img)
    img_list = []
    img_list.append(img)
    img_list = np.array(img_list)

    prediction = model.predict(img_list)
    prediction = prediction.round()
    
    assert response.request.method == "POST"
    assert response.status_code ==  HTTPStatus.OK 

    jsonResponse = response.text
    jsonResponse  = json.loads(jsonResponse)

    if prediction[0] == 1:
        assert jsonResponse["message"] == "Glasses detected!"
    else:
        assert jsonResponse["message"] == "Glasses NOT detected!"

def test_not_image():
    
    url = 'http://127.0.0.1:8000/predict'
    file = os.path.join(dir,'..','requirements.txt')
    dataTag = 'maybeImage'
    data = {dataTag: open(file, 'rb')}
    response = client.post(url=url, files=data)    

    
    assert response.request.method == "POST"
    assert response.status_code ==  HTTPStatus.NOT_ACCEPTABLE 
    jsonResponse = response.text
    jsonResponse  = json.loads(jsonResponse)
    assert jsonResponse["message"] == "Image needed!"
