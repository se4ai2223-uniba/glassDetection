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


checkpoint_filepath_glasses = os.path.join(
    dir, "..", "models", "finalModelGlassDetection255"
)
model = load_model(checkpoint_filepath_glasses)

def test_post():
    url = 'http://127.0.0.1:8000/predict/'
    path_image = 'test_img.jpg'
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
    assert response.json()["message"] in ["Glasses detected!", "Glasses NOT detected!"]
