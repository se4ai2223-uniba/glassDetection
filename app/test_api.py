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
    path_image = 'test2.jpg'
    url = 'http://127.0.0.1:8000/predict/'
    path_image = 'test_img.jpeg'
    dataTag = 'maybeImage'
    data = {dataTag: open(path_image, 'rb')}
    response = client.post(url=url, files=data)    

    img = Image.open(path_image)
    img.load()
    nparr = np.array(img)
    nparr = np.expand_dims(nparr, axis=0)

    img = _face_alignment(nparr[0])
    img_list = []
    img_list.append(img)
    img_list = np.array(img_list)

    prediction = model.predict(img_list)
    prediction = prediction.round()
    
    assert response.request.method == "POST"
    assert response.status_code ==  HTTPStatus.OK 

    jsonResponse = json.loads(response.json())

    if prediction[0] == 1:
        assert jsonResponse["message"] == "Glasses detected!"
    else:
        assert jsonResponse["message"] == "Glasses NOT detected!"
    


test_image()