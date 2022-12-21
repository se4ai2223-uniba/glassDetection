"""
This module contains all the utilities for the initialization of a web app using the gradio library
"""
import io
import json
import os
import sys
import time

import requests
import gradio as gr
import cv2
from PIL import Image
import numpy as np
from fastapi import FastAPI
from fastapi import File, UploadFile

CUSTOM_PATH = "/frontend"
dir = os.path.dirname(__file__)
app = FastAPI()

@app.get("/")
def read_main():
    return {"message": "This is the Glasses-detection frontend."}

TITLE = 'Glasses Detection Web App'

DESCRIPTION = """
<p>
<center>
This app tells you whether the subject in your photo is wearing glasses or not.
</center>
</p>
"""


def classify_image(input_img):
    """
    Main function used by gradio in order to produce the output for a certain input
    Args:
        input_img: numpy array image which will be passed to the predict method of the API
    """

    if input_img is None:
        return {}

    url = "http://localhost:8000/predict"
    data_tag = "maybeImage"
    img = Image.fromarray(input_img)
    temp_path=os.path.join(dir,"test")
    image_path = os.path.join(temp_path,"test_img.jpg")
    if not os.path.isdir(temp_path):
        # if the temp_path directory is 
        # not present then create it.
        os.makedirs(temp_path)
    img.save("test_img.jpg")
    img_load =  open("test_img.jpg", "rb")
    data = {data_tag: img_load}
    response = requests.post(url=url, files=data).content

    json_response = json.loads(response)
    print(json_response)
    # if json_response["status-code"] != 200:
    #     raise ValueError(json_response["message"])

    return json_response["message"]


iface = gr.Interface(
        classify_image,
        gr.components.Image(label="Input image"),
        gr.components.Label(num_top_classes=5, label="Output prediction and confidence"),
        title=TITLE,
        description=DESCRIPTION,
        allow_flagging="never")

iface.dev_mode = False
iface.config = iface.get_config_file()
gradio_app = gr.routes.App.create_app(iface)
app.mount(CUSTOM_PATH, gradio_app)


@gradio_app.on_event("startup")
async def _startup():
    backoff_time = 15
    max_retries = 5

    current_retry = 1
    while True:
        if current_retry == max_retries:
            raise ConnectionError("Web API is not listening!")

        current_retry += 1
        time.sleep(backoff_time)