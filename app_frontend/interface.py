"""
This module contains all the utilities for the initialization of a web app using the gradio library
"""
from http import HTTPStatus
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


TITLE = "Glasses Detection Web App"

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
    img_load = open(input_img, "rb")
    data = {data_tag: img_load}
    response = requests.post(url=url, files=data)

    json_response = json.loads(response.content)
    if response.status_code != HTTPStatus.OK:
        raise ValueError(response.status_code + ": " + json_response["message"])

    return json_response["message"]


iface = gr.Interface(
    classify_image,
    gr.components.Image(label="Input image", type="filepath", source="upload"),
    gr.components.Label(num_top_classes=5, label="Output prediction and confidence"),
    title=TITLE,
    description=DESCRIPTION,
    allow_flagging="never",
    examples=[
        os.path.join(os.path.dirname(__file__), "..", "app", "test_img.jpeg"),
        os.path.join(os.path.dirname(__file__), "..", "app", "test_img.jpg"),
        os.path.join(os.path.dirname(__file__), "..", "app", "test2.jpg"),
    ],
)

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
