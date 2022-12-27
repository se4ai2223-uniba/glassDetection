import time
from locust import HttpUser, task, between
import os
import cv2
import numpy as np
import io
from PIL import Image

class TypicalIrisUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def sanity_check(self):
        self.client.get("/")

    @task(5)
    def CNN_prediction(self):

        #url = "https://yfvpqbuhav.eu-west-1.awsapprunner.com"

        dir = os.path.dirname(__file__)
        image = Image.open(os.path.join(dir, "app",  "test_img.jpg"))

        stream = io.BytesIO()
        image.save(stream, 'png')
        stream.seek(0)

        image_stream = [('maybeImage', stream)]

        self.client.post("/predict", files=image_stream)