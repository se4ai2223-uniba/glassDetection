"""Script for load testing the api
"""
import io
import os
import random
from locust import HttpUser, task, between
from PIL import Image


class TypicalIrisUser(HttpUser):
    """Class to model an user

    Args:
        HttpUser (string): the object user created
    """

    wait_time = between(1, 5)

    @task
    def sanity_check(self):
        """function to check the availability of the api"""
        self.client.get("/")

    @task(5)
    def cnn_prediction(self):
        """Function to send a post request to the api"""

        dir = os.path.dirname(__file__)

        rand = random.randint(0, 1)
        if rand == 0:
            image = os.path.join(dir, "..", "app", "test2.jpg")
        else:
            image = os.path.join(dir, "..", "app", "test_img.jpg")
        image_stream = [("maybeImage", open(image, "rb"))]

        self.client.post("/predict", files=image_stream)
