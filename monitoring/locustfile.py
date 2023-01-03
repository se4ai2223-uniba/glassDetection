"""Script for load testing the api
"""
import io
import os
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

        # url = "https://yfvpqbuhav.eu-west-1.awsapprunner.com"

        dir = os.path.dirname(__file__)
        image = Image.open(os.path.join(dir, "app", "test_img.jpg"))
        image_stream = [("maybeImage", image)]

        self.client.post("/predict", files=image_stream)
