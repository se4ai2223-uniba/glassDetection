"""
    Scripts for testing the api
"""
# pylint: disable=protected-access
# pylint: disable=redefined-builtin
# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=wrong-import-order

import os
import requests
from http import HTTPStatus


def test_server():

    """
    Function for testing the api
    """
    url = "ec2-34-246-171-145.eu-west-1.compute.amazonaws.com:8000/"
    response = requests.get(url=url)

    assert response.request.method == "GET"
    assert response.status_code == HTTPStatus.OK
