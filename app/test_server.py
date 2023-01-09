"""
    Scripts to test the api's server
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
    HOST_URL = os.environ.get("SSH_HOST")
    url = "http://" + HOST_URL + "/"
    response = requests.get(url=url)

    assert response.request.method == "GET"
    assert response.status_code == HTTPStatus.OK
