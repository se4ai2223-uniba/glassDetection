"""
    Scripts for testing the api
"""
# pylint: disable=protected-access
# pylint: disable=redefined-builtin
# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=wrong-import-order

import requests
from http import HTTPStatus


def test_server():

    """
    Function for testing the api
    """

    url = "https://yfvpqbuhav.eu-west-1.awsapprunner.com/"
    response = requests.get(url=url)

    assert response.request.method == "GET"
    assert response.status_code == HTTPStatus.OK
