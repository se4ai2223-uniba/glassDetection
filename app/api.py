"""
module that provide an api for the project
"""
# pylint: disable=protected-access
# pylint: disable=redefined-builtin
# pylint: disable=import-error

import io
import os
import sys
from datetime import datetime
from functools import wraps
from http import HTTPStatus
import cv2

import numpy as np
from fastapi import Depends, FastAPI, File, Request, Response, UploadFile
from keras.models import load_model
from PIL import Image

dir = os.path.dirname(__file__)
sys.path.insert(1, os.path.join(dir))
from schemas import PredictPayload

sys.path.insert(1, os.path.join(dir, "..", "src", "models"))
from predict_model import create_test_set

checkpoint_filepath_glasses = os.path.join(dir, "..", "models", "CNN")
best_model_glasses = load_model(checkpoint_filepath_glasses)


img_set, label_set = create_test_set()
img = []
img.append(img_set[0])
img = np.array(img)
# Define application
app = FastAPI(
    title="Project for glass detection",
    description="This API lets you make a prediction wether a subject wears glass",
    version="0.1",
)


def construct_response(f):
    """Construct a JSON response for an endpoint's results."""

    @wraps(f)
    async def wrap(request: Request, response: Response, *args, **kwargs):
        results = await f(request, response, *args, **kwargs)

        # Construct response

        response.status_code = results["status-code"]
        _response = {
            "message": results["message"],
            "method": request.method,
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
        }

        # Add data
        if "data" in results:
            response["data"] = results["data"]

        return _response

    return wrap


@app.get("/", tags=["General"])  # path operation decorator
@construct_response
def _index(request: Request, response: Response):
    """Root endpoint."""

    response = {
        "message": HTTPStatus.OK.phrase,
        "method": request.method,
        "status-code": HTTPStatus.OK,
        "data": {"message": "Welcome to glasses classifier! Please, read the `/docs`!"},
    }
    return response


@app.post("/models", tags=["Prediction"])
@construct_response
# type is the name of the model
def _predict(request: Request, response: Response):
    """Classifies Iris flowers based on sepal and petal sizes."""

    # sklearn's `predict()` methods expect a 2D array of shape [n_samples, n_features]
    # therefore, we need to convert our single data point into a 2D array

    if best_model_glasses:

        prediction = best_model_glasses.predict(img)
        prediction = prediction.round()

        # we build a response

        response = {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "data": {
                "message": str(prediction),
                # "model-type": "CNN",
                # "prediction": prediction,
                # "predicted_type": type(prediction),
            },
        }
    else:
        response = {
            "message": "Model not found",
            "status-code": HTTPStatus.BAD_REQUEST,
        }
    return response


# @app.post("/uploadfile/", tags=["Upload"])
# @construct_response
# async def create_upload_file(file: UploadFile = File(...)):

#     if not file:
#         response = {
#             "message": "No upload file sent",
#             "status-code": HTTPStatus.BAD_REQUEST,
#         }
#     else:
#         response = {
#             "message": str(file.filename),
#         }

#     return response


@app.post("/prediction/", tags=["Prediction"])
@construct_response
async def prediction_route(
    request: Request,
    response: Response,
    file: PredictPayload = Depends(),
):

    image = file.maybeImage
    # CATCH PredictPayload exception
    if str(image) == "Image needed!":
        error_from_validation = str(image)
        return {
            "message": error_from_validation,
            "status-code": HTTPStatus.NOT_ACCEPTABLE,
        }

    content = image
    pil_image = Image.open(io.BytesIO(content))
    pil_image = pil_image.resize((227, 227))

    pil_image = pil_image.convert("RGB")

    numpy_image = np.array(pil_image) / 255.0
    # numpy_image = np.expand_dims(numpy_image, axis=0)

    # numpy_image.shape

    img1 = []
    img1.append(numpy_image)
    img1 = np.array(img1)

    prediction = best_model_glasses.predict(img1)
    prediction = prediction.round()

    if prediction[0] == 1:
        response = {
            "message": "You are wearing glasses!",
            "status-code": HTTPStatus.OK,
        }
    else:
        response = {
            "message": "You are not wearing glasses!",
            "status-code": HTTPStatus.OK,
        }

    return response
