# pylint: disable=no-name-in-module
# pylint: disable=too-few-public-methods
# pylint: disable=no-self-argument
"""
    Scripts for defining the schemas of the input

    Returns:
        _type_: test2
"""
import io

from pydantic import BaseModel, validator
from fastapi import UploadFile, File

import PIL
from PIL import Image


class PredictPayload(BaseModel):

    # Same name as parameter of body request
    maybeImage: UploadFile = File(...)

    @validator("*")
    def is_image(cls, v):
        contents = v.file.read()
        try:
            Image.open(io.BytesIO(contents))
        except PIL.UnidentifiedImageError:
            return "Image needed!"

        return contents
