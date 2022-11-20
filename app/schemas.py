import io
from http import HTTPStatus
from typing import List, Optional, Literal

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
