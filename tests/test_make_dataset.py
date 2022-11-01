"""
module that tests all functions in the make_dataset module
"""

# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=wrong-import-position
# pylint: disable=redefined-builtin
# pylint: disable=import-error

import csv
import os
import sys
import zipfile

import cv2
import numpy as np

dir = os.path.dirname(__file__)
sys.path.insert(1, os.path.join(dir, "..", "src", "data"))
from make_dataset import (
    _blur_pass,
    _brightness_shift_pass,
    _face_alignment,
    _horizontal_flip_pass,
    _img_augmentation,
    _noise_pass,
    _rotate_pass,
)


def test_blur_pass():
    """Check if an image has blur:
    If the variance falls below a pre-defined threshold,
    then the image is considered blurry; otherwise, the image is not blurry.
    """
    img_new = _blur_pass(load_image)
    var_blurred = cv2.Laplacian(img_new, cv2.CV_64F).var()
    var_original = cv2.Laplacian(load_image, cv2.CV_64F).var()
    assert var_original > var_blurred * 2


def hsv_saturation_percentage(img):
    """
    Convert image to HSV color space
    Calculate histogram of saturation channel
    Calculate percentage of pixels with saturation >= p

    Args:
        img ('uint8'): cv2 image on which we want check the number of
        pixels that are below a value of saturations

    Returns:
        'float': pecentage of pixels with saturation >= p
    """
    # Convert image to HSV color space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Calculate histogram of saturation channel
    s = cv2.calcHist([img], [1], None, [256], [0, 256])

    # Calculate percentage of pixels with saturation >= p
    p = 0.05
    return np.sum(s[int(p * 255) : -1]) / np.prod(img.shape[0:2])


def test_noise_pass():
    """Check if an image has noise respect to the original:
    comparing the pecentage of pixels with saturation >= p

    """
    img_new = _noise_pass(load_image)

    # HSV SATURATION CHECK
    s_perc_img_new = hsv_saturation_percentage(img_new)
    s_perc_img_original = hsv_saturation_percentage(load_image)

    assert s_perc_img_original > s_perc_img_new


def test_brightness_shift_pass():
    """Check if an image has only different birghtness than the original:
    comparing the average intensity of colors

    """
    img_new = _brightness_shift_pass(load_image)
    avg_color_img_new = cv2.mean(cv2.blur(img_new, (5, 5)))
    avg_color_img_original = cv2.mean(cv2.blur(load_image, (5, 5)))
    # intensity color should be different
    assert avg_color_img_new != avg_color_img_original


def test_face_alignment():
    """Check if an image is contained in another one,
    because the face alignment makes a focus on a target

    """
    img = cv2.imread(img_path_test)
    img_new = _face_alignment(img)

    method = cv2.TM_SQDIFF_NORMED

    result = cv2.matchTemplate(img_new, img, method)
    result_different_imgs = cv2.matchTemplate(load_image2, load_image, method)

    # if no boundy box has been found the shape will contains more than 1 value
    assert result.shape != (1, 1)
    assert result_different_imgs.shape == (1, 1)


def test_img_augmentation():
    """Checking if the augmentation fucntion makes equal image"""
    imgs_new = _img_augmentation(load_image)
    index = 0
    for img in imgs_new:
        index = index + 1
        difference = cv2.absdiff(img, load_image)
        assert np.any(difference)


def vertical_horizontal_histogram(img):
    """calculating the vertical and horizontal histogram

    Args:
        img ('uint8'): cv2 image on which we want calculate the histograms

    Returns:
        ['float']: array containing the value of each point of the histogram on the x
        ['float']: array containing the value of each point of the histogram on the y
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = 255 - gray
    x_sum = np.sum(gray, axis=1).tolist()
    y_sum = np.sum(gray, axis=0).tolist()
    return x_sum, y_sum


def test_horizontal_flip_pass():
    """check if an image is the horizontal flip of another one"""
    img_new = _horizontal_flip_pass(load_image)

    x_sum_original, y_sum_original = vertical_horizontal_histogram(load_image)
    x_sum_img_new, y_sum_img_new = vertical_horizontal_histogram(img_new)

    assert np.array_equal(x_sum_original, x_sum_img_new)
    assert np.array_equal(y_sum_original, y_sum_img_new[::-1])

    x_sum_different_img, y_sum_different_img = vertical_horizontal_histogram(
        load_image2
    )

    assert not np.array_equal(x_sum_original, x_sum_different_img)
    assert not np.array_equal(y_sum_original, y_sum_different_img[::-1])
    assert not np.array_equal(x_sum_img_new, x_sum_different_img)
    assert not np.array_equal(x_sum_img_new, y_sum_different_img[::-1])


def test_rotate_pass():
    """check if an image is the rotation of another one"""
    img_new = _rotate_pass(load_image)

    img_new = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)
    original_gray = cv2.cvtColor(load_image, cv2.COLOR_BGR2GRAY)

    original_not_black_pixels = cv2.countNonZero(original_gray)
    new_not_black_pixels = cv2.countNonZero(img_new)

    assert original_not_black_pixels >= new_not_black_pixels


IMG_SIZE = 227
filename_raw = os.path.join(dir, "..", "data", "Selfie_reduced", "raw")
filename_processed = os.path.join(dir, "..", "data", "Selfie_reduced", "processed")
sys.path.insert(0, filename_raw)

zip_path = os.path.join(filename_raw, "Selfie-dataset.zip")
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(filename_processed)

csv_path = os.path.join(filename_processed, "selfie_dataset.csv")
img_path = os.path.join(filename_processed, "images")
img_path_test = None
load_image = None
with open(csv_path, encoding="utf-8") as csvfile:
    spamreader = csv.reader(csvfile, delimiter=";")
    i = 0
    j = 15
    for row in spamreader:
        if i == j:
            img_path_test = os.path.join(img_path, row[0] + ".jpg")
            load_image2 = cv2.imread(img_path_test)
            load_image2 = cv2.resize(load_image2, (IMG_SIZE, IMG_SIZE))
        if i > j:
            img_path_test = os.path.join(img_path, row[0] + ".jpg")
            load_image = cv2.imread(img_path_test)
            load_image = cv2.resize(load_image, (IMG_SIZE, IMG_SIZE))

            break
        i = i + 1
