# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=missing-module-docstring

# -*- coding: utf-8 -*-
import csv
import logging

# move files
import os
import random
import sys
import zipfile
from fileinput import filename
from pathlib import Path

import cv2
import h5py
import imutils
import numpy as np
from dotenv import find_dotenv, load_dotenv
from skimage.util import random_noise


dir = os.path.dirname(__file__)
sys.path.insert(1, dir)
from FaceAlignerNetwork import FaceAligner


def _blur_pass(img, sigmaX=None):
    sx = 0
    if sigmaX is not None:
        sx = sigmaX
    return cv2.GaussianBlur(img, (3, 3), sx)


def _noise_pass(img):
    float_img = random_noise(img, var=random.randrange(1, 11) * 0.002)
    return np.array(255 * float_img, dtype="uint8")


def _brightness_shift_pass(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    val = 0
    rand = random.randint(-80, 80)
    for x in range(v.shape[0]):
        for y in range(v.shape[1]):
            val = v[x][y]
            if rand >= 0:
                v[x][y] = min(255, val + rand)
            else:
                v[x][y] = max(0, val + rand)

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def _contrast_shift_pass(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=random.uniform(0.3, 4), tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final


def _rotate_pass(img):
    degree = random.randint(-90, 90)
    rotated = imutils.rotate(img, degree)
    return rotated


def _translation_pass(img):
    tx, ty = (random.randint(-20, 20), random.randint(-20, 20))
    translation_matrix = np.array([[1, 0, tx], [0, 1, ty]], dtype="float32")
    return cv2.warpAffine(img, translation_matrix, img.shape[:2])


def _horizontal_flip_pass(img):
    return cv2.flip(img, 1)


def _img_augmentation(img):
    augmented_imgs = []
    filp_new_img = np.copy(img)
    rotation_new_img = np.copy(img)
    brighnes_new_img = np.copy(img)

    filp_new_img = _horizontal_flip_pass(img)
    augmented_imgs.append(filp_new_img)

    rotation_new_img = _rotate_pass(img)
    augmented_imgs.append(rotation_new_img)

    brighnes_new_img = _brightness_shift_pass(img)
    augmented_imgs.append(brighnes_new_img)

    return augmented_imgs


def _face_alignment(path):

    IMG_SIZE = 227
    KERNEL = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    load_image = cv2.imread(path)
    face_aligner = FaceAligner(desiredLeftEye=(0.37, 0.28), desiredFaceWidth=IMG_SIZE)
    grey_image = cv2.cvtColor(load_image, cv2.COLOR_BGR2GRAY)
    load_image, _ = face_aligner.align(grey_image, load_image)
    load_image = cv2.filter2D(src=load_image, ddepth=-1, kernel=KERNEL)
    load_image = cv2.resize(load_image, (IMG_SIZE, IMG_SIZE))
    return load_image


def main():
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    data_image = []
    data_label1 = []
    data_label2 = []

    filename_raw = os.path.join(dir, "..", "..", "data", "Selfie_reduced", "raw")
    filename_processed = os.path.join(
        dir, "..", "..", "data", "Selfie_reduced", "processed"
    )
    sys.path.insert(0, filename_raw)

    zip_path = os.path.join(filename_raw, "Selfie-dataset.zip")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(filename_processed)

    csv_path = os.path.join(filename_processed, "selfie_dataset.csv")
    img_path = os.path.join(filename_processed, "images")
    with open(csv_path) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=";")
        i = 0
        for row in spamreader:
            if i == 0:
                print(row[0],row[19],row[20])
            if i > 0:
                load_image = _face_alignment(os.path.join(img_path, row[0] + ".jpg"))
                data_image.append(load_image)
                data_label1.append(int(row[19]))
                data_label2.append(int(row[20]))

                if str(row[19]) == "1" or str(row[20]) == "1":
                    imgs_augmented = _img_augmentation(load_image)
                    for img in imgs_augmented:
                        data_image.append(img)
                        data_label1.append(int(row[19]))
                        data_label2.append(int(row[20]))
                # counter = counter + 1

            i = i + 1
            if i == 100: #max size of the selfie_reduced dataset is 101
                break

    h5_path = os.path.join(filename_processed, "selfie_reduced.h5")
    hf = h5py.File(h5_path, "w")

    hf.create_dataset("img", data=data_image)
    hf.create_dataset("wearing_glasses", data=data_label1)
    hf.create_dataset("wearing_sunglasses", data=data_label2)
    hf.close()

    print("End procedure")


if __name__ == "__main__":
    LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
