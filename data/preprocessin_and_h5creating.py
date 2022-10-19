
from fileinput import filename
import random
import imutils
from skimage.util import random_noise

import zipfile
import numpy as np
import h5py
import os
import csv
import cv2
import glob
#move files
import os, shutil
import sys

dir = os.path.dirname(__file__)

filename=os.path.join(dir, "..", "src")
sys.path.insert(0, filename)
from FaceAlignerNetwork import FaceAligner



def _blur_pass(img, sigmaX = None):
    sx = 0
    if sigmaX is not None:
        sx = sigmaX
    return cv2.GaussianBlur(img, (3,3), sx)
 
def _noise_pass(img):
    float_img = random_noise(img, var= random.randrange(1,11) * 0.002)
    return np.array(255*float_img, dtype = 'uint8')
 
def _brightness_shift_pass(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    val = 0
    rand = random.randint(-80,80)
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
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=random.uniform(0.3,4), tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final
 
def _rotate_pass(img):
    degree = random.randint(-90,90)
    rotated = imutils.rotate(img, degree)
    return rotated
def _translation_pass(img):
    tx,ty = (random.randint(-20,20), random.randint(-20,20))
    translation_matrix = np.array([
        [1,0,tx],
        [0,1,ty]
    ], dtype="float32")
    return cv2.warpAffine(img,translation_matrix, img.shape[:2])
 
def _horizontal_flip_pass(img):
    return cv2.flip(img,1)


kernel = np.array([[-1, -1, -1],

                   [-1, 9,-1],

                   [-1, -1, -1]])
img_size = 227
data_image = []
data_label1 = []
data_label2 = []

filename=os.path.join(dir, "..", "data","Selfie_reduced")
sys.path.insert(0, filename)

zip_path = os.path.join(filename ,"Selfie-dataset.zip")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(filename)

csv_path = os.path.join(filename ,"selfie_dataset.csv")
img_path = os.path.join(filename ,"images")
with open(csv_path) as csvfile:
    spamreader = csv.reader(csvfile, delimiter=';')
    i = 0
    for row in spamreader:
        if i > 0:
            load_image = cv2.imread(os.path.join(img_path ,row[0]+".jpg"))
            face_aligner = FaceAligner(desiredLeftEye=(0.37,0.28),desiredFaceWidth=img_size)
            grey_image = cv2.cvtColor(load_image, cv2.COLOR_BGR2GRAY)
            load_image, _ = face_aligner.align(grey_image, load_image)
            load_image = cv2.filter2D(src=load_image, ddepth=-1, kernel=kernel)
            load_image = cv2.resize(load_image, (img_size,img_size))
            data_image.append(load_image)
            data_label1.append(row[19])
            data_label2.append(row[20])

            if str(row[19]) == '1' or str(row[20])== '1':
                filp_new_img = np.copy(load_image)
                rotation_new_img = np.copy(load_image)
                brighnes_new_img = np.copy(load_image)

                filp_new_img = _horizontal_flip_pass(load_image)
                data_image.append(filp_new_img)
                data_label1.append(row[19])
                data_label2.append(row[20])

                rotation_new_img = _rotate_pass(load_image)
                data_image.append(rotation_new_img)
                data_label1.append(row[19])
                data_label2.append(row[20])

                brighnes_new_img = _brightness_shift_pass(load_image)
                data_image.append(brighnes_new_img)
                data_label1.append(row[19])
                data_label2.append(row[20])
                # counter = counter + 1


        i = i+1
        if i == 101 :
            break

h5_path = os.path.join(filename ,"selfie_reduced.h5")
hf = h5py.File(h5_path, 'w')

hf.create_dataset('img', data=data_image)
hf.create_dataset('wearing_glasses', data=data_label1)
hf.create_dataset('wearing_sunglasses', data=data_label2)
hf.close()
print("End procedure")
