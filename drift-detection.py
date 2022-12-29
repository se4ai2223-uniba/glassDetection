"""
Scripts for drift detection
"""
# pylint: disable=protected-access
# pylint: disable=redefined-builtin
# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=wrong-import-order

import os
import sys
import numpy as np
#--------- IMPORT FOR DATA DRIFT --------
from functools import partial

from alibi_detect.cd import MMDDrift
from alibi_detect.cd.tensorflow import preprocess_drift
from timeit import default_timer as timer
from keras.models import load_model

from imagecorruptions import corrupt


dir = os.path.dirname(__file__)
sys.path.insert(1, os.path.join(dir, "src", "models"))

from predict_model import create_test_set
from train_model import create_train_val_sets

#function for the creation of a noise sets
def create_noise_sets(train, val, test):
    """Function used to create train, valid and test sets with some noise

    Args:
        train (array): original train set
        val (array): original val set
        test (array): original test set

    Returns:
        array: the noise versions of the sets
    """
    train_noise = []
    val_noise = []
    test_noise = []

    for _, img in enumerate(train):
        gaussian_noise_imgs = corrupt(img, corruption_number=7, severity=5)
        train_noise.append(gaussian_noise_imgs)

    for _, img in enumerate(val):
        gaussian_noise_imgs = gaussian_noise_imgs = corrupt(img, corruption_number=7, severity=5)
        val_noise.append(gaussian_noise_imgs)

    for _, img in enumerate(test):
        gaussian_noise_imgs = gaussian_noise_imgs = corrupt(img, corruption_number=6, severity=5)
        test_noise.append(gaussian_noise_imgs)

    train_noise = np.array(train_noise)
    val_noise = np.array(val_noise)
    test_noise = np.array(test_noise)

    return train_noise, val_noise, test_noise

#----------------- IMPLEMENTING DRIFT -------------------------------------------------

# Create train and validation set
X_train, y_train, X_valid, y_valid = create_train_val_sets()

# Create the test set
X_test, y_test = create_test_set()

# Create a noise train and val set
X_train_noise, X_valid_noise, x_corr = create_noise_sets(X_train, X_valid, X_test)

# Loading the CNN
CHECKPOINT_FILEPATH_GLASSES = os.path.join(
    dir, "models", "finalModelGlassDetection255")

# Load best model from checkpoint
glasses_model = load_model(CHECKPOINT_FILEPATH_GLASSES)

# define preprocessing function
preprocess_fn = partial(preprocess_drift, model=glasses_model, batch_size=32)

# initialise drift detector
cd = MMDDrift(X_train, backend='tensorflow', p_val=.05,
              preprocess_fn=preprocess_fn, n_permutations=100)

labels = ['No!', 'Yes!']

corruption = ['motion_blur']

def make_predictions(cd, x_h0, x_corr, corruption):
    """Function for detecting the drift

    Args:
        cd : the drift detector
        x_h0 (list): original data
        x_corr (list): corrupted data
        corruption (list): the type of corruption
    """
    time_0 = timer()
    preds = cd.predict(x_h0)
    dt = timer() - time_0
    print('No corruption')
    print('Drift? {}'.format(labels[preds['data']['is_drift']]))
    print(f'p-value: {preds["data"]["p_val"]:.3f}')
    print(f'Time (s) {dt:.3f}')

    if isinstance(x_corr, list):
        for x, c in zip(x_corr, corruption):
            time_0 = timer()
            preds = cd.predict(x)
            dt = timer() - time_0
            print('')
            print(f'Corruption type: {c}')
            print('Drift? {}'.format(labels[preds['data']['is_drift']]))
            print(f'p-value: {preds["data"]["p_val"]:.3f}')
            print(f'Time (s) {dt:.3f}')


make_predictions(cd, X_test, x_corr, corruption)
