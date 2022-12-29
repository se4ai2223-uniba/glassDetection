import os
import sys

import numpy as np
import pytest
from keras.models import load_model
from sklearn.metrics import accuracy_score

dir = os.path.dirname(__file__)
sys.path.insert(1, os.path.join(dir, "tests"))

from test_model import create_noise_sets

dir = os.path.dirname(__file__)
sys.path.insert(1, os.path.join(dir, "src", "models"))

from predict_model import compute_model_accuracy, compute_predictions, create_test_set
from train_model import create_train_val_sets, model_creation, model_training

sys.path.insert(1, os.path.join(dir, "src", "data"))
from make_dataset import (
    _blur_pass,
    _brightness_shift_pass,
    _horizontal_flip_pass,
    _noise_pass,
)

#--------- IMPORT FOR DATA DRIFT --------
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from alibi_detect.cd import MMDDrift
from alibi_detect.models.tensorflow import scale_by_instance
from alibi_detect.utils.fetching import fetch_tf_model
from alibi_detect.saving import save_detector, load_detector
from alibi_detect.datasets import fetch_cifar10c, corruption_types_cifar10c
from alibi_detect.cd.tensorflow import preprocess_drift


from timeit import default_timer as timer