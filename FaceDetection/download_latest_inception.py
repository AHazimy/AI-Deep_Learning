# Trusty machine learning imports
import tensorflow as tf
import numpy as np
# Make sure to run notebook within slim folder
from datasets import dataset_utils
import os
# Base url
TF_MODELS_URL = "http://download.tensorflow.org/models/"
# Modify this path for a different CNN
INCEPTION_V3_URL = TF_MODELS_URL + "inception_v3_2016_08_28.tar.gz"
# Directory to save model checkpoints
MODELS_DIR = "models/cnn"
INCEPTION_V3_CKPT_PATH = MODELS_DIR + "/inception_v3.ckpt"
# Make the model directory if it does not exist
if not tf.gfile.Exists(MODELS_DIR):
 tf.gfile.MakeDirs(MODELS_DIR)
 
# Download the appropriate model if haven't already done so
if not os.path.exists(INCEPTION_V3_CKPT_PATH): 
 dataset_utils.download_and_uncompress_tarball(INCEPTION_V3_URL, MODELS_DIR)