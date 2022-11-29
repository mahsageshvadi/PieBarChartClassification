# prepare data


import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os

from config import Config
config  = Config()

IMAGE_SIZE=(config.image_width, config.image_height)
IMAGE_CHANNELS= config.number_of_channels

def prepare_data():
    filenames = os.listdir("../Data/")
    categories = []
    for filename in filenames:
        category = filename.split('_')[0]
        if category == 'Bar':
            categories.append(1)
        else:
            categories.append(0)

    df = pd.DataFrame({
        'filename': filenames,
        'category': categories
    })
    print(df)





prepare_data()
