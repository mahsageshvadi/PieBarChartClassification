# prepare data

import matplotlib.pyplot as plt
import random
import os
import numpy as np
import pandas as pd
import tensorflow as tf


from keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.utils import load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

from config import Config
from image_generator import generate_data


config  = Config()

IMAGE_SIZE=(config.image_width, config.image_height)
IMAGE_CHANNELS= config.number_of_channels

save_dir = config.train_data_dir


def get_generated_data_from_file():
    filenames = os.listdir(save_dir)

    categories = []
    images_data = []

    for filename in filenames:
        category = filename.split('_')[0]
        if category == 'Bar':
            categories.append(1)
        else:
            categories.append(0)
        images_data.append(np.load(filename))

    df = pd.DataFrame({
        'data': images_data,
        'category': categories
    })
    
    df = df.sample(frac = 1)
    
    df["category"] = df["category"].replace({0: 'Pie', 1: 'Bar'})

    train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
    train_df = train_df.reset_index(drop=True)
    validate_df = validate_df.reset_index(drop=True)

    return train_df, validate_df
    


    


def get_CatDog_model():

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(config.image_width, config.image_height, config.number_of_channels)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax')) # 2 because we have cat and dog classes

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    model.summary()

    return model




train_data, validation_data = get_generated_data_from_file()

#train_data, validation_data  = get_generated_data()
#model = get_CatDog_model()

print(train_data.head())
