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
        images_data.append(np.load(save_dir + '/'+ filename))

    df = pd.DataFrame({
        'data': images_data,
        'category': categories
    })
    
    df = df.sample(frac = 1)
    
    df["category"] = df["category"].replace({'Pie': 0, 'Bar': 1})

    train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
    train_df = train_df.reset_index(drop=True)
    validate_df = validate_df.reset_index(drop=True)

    return train_df, validate_df
    


    


def get_CatDog_model():

    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(100, 100, 3)))
    model.add(MaxPool2D(strides=2))
    model.add(Conv2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu'))
    model.add(MaxPool2D(strides=2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.build()
    model.summary()


    return model




train_data, validation_data = get_generated_data_from_file()

Y_train = train_data.iloc[:, 1:]
X_train = train_data.iloc[:, 0]

X_train = np.array(X_train)
Y_train = np.array(Y_train)
# Normalize inputs
X_train = X_train / 255.0

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(train_data ,Y_train, steps_per_epoch = 10, epochs = 42)






