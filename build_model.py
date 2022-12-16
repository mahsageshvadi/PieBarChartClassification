import os
import pandas as pd
import numpy as np
from PIL import Image

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from keras.preprocessing.image import (ImageDataGenerator,
                                       img_to_array,
                                       array_to_img,
                                       load_img)

from sklearn.model_selection import train_test_split

from sklearn.metrics import (confusion_matrix,
                             classification_report,
                             accuracy_score,
                             f1_score,
                             roc_auc_score)

import tensorflow as tf
from keras.models import Model,Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense,Flatten

from keras import optimizers
from keras import regularizers
from keras.callbacks import EarlyStopping,LearningRateScheduler
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization


from keras import backend as K
K.clear_session()

from sklearn.utils import shuffle




config  = Config()

IMAGE_SIZE=(config.image_width, config.image_height)
IMAGE_CHANNELS= config.number_of_channels

train_dir = config.train_data_dir
test_dir = config.test_data_dir


def label(path):
    return [file.split('_')[0] for file in os.listdir(path)]

def filename(path):
    return [file for file in os.listdir(path)]
    
    

def get_train_validation_test():

    train_names = filename(train_dir)
    test_names = filename(test_dir)
    train_class = label(train_dir)
    
    test_class = label(test_dir)
    test_df = pd.DataFrame({'filename': test_names , 'category': test_class})
    
    train_df = pd.DataFrame({ 'filename': train_names, 'category': train_class})
    
    valid_df, train_df = train_test_split(train_df, test_size = (6/8))

    return train_df, valid_df, test_df

    

def get_model():

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(config.image_width, config.image_height, config.number_of_channels
                                                                )))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    
    
    return model


def train_model(model, train_df, valid_df, epochs, learning_rate, batch_size)
    
    train_map = ImageDataGenerator()
    valid_map = ImageDataGenerator()
    
    
    train_data = train_map.flow_from_dataframe(
            train_df,train_dir,
            x_col = 'filename',
            y_col = 'category',
            target_size = IMAGE_SIZE,
            batch_size = batch_size,
            class_mode = 'categorical')

    valid_data = valid_map.flow_from_dataframe(
             valid_df, train_dir,
             x_col = 'filename',
             y_col = 'category',
             target_size = IMAGE_SIZE,
             batch_size = batch_size,
             class_mode = 'categorical')
             
    
    loss = 'categorical_crossentropy'
    opt = tf.keras.optimizers.Adam(learning_rate= learning_rate)
    metrics = ['accuracy']

    train_images = train_df.shape[0]
    valid_images = valid_df.shape[0]
    model.compile(loss = loss, optimizer = opt, metrics = metrics)


    history = model.fit(train_data, epochs = epochs,
                              validation_data = valid_data,
                              validation_steps= valid_images//batch_size,
                              steps_per_epoch= train_images//batch_size)
                              
    return history
    


learning_rate = 0.001
batch_size = 132
epochs = 50

train_df, valid_df, test_df = get_train_validation_test()

model = get_model()

history = train_model(model, train_df, valid_df, epochs, learning_rate, batch_size)
