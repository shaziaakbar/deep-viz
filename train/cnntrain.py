'''
File: 	    cnntrain.py
Author: 	Shazia Akbar

Description:
Dummy code for training a very simple CNN.
'''

import tensorflow as tf
sess = tf.Session()

import keras.backend as K
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Conv2D, Flatten, Dense, Dropout, MaxPool2D
from keras.models import load_model
from keras.utils import to_categorical
from keras.optimizers import Adam

import numpy as np
import h5py

# define input image dimensions
img_rows, img_cols, img_chns = 28, 28, 1
batch_size = 100
if K.image_data_format() == 'channels_first':
    original_img_size = (img_chns, img_rows, img_cols)
else:
    original_img_size = (img_rows, img_cols, img_chns)

epochs = 10

# define layers of CNN
x = Input(shape=original_img_size)
conv1 = Conv2D(32, (3, 3), activation='relu', name='conv1')(x)
conv2 = Conv2D(64, (3, 3), activation='relu', name='conv2')(conv1)
pool1 = MaxPool2D((2, 2), name='pool1')(conv2)
drop1 = Dropout(0.25, name='dropout1')(pool1)
flatten = Flatten()(drop1)
d1 = Dense(128, activation='relu', name='dense1')(flatten)
drop2 = Dropout(0.5, name='dropout2')(d1)
output = Dense(10, activation='softmax', name='output')(drop2)

# load mnist data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape((x_train.shape[0],) + original_img_size)
x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape((x_test.shape[0],) + original_img_size)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

with sess.as_default():
    cnn = Model(x, output)
    print(cnn.summary())
    cnn.compile(optimizer=Adam(lr=0.01), loss='categorical_crossentropy')

    # train
    cnn.fit(x_train, y_train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, y_test))

    print("Saving model")
    cnn.save("./../models/simple_cnn.h5")
