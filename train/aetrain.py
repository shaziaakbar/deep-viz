'''
File: 	    aetrain.py
Author: 	Shazia Akbar

Description:
Dummy code for training a very simple autoencoder.
'''

import tensorflow as tf
sess = tf.Session()

import keras.backend as K
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Conv2D, Lambda, Dense, Dropout, Reshape, Flatten
from keras.optimizers import Adam
from keras.models import load_model

import numpy as np
import h5py

img_rows, img_cols, img_chns = 28, 28, 1
batch_size = 100
if K.image_data_format() == 'channels_first':
    original_img_size = (img_chns, img_rows, img_cols)
else:
    original_img_size = (img_rows, img_cols, img_chns)

latent_dim = 2
intermediate_dim = 6
epochs = 10
epsilon_std = 0.001

x = Input(shape=original_img_size)
x_flatten = Flatten()(x)
init_drop = Dropout(0.2)(x_flatten)
h = Dense(intermediate_dim, activation='relu')(init_drop)
z_mean = Dense(latent_dim, name="encoding")(h)
z_log_var = Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=K.shape(z_mean), mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(img_rows * img_cols * img_chns, activation='linear')

if K.image_data_format() == 'channels_first':
    output_shape = (batch_size, img_chns, img_rows, img_cols)
else:
    output_shape = (batch_size, img_rows, img_cols, img_chns)
decoder_reshape = Reshape(original_img_size)
h_decoded = decoder_h(z)
x_decoded_mean_squashed = decoder_mean(h_decoded)
x_decoded_mean = decoder_reshape(x_decoded_mean_squashed)

from keras.objectives import binary_crossentropy
def vae_loss(x, x_decoded_mean):
    xent_loss = binary_crossentropy(K.flatten(x), K.flatten(x_decoded_mean))
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss

if False:
    with sess.as_default():
        vae = Model(x, x_decoded_mean)
        print(vae.summary())
        vae.compile(optimizer=Adam(lr=0.01), loss=vae_loss)

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype('float32') / 255.
        x_train = x_train.reshape((x_train.shape[0],) + original_img_size)
        x_test = x_test.astype('float32') / 255.
        x_test = x_test.reshape((x_test.shape[0],) + original_img_size)

        vae.fit(x_train, x_train,
                shuffle=True,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, x_test))

        print("Saving trained encoder")
        vae.save("./../models/simple_vae.h5")

def load_vae_model(model_name):
    model = load_model(model_name, custom_objects={'latent_dim': latent_dim, 'epsilon_std': epsilon_std, 'vae_loss': vae_loss})
    return model
