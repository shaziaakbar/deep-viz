'''
File: 	aeviz.py
Author: 	Shazia Akbar

Description:
Methods for visualizing the encoder layer in a trained autoencoder. Note: The encoding layer must have the name "encoder" attached to it.
'''

import keras.backend as K
from keras.models import load_model
import utils
import math
import numpy as np

import sys
sys.path.insert(0, "./train")
import aetrain

def load_layer(model, layer_name="encoding"):
    """Loads the layer in a saved model and create a function for querying that layer.
        
        Args:
            model (Keras model) in which the layer resides
            layer_name (string): specifies layer to be retrieves. default is the "encoding" layer
            
        Returns:
            Keras layer corresponding to the encoding layer
    """
    layer = K.function([model.layers[0].input, K.learning_phase()], [model.get_layer(name=layer_name).output])
    return layer


def visualize_cnn_layer(model, layer_function, test_image):
    """Run image through CNN and displays output
        
        Args:
        model: full path to model to be visualized
        layer_function (string): Keras function which processes input data
        test_image: ONE test image to be visualized - all features for the same
            image will be displayed
        """
    input_size = model.layers[0].input_shape[1:]
    
    # Get sample texture if none provided
    if test_image is None:
        test_image = utils.get_sample_texture((128, 128, 1))

    results = np.squeeze(layer_function((test_image[np.newaxis,...], 0)))
    x_pad = int(math.ceil((test_image.shape[0] - results.shape[0]) / 2.))
    y_pad = int(math.floor((test_image.shape[1] - results.shape[1]) / 2.))
    results = np.pad(results, ((x_pad, x_pad), (y_pad, y_pad), (0, 0)), mode='constant')

    results = np.concatenate((test_image, results), axis=2).transpose(2, 0, 1)
    utils.print_images(results, columns=10)


def visualize_layer_generic(model, layer_function, test_image):
    input_size = model.layers[0].input_shape[1:]
    
    # Get sample texture if none provided
    if test_image is None:
        test_image = utils.get_sample_texture((128, 128, 1))

    encodings = np.zeros((2, test_image.shape[0], test_image.shape[1]))
    # Specify amount of padding needed to visualize edges of image as well
    pad_values = (int(input_size[0]/2), int(input_size[1]/2))
    test_image = np.pad(test_image, ((pad_values[0], pad_values[0]), (pad_values[1], pad_values[1]), (0,0)), mode='constant')

    # For each patch in image, get encoding
    num_x_tiles, num_tiles = 0, 0
    patch_size_x, patch_size_y = input_size[0], input_size[1]
    for x1 in range(0, test_image.shape[0] - patch_size_x, 1):
        i = 0
        for y1 in range(0, test_image.shape[1] - patch_size_y, 1):
            encodings[:, num_x_tiles, i] = np.max(layer_function((test_image[x1:x1+patch_size_x, y1:y1+patch_size_y][np.newaxis,...], 0))[0])
            i = i + 1
        num_x_tiles = num_x_tiles + 1
        num_tiles = num_tiles + i

    encodings = np.concatenate((test_image[pad_values[0]: -pad_values[0], pad_values[1]:-pad_values[1]].transpose(2,0,1), encodings))
    utils.print_images(encodings, columns=3)


def vizualize_encoder(keras_model, viz_input=None):
    """Function for visualizing the encoding layer in an autoencoder (see aetrain.py)
        
        Args:
            keras_model: full path to autoencoder to be visualized
            viz_input: image to be used for the visualization (optional)
    """
    model = aetrain.load_vae_model(keras_model)
    f = load_layer(model)

    visualize_layer_generic(model, f, viz_input)


def vizualize_cnn(keras_model, layer_name, viz_input=None):
    """Function for visualizing a layer in a deep network
        
        Args:
            keras_model: full path to model to be visualized
            layer_name (string): name assigned to the layer that you wish to visualize
            viz_input: image to be used for the visualization (optional)
    """
    model = load_model(keras_model)
    f = load_layer(model, layer_name)

    visualize_cnn_layer(model, f, viz_input)


