'''
File: 	aeviz.py
Author: 	Shazia Akbar

Description:
Methods for visualizing the encoder layer in a trained autoencoder. Note: The encoding layer must have the name "encoder" attached to it.
'''

import keras.backend as K
import utils
import numpy as np
import aetrain

def load_encoding_layer(model_filename):
    """Loads the autoencoder layer in a saved model and create a function for querying that layer. Note that this layer must have the name "encoding".
        
        Args:
            model_filename (string): full path to saved autoencoder model
        Returns:
            1) Keras layer corresponding to the encoding layer
            2) shape of input to the model
    """
    model = aetrain.load_vae_model(model_filename)
    encoding_layer = K.function([model.layers[0].input, K.learning_phase()], [model.get_layer(name="encoding").output])
    return encoding_layer, model.layers[0].input_shape[1:]


#
from skimage import draw
def get_sample_texture(input_size):
    """Function which retrieves dummy texture
        
        Args:
            input_size ((int, int, int)): Specify width, height and channels for sample texture
        Returns:
            ndarray binary image containing two circles
    """
    w, h, c = input_size
    arr = np.zeros((w, h, c))
    rr, cc = draw.circle_perimeter(20, 20, radius=8, shape=arr.shape)
    arr[rr, cc, :] = 1
    rr, cc = draw.circle_perimeter(100, 50, radius=20, shape=arr.shape)
    arr[rr, cc, :] = 1
    return arr


# Take as input a pretrained keras_model and (optionally) a patch that is to be visualized
def vizualize_encoder(keras_model, viz_input=None):
    """Function for visualizing the encoding layer in a pretrained autoencoder (see aetrain.py)
        
        Args:
            keras_model: full path to autoencoder to be visualized
            viz_input: image to be used for the visualization (optional)
    """
    f, input_size = load_encoding_layer(keras_model)
    
    # Get sample texture if none provided
    if viz_input is None:
        viz_input = get_sample_texture((128, 128, 1))

    encodings = np.zeros((2, viz_input.shape[0], viz_input.shape[1]))
    # Specify amount of padding needed to visualize edges of image as well
    pad_values = (int(input_size[0]/2), int(input_size[1]/2))
    viz_input = np.pad(viz_input, ((pad_values[0], pad_values[0]), (pad_values[1], pad_values[1]), (0,0)), mode='constant')

    # For each patch in image, get encoding
    num_x_tiles, num_tiles = 0, 0
    patch_size_x, patch_size_y = input_size[0], input_size[1]
    for x1 in range(0, viz_input.shape[0] - patch_size_x, 1):
        i = 0
        for y1 in range(0, viz_input.shape[1] - patch_size_y, 1):
            encodings[:, num_x_tiles, i] = f((viz_input[x1:x1+patch_size_x, y1:y1+patch_size_y][np.newaxis,...], 0))[0]
            i = i + 1
        num_x_tiles = num_x_tiles + 1
        num_tiles = num_tiles + i

    encodings = np.concatenate((viz_input[pad_values[0]: -pad_values[0], pad_values[1]:-pad_values[1]].transpose(2,0,1), encodings))
    utils.print_images(encodings, columns=3)


