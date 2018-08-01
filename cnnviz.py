'''
File: 	aeviz.py
Author: 	Shazia Akbar

Description:
Methods for visualizing the encoder layer in a trained autoencoder. Note: The encoding layer must have the name "encoder" attached to it.
'''

import keras.backend as K
import utils

def load_layer(model_filename, layer_index):
    """Loads a single Keras layer from a saved CNN model
    
    Parameters:
        model_filename: location to CNN model
        layer_index: an integer indicating which layer you would like to retrieve
    Returns:
        Function for querying specified layer
    """
    model = K.load_layer(model_filename, )
    layer_func = K.function([model.layers[0].input, K.learning_phase()], [model.get_layer(layer_index).output])
    return layer_func



