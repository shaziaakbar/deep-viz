'''
File: 	aeviz.py
Author: 	Shazia Akbar

Description:
Methods for visualizing the encoder layer in a trained autoencoder. Note: The encoding layer must have the name "encoder" attached to it.
'''

import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import math
import aetrain

def load_encoding_layer(model_filename):
    model = aetrain.load_vae_model(model_filename)
    encoding_layer = K.function([model.layers[0].input, K.learning_phase()], [model.get_layer(name="encoding").output])
    return encoding_layer, model.layers[0].input_shape[1:]


# Function which retrieves dummy textures if none provided
from skimage import draw
def get_sample_texture(input_size):
    w, h, c = input_size
    arr = np.zeros((w, h, c))
    rr, cc = draw.circle_perimeter(20, 20, radius=8, shape=arr.shape)
    arr[rr, cc, :] = 1
    rr, cc = draw.circle_perimeter(100, 50, radius=20, shape=arr.shape)
    arr[rr, cc, :] = 1
    return arr

# For displaying results...
def setup_figure(rows, columns, figsize):
    figs, axes = plt.subplots(rows, columns, figsize=figsize, sharex=True, sharey=True)
    if(columns > 1):
        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')
    else:
        axes.set_xticks([])
        axes.set_yticks([])
        axes.axis('off')
    return axes

def print_images(image_list, columns=5, figsize=(5,5)):
    rows = int(math.ceil(float(len(image_list)) / float(columns)))
    axes = setup_figure(rows,columns,figsize)
    index = 0
    for patch in image_list:
        if(rows == 1):
            ax = axes
        else:
            ax = axes[int(math.floor(float(index / columns)))]
    
        if(len(patch.shape) == 2):
            plt.rcParams['image.cmap'] = 'gray'

        if isinstance(ax, (list, tuple, np.ndarray)):
            ax[index % columns].imshow(patch)
        else:
            ax.imshow(patch)
                
        index = index + 1
    plt.show()


# Take as input a pretrained keras_model and (optionally) a patch that is to be visualized
def vizualize_encoder(keras_model, viz_input=None):
    f, input_size = load_encoding_layer(keras_model)
    if viz_input is None:
        viz_input = get_sample_texture((128, 128, 1))

    encodings = np.zeros((2, viz_input.shape[0], viz_input.shape[1]))
    pad_values = (int(input_size[0]/2), int(input_size[1]/2))
    viz_input = np.pad(viz_input, ((pad_values[0], pad_values[0]), (pad_values[1], pad_values[1]), (0,0)), mode='constant')

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
    print_images(encodings, columns=3)


