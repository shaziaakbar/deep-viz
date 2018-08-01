'''
File: 	utils.py
Author: 	Shazia Akbar

Description:
Methods for setting up environment and displaying images. Used to visualize various types of models
'''

import matplotlib.pyplot as plt
import numpy as np
import math

def setup_figure(rows, columns, figsize):
    """Sets up a matplotlib figure window with specified rows and columns. These can be padded with images later (see aeviz.py for an example)
    """
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
    """Function for displaying multiple images in one figure
    
    Args:
        image_list (ndarray): shape[0] number of images stacked together
        columns (int): how many columns in figure window user would like [optional]
        figsize ((int, int)): size of matplotlib figure window [optional]
    Returns:
        (None) 
    
    """
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

