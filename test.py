'''
File: 	test.py
Author: 	Shazia Akbar

Description:
Short and simple example of visualization tool
'''

import viz
from keras.datasets import mnist


def test_encoder_viz():
    viz.vizualize_encoder('./models/simple_vae.h5', sample_input)


def test_cnn_viz():
    _, sample_input = mnist.load_data()
    sample_input = sample_input[0][1].reshape(28, 28, 1) / 255.
    
    viz.vizualize_cnn('./models/simple_cnn.h5', 'conv1', sample_input)
