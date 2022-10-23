import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *


def Create_Model(filters=None, kernel_size=None, activations=None, num=None, num_of_steps=None):
    if kernel_size is None:
        kernel_size = [7, 5, 5]
    if filters is None:
        filters = [64, 32, 32]
    if activations is None:
        activations = ['linear', 'relu', 'relu']
    if num is None:
        num = 1
    if num_of_steps is None:
        num_of_steps = 11

    # inputting the permeability and saturation
    perm_input = Input(shape=(num_of_steps, 100, 100, 2), name="permeability")

    sat_input = Input(shape=(num_of_steps, 50, 50, 1), name="Average_saturation")


    sat = Conv3DTranspose(
        filters=1,
        kernel_size=(1, 2, 2),
        strides=(1, 2, 2),
        padding="valid")(sat_input)



    # merging the permeability with the saturation
    merged = concatenate([perm_input, sat])

    merged = ConvLSTM2D(filters=filters[0], kernel_size=kernel_size[0],
                        padding='same', return_sequences=True, activation=activations[0])(merged)

    merged = ConvLSTM2D(filters=filters[1], kernel_size=kernel_size[1],
                        padding='same', return_sequences=True, activation=activations[1])(merged)

    merged = ConvLSTM2D(filters=filters[2], kernel_size=kernel_size[2],
                        padding='same', return_sequences=True, activation=activations[2])(merged)

    output_layer = ConvLSTM2D(filters=2, kernel_size=1, padding='same', return_sequences=True, activation='relu')(
        merged)

    # creating the model
    model = tf.keras.Model([perm_input, sat_input],
                           output_layer,
                           name="SRCNN_for_reservoir_simulation")

    return model
