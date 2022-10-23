import tensorflow as tf
from tensorflow.keras.layers import *


def Create_Model(filters=None, kernel_size=None, activations=None, num=None):
    if kernel_size is None:
        kernel_size = [7, 5, 5]
    if filters is None:
        filters = [64, 64, 32]
    if activations is None:
        activations = ['linear', 'relu', 'relu']
    if num is None:
        num = 1

    # inputting the permeability and saturation
    perm_input = Input(shape=(100, 100, 2), name="permeability")
    sat_input = Input(shape=(50, 50, 1), name="Average_saturation")

    # increasing the number of pixels
    sat = Conv2DTranspose(
        filters=1,
        kernel_size=2,
        strides=2,
        padding="valid")(sat_input)

    '''
    # increasing the number of pixels
    sat = UpSampling2D(size=(2, 2), name="Bicubic_interpolation",
                       interpolation="bicubic")(sat_input)'''

    # merging the permeability with the saturation
    merged = concatenate([perm_input, sat], name="Saturation_and_Permeability")

    # modifying range
    merged = Conv2D(filters=filters[0], kernel_size=kernel_size[0], padding="same",
                    activation=activations[0])(merged)

    merged = Conv2D(filters=filters[1], kernel_size=kernel_size[1], padding="same",
                    activation=activations[1])(merged)

    merged = Conv2D(filters=filters[2], kernel_size=kernel_size[2], padding="same",
                    activation=activations[2])(merged)


    '''
    # averaging pixels
    merged = Conv2D(filters=filters[3], kernel_size=kernel_size[3], padding="same")(merged)

    
    # averaging pixels
    merged = Conv2D(filters=filters[2], kernel_size=kernel_size[1], padding="same")(merged)
    
    '''

    # output layer
    output_layer = Conv2D(filters=2, kernel_size=[1, 1], name='Output_layer', padding='same')(merged)

    # creating the model
    model = tf.keras.Model([perm_input, sat_input],
                           output_layer,
                           name="SRCNN_for_reservoir_simulation")

    return model
