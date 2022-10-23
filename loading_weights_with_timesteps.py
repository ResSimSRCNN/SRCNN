import Created_Functions as cf
from only_RNN import Create_Model
import os
import numpy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

plot_labels = [
    "(Prediction) \n $1^{st}$ $S_o$ Layer",
    "(Truth) \n $1^{st}$ $S_o$ Layer",
]

training, testing = cf.matrix_org_with_timesteps(
    "./data_extraction_with_timesteps.mat", train_set=450)

model = Create_Model()
model.load_weights("./srcnn_final_RNN.hdf5")

modified_pic, truth_pic, input_pic = cf.get_model_and_truth_pic(
    model,
    testing,
    30
)

plot_maps_with_time = [
    modified_pic[0, :, :, :, :],
    truth_pic[0, :, :, :, :],
    '''input_pic[0, :, :, :, :]'''
]

cf.plotting_with_timesteps(pics=plot_maps_with_time,
                figure_num=1,
                names=plot_labels,
                color_map=cf.color_map()
                )
