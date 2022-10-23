import Created_Functions_ali as cf
from model import Create_Model
import os
import numpy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

plot_labels = [
    r"$1^{st}$ Permeability Layer", r"$2^{nd}$ Permeability Layer",
    "Average $S_o$ Input", "Average $S_o$ input \n",
    "(Prediction) \n $1^{st}$ $S_o$ Layer", "(Prediction) \n $2^{nd}$ $S_o$ Layer",
    "(Truth) \n $1^{st}$ $S_o$ Layer", "(Truth) \n $2^{nd}$ $S_o$ Layer"
]

training, testing = cf.matrix_org_with_timesteps_without_timesteps(
    "./data_extraction_with_timesteps.mat", train_set=450, step_num=11)


filters_values = [64, 64, 16]
# k is for kernel size
k = [7, 5, 3]
# activation
activations = ['linear', 'relu', 'tanh']

model = Create_Model(filters=filters_values, kernel_size=k, activations=activations)

model.load_weights("./model_finale.hdf5")

pic_num = 336

total_mean_absolute_error = 0
for num in range(1):
    modified_pic, truth_pic, input_pic = cf.get_model_and_truth_pic(
        model,
        training,
        pic_num
    )

    plot_maps = [
        input_pic[0][0, :, :, 0:1], input_pic[0][0, :, :, 1:2],
        input_pic[1][0, :, :, 0:1], input_pic[1][0, :, :, 0:1],
        modified_pic[0, :, :, 0:1], modified_pic[0, :, :, 1:2],
        truth_pic[0, :, :, 0:1], truth_pic[0, :, :, 1:2]
    ]

    cf.plotting(pics=plot_maps,
                figure_num=num,
                names=plot_labels,
                color_map=cf.color_map()
                )

    total_mean_absolute_error, error_matrix = cf.MAE_old(
        modified_pic,
        truth_pic,
        num,
        total_mean_absolute_error
    )

    print("Mean Absolute Error is: " + str(total_mean_absolute_error))
