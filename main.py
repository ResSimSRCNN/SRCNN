from model import Create_Model
import Created_Functions_ali as cf
from time import time
import os
import pandas as pd
import numpy as np
from itertools import product
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import gc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

training, testing = cf.matrix_org_with_timesteps_without_timesteps(
    "./data_extraction_with_timesteps.mat", train_set=450, step_num=11)

error_val, accuracy_val, accuracy_val2, psnr_val = [], [], [], []
all_outputs, all_truths = [], []
error_all, error_semi_log_all = [], []
k_all, f_all, t, all_activations, all_layers = [], [], [], [], []
pic_num = 29
counting = 0
activations_types = ['linear', 'relu', 'tanh', 'gelu']
kernel_types = [3, 5, 7, 9, 11]
filter_types = [8, 16, 32, 64, 128, 256]

filter_combinations = list(product(filter_types, repeat=3))
count = 0

for ind, f_size in enumerate(filter_combinations[:]):  # leave the [:] it is critical IDK the reason
    if (f_size[0] < f_size[1]) or (f_size[1] < f_size[2]):
        del filter_combinations[ind - count]
        count += 1
# length of filter combinations is 56
activations_combinations = list(product(activations_types, repeat=3))
kernel_combinations = list(product(kernel_types, repeat=3))

'''base case:
        filter = [8,8,8]
        kernel = [3,3,3]
        activation = [linear,linear,linear]
        
        '''

ite_num = 0
ite = 5 * ite_num

path = "C:/Users/fbarg/PycharmProjects/pythonProject2/html_data/good_model/" + str(ite_num)
# os.mkdir(path)
path = path + "/"
truth_path = cf.plotting_one_image(testing[2][pic_num, :, :, 0:1], name="TRUTH", path=path)
cf.plotting_one_image(testing[2][pic_num, :, :, 1:2], name="TRUTH_L2", path=path)
cf.plotting_one_image(testing[0][pic_num, :, :, 0:1], name="PERM_L1", path=path)
cf.plotting_one_image(testing[0][pic_num, :, :, 1:2], name="PERM_L2", path=path)
cf.plotting_one_image(testing[1][pic_num, :, :, 0:1], name="SAT", path=path)
truth_html = cf.path_to_html(truth_path)

# for i in range(len(kernel_combinations[ite - 5:ite])):
for ind in range(1):
    i = ind
    print("iteration number " + str(i + 1))
    counting += 1
    layers_num = 3 + i
    # f is for filter_values
    filters_values = [64, 64, 16]

    # k is for kernel size
    k = [7, 5, 3]

    # activation
    activations = ['linear', 'relu', 'tanh']

    tf.keras.backend.clear_session()
    # change the multiplication if you need for filters
    model = Create_Model(filters=filters_values, kernel_size=k, activations=activations, num=i + 1)

    tf.keras.utils.plot_model(
        model, "model_with_shape_info.png",
        show_shapes=True,
        expand_nested=True,
        show_layer_activations=True)

    path_checkpoint = path + "model_finale" + str(ind + 1) + ".hdf5"

    save_checkpoint = ModelCheckpoint(
        path_checkpoint,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
    )
    print()

    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=["mean_absolute_error", cf.psnr, cf.accuracy_threshold_5_percent, cf.accuracy_threshold_10_percent]
    )

    # this is only for getting the time correctly for the first time (due to needing more time for the first time)
    if i == 0:
        model.fit(training[0:2], training[2], epochs=1)

    start = time()

    history = model.fit(
        training[0:2], training[2],
        validation_data=[testing[0:2], testing[2]],
        epochs=40,
        callbacks=save_checkpoint
    )

    end = time()

    print("time elapsed")
    count = end - start
    count = count.__round__(2)
    print("    " + str(count) + " Seconds \n\n")

    t = np.append(t, count)
    error_val = np.append(error_val, history.history['mean_absolute_error'][-1])
    accuracy_val = np.append(accuracy_val, history.history['accuracy_threshold_5_percent'][-1])
    accuracy_val2 = np.append(accuracy_val2, history.history['accuracy_threshold_10_percent'][-1])
    psnr_val = np.append(psnr_val, history.history['psnr'][-1])

    modified_pic, truth_pic, input_pic = cf.get_model_and_truth_pic(
        model,
        testing,
        pic_num
    )

    output_path = cf.plotting_one_image(modified_pic[0, :, :, 0:1], i, name="output", path=path)
    cf.plotting_one_image(modified_pic[0, :, :, 1:2], i, name="output_L2", path=path)
    output_html = cf.path_to_html(output_path)
    all_outputs = np.append(all_outputs, output_html)

    error, error_semi_log = cf.plot_error(history, i, path=path)

    error_semi_log_html = cf.path_to_html(error_semi_log)
    error_semi_log_all = np.append(error_semi_log_all, error_semi_log_html)

    error_html = cf.path_to_html(error)
    error_all = np.append(error_all, error_html)

    k_all = np.append(k_all, k)
    f_all = np.append(f_all, filters_values)
    all_activations = np.append(all_activations, activations)
    all_layers = np.append(all_layers, layers_num)

    f_all_edited = cf.rearrange_parameters(f_all, filters_values)
    k_all_edited = cf.rearrange_parameters(k_all, k)
    all_activations_edited = np.reshape(all_activations, [-1, np.shape(activations)[0]])

    accuracy_val_edited = ['{:.1%}'.format(x.round(4)) for x in accuracy_val]
    accuracy_val2_edited = ['{:.1%}'.format(x.round(4)) for x in accuracy_val2]
    psnr_val_edited = ['{:.2f}'.format(x.round(2)) for x in psnr_val]
    error_val_edited = ['{:.4f}'.format(x.round(4)) for x in error_val]

    all_truths_edited = [np.append(all_truths, truth_html) for x in range(counting)]

    df1 = pd.DataFrame(
        [t, all_activations_edited, all_layers, f_all_edited, k_all_edited, psnr_val_edited, error_val_edited,
         accuracy_val_edited, accuracy_val2_edited,
         all_outputs, all_truths_edited, error_all, error_semi_log_all],
        index=["time in seconds", "activation functions", "number of layers", "filter number", "kernel size",
               "PSNR",
               "  Mean Absolute Error  ",
               "accuracy threshold 5%", "accuracy threshold 10%", "output", "ground truth", "error",
               "error \n semi log"])

    with open(path + 'data_comparison.html', 'w') as fo:
        fo.write(df1.T.to_html(
            render_links=True, escape=False, justify='center', col_space=90
        ).replace('<tr>', '<tr style="text-align: center;">')
                 )

    df2 = pd.DataFrame(
        [t, f_all_edited, k_all_edited, all_activations_edited, all_layers, psnr_val_edited, error_val_edited,
         accuracy_val_edited, accuracy_val2_edited],
        index=["time in seconds", "filter number", "kernel size", "activation functions", "number of layers",
               "PSNR",
               "  Mean Absolute Error  ", "accuracy threshold 5%", "accuracy threshold 10%"])

    with open(path + 'value_comparison.html', 'w') as fo:
        html = df2.T.to_html(
            render_links=True, escape=False, justify='center', col_space=90
        )
        html = html.replace('<tr>', '<tr style="text-align: center;">')

        fo.write(html)

    del model, accuracy_val_edited, accuracy_val2_edited, psnr_val_edited, error_val_edited, all_activations_edited, k_all_edited, f_all_edited
    gc.collect()

plot_maps = [
    input_pic[0][0, :, :, 0:1], input_pic[0][0, :, :, 1:2],
    input_pic[1][0, :, :, 0:1], input_pic[1][0, :, :, 0:1],
    modified_pic[0, :, :, 0:1], modified_pic[0, :, :, 1:2],
    truth_pic[0, :, :, 0:1], truth_pic[0, :, :, 1:2]
]
plot_labels = [
    r"$1^{st}$ Permeability Layer", r"$2^{nd}$ Permeability Layer",
    "Average $S_o$ Input", "Average $S_o$ input",
    "(Prediction) \n $1^{st}$ $S_o$ Layer", "(Prediction) \n $2^{nd}$ $S_o$ Layer",
    "(Truth) \n $1^{st}$ $S_o$ Layer", "(Truth) \n $2^{nd}$ $S_o$ Layer"
]

cf.plotting(pics=plot_maps,
                names=plot_labels,
                color_map=cf.color_map()
                )
