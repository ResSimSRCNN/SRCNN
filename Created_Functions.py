import numpy
import matplotlib.pyplot as plt
import scipy.io as sio
import matplotlib as mplt
import tensorflow.keras.backend as K
from matplotlib.colors import ListedColormap


def color_map(x='jet'):
    scale = mplt.cm.get_cmap(x)
    cmp = scale(numpy.linspace(0, 1, 256))
    cmp = ListedColormap(cmp)
    return cmp


def MAE(model_pic, truth_pic, totalMAE):
    vic = [[0], [0]]
    for i in range(2):
        subtraction = numpy.subtract(
            truth_pic[:, :, i:i + 1],
            model_pic[:, :, i:i + 1]
        )
        absolute_subtraction = numpy.abs(subtraction)
        error_each_fig = absolute_subtraction.mean()
        totalMAE += error_each_fig

    return totalMAE


def MAE_old(model_pic, truth_pic, num, totalMAE):
    vic = [[0], [0]]
    for i in range(2):
        subtraction = numpy.subtract(
            truth_pic[0, :, :, i:i + 1],
            model_pic[0, :, :, i:i + 1]
        )
        absolute_subtraction = numpy.abs(subtraction)
        error_each_fig = absolute_subtraction.mean()
        totalMAE += error_each_fig

        print("Layer Number(" + str(i + 1) +
              ") FIGURE " + str(num + 1) + ": " +
              str(numpy.round(error_each_fig, decimals=4))
              )
        vic[i] = absolute_subtraction
    return totalMAE, vic


def all_error_calc(model_pic, truth_pic, total_errors):
    mae = MAE(model_pic, truth_pic, total_errors[0])
    acc5 = numpy.mean(accuracy_threshold_5_percent(truth_pic, model_pic)) + total_errors[1]
    acc10 = numpy.mean(accuracy_threshold_10_percent(truth_pic, model_pic)) + total_errors[2]
    error_psnr = numpy.mean(psnr(truth_pic, model_pic)) + total_errors[3]

    errors = [mae, acc5, acc10, error_psnr]
    return errors


def matrix_org(direction, train_set=250):
    mat_contents1 = sio.loadmat(direction)
    test_set = len(mat_contents1['K']) - train_set

    perm = numpy.array(mat_contents1['K'])
    truth_sat = numpy.array(mat_contents1['Sat'])
    Sat = numpy.array(mat_contents1['avg_saturation'])
    Sat = numpy.reshape(Sat, [len(mat_contents1['avg_saturation']), 50, 50, 1])

    train_perm = perm[0:train_set]
    test_perm = perm[train_set:test_set + train_set]

    train_sat = Sat[0:train_set]
    test_sat = Sat[train_set:test_set + train_set]

    train_truth = truth_sat[0:train_set]
    test_truth = truth_sat[train_set:test_set + train_set]

    training = [train_perm, train_sat, train_truth]
    testing = [test_perm, test_sat, test_truth]

    return training, testing


def matrix_org_with_timesteps(direction, train_set=250):
    mat_contents1 = sio.loadmat(direction)
    test_set = len(mat_contents1['K']) - train_set

    perm = numpy.array(mat_contents1['K'])
    truth_sat = numpy.array(mat_contents1['Sat'])
    Sat = numpy.array(mat_contents1['avg_saturation'])
    Sat = numpy.reshape(Sat, [len(mat_contents1['avg_saturation']), 11, 50, 50, 1])

    # for the final step only
    perm_time = numpy.zeros([500, 11, 100, 100, 2])
    for i in range(11):
        perm_time[:, i, :, :, :] = perm

    perm = perm_time

    train_perm = perm[0:train_set]
    test_perm = perm[train_set:test_set + train_set]

    train_sat = Sat[0:train_set]
    test_sat = Sat[train_set:test_set + train_set]

    train_truth = truth_sat[0:train_set]
    test_truth = truth_sat[train_set:test_set + train_set]

    training = [train_perm, train_sat, train_truth]
    testing = [test_perm, test_sat, test_truth]

    return training, testing


def matrix_org_with_timesteps_without_timesteps(
        direction,
        train_set=250,
        test_set=50,
        step_num=11
):
    mat_contents1 = sio.loadmat(direction)
    step_num = step_num - 1

    perm = numpy.array(mat_contents1['K'])
    truth_sat = numpy.array(mat_contents1['Sat'])
    Sat = numpy.array(mat_contents1['avg_saturation'])
    Sat = numpy.reshape(Sat, [len(mat_contents1['avg_saturation']), 11, 50, 50, 1])

    Sat = Sat[:, step_num, :, :, :]
    truth_sat = truth_sat[:, step_num, :, :, :]

    train_perm = perm[0:train_set]
    test_perm = perm[train_set:test_set + train_set]

    train_sat = Sat[0:train_set]
    test_sat = Sat[train_set:test_set + train_set]

    train_truth = truth_sat[0:train_set]
    test_truth = truth_sat[train_set:test_set + train_set]

    training = [train_perm, train_sat, train_truth]
    testing = [test_perm, test_sat, test_truth]

    return training, testing


def pic_model_format(pic):
    pic_shape = numpy.shape(pic)
    new_pic_shape = [1]
    new_pic_shape.extend(pic_shape)
    new_pic = numpy.reshape(pic, new_pic_shape)

    return new_pic


def plotting(pics, figure_num=None, names="none", color_map=mplt.cm.viridis):
    if figure_num is None:
        figure_num = 0
    plt.rcParams['figure.figsize'] = 35, 10.5
    plot_dim = [2, 4]
    fig, plot = plt.subplots(plot_dim[0], plot_dim[1])
    count = 0
    norm = mplt.colors.Normalize(vmin=0, vmax=1)
    axes = numpy.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    for x in range(plot_dim[0]):
        for y in range(plot_dim[1]):

            plot[x, y].imshow(
                pics[count],
                cmap=color_map,
                norm=norm
            )
            plot[x, y].set_title(names[count])
            plot[x, y].set_ylabel("Pixel Count")
            plot[x, y].set_xlabel("Pixel Count")

            if numpy.shape(pics[count]) != [100, 100, 1]:
                new_axes = axes[0:6]
            else:
                new_axes = axes

            plot[x, y].set_yticks(new_axes, minor=True)
            plot[x, y].set_xticks(new_axes, minor=True)
            cbar = fig.colorbar(
                mappable=mplt.cm.ScalarMappable(cmap=color_map),
                ax=plot[x, y],
                orientation="vertical")

            if (count == 0) or (count == 1):
                cbar.set_label('Permeability')
            else:
                cbar.set_label('Saturation')

            if count == ((plot_dim[0] * plot_dim[1]) - 2):
                count = 1
            else:
                count += 2

            # set the spacing between subplots
            plt.subplots_adjust(left=0.1,
                                bottom=0.1,
                                right=0.9,
                                top=0.9,
                                wspace=0.4,
                                hspace=0.4)

    plt.savefig('./Figures/training/figure number[' + str(figure_num + 1) + '].png')


def plotting_one_image(pic, num=0, name="none", color=color_map(), path="./data/"):
    plt.rcParams['figure.figsize'] = 4, 4

    fig, ax = plt.subplots()

    for item in [fig, ax]:
        item.patch.set_visible(False)

    im = ax.imshow(pic[:, :, 0], cmap=color, vmin=0, vmax=1)
    im.axes.get_xaxis().set_visible(False)
    im.axes.get_yaxis().set_visible(False)

    ax.set_axis_off()
    fig.add_axes(ax)

    fig.colorbar(mplt.cm.ScalarMappable(cmap=color),
                 ax=ax, orientation='vertical')

    path = path + name + "_[" + str(num) + '].png'
    plt.savefig(path, bbox_inches='tight')

    return path


def get_model_and_truth_pic(model, pic, set_num):
    perm_pic = pic[0][set_num]
    input_perm_pic = pic_model_format(perm_pic)

    sat_pic = pic[1][set_num]
    input_sat_pic = pic_model_format(sat_pic)

    input_pic = [input_perm_pic, input_sat_pic]

    modified_pic = model(input_pic)
    truth_pic = pic[2][set_num:set_num + 1]

    return modified_pic, truth_pic, input_pic


def plot_error(model_fitting, num="", path="./data/"):
    plt.figure(figsize=(6, 4))
    plt.plot(model_fitting.history['loss'])
    plt.plot(model_fitting.history['val_loss'], '--')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    path1 = path + 'loss_[' + str(num) + '].png'
    plt.savefig(path1)

    plt.figure(figsize=(6, 4))
    plt.semilogy(model_fitting.history['loss'])
    plt.semilogy(model_fitting.history['val_loss'], '--')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    path2 = path + 'loss_semilogy_[' + str(num) + '].png'
    plt.savefig(path2)

    return path1, path2


def psnr(y_true, y_pred):
    MSE = K.mean(K.flatten(y_true - y_pred) ** 2)
    pixel_max = K.max(K.flatten(y_true))
    eq = 20 * K.log(pixel_max / (MSE ** 0.5))
    return eq


def accuracy_threshold_5_percent(y_true, y_pred):
    threshold = 0.05
    error = K.flatten(K.abs(y_true - y_pred)) <= threshold
    return error


def accuracy_threshold_10_percent(y_true, y_pred):
    threshold = 0.1
    error = K.flatten(K.abs(y_true - y_pred)) <= threshold
    return error


def path_to_html(path):
    return '''<img src="''' + path + '''">'''


def rearrange_parameters(array, parameter):
    arranged = numpy.reshape(array, [-1, len(parameter)]).tolist()
    output = []
    for sublist in arranged:
        output.append(list(map(int, sublist)))

    return output


def plotting_with_timesteps(pics, figure_num, names="none", color_map=mplt.cm.viridis):
    plt.rcParams['figure.figsize'] = 20, 20
    rows = 4
    plot_dim = [rows, 4]
    fig, plot = plt.subplots(plot_dim[0], plot_dim[1])
    norm = mplt.colors.Normalize(vmin=0, vmax=1)
    axes = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for y in range(plot_dim[0]):
        for x in range(plot_dim[1]):
            if y < int(rows/2):
                plot[y, x].imshow(
                    pics[y][int(x*3+1), :, :, 0:1],
                    cmap=color_map,
                    norm=norm,
                    extent=[0, 100, 100, 0])
            else:
                plot[y, x].imshow(
                    pics[y-int(rows/2)][int(x*3+1), :, :, 1:2],
                    cmap=color_map,
                    norm=norm,
                    extent=[0, 100, 100, 0])

            plot[y, x].set_title(" t= " + str(x*3+1))
            plot[y, x].set_ylabel("Pixel Count")
            plot[y, x].set_xlabel("Pixel Count")
            plot[y, x].set_yticks(axes, minor=True)
            plot[y, x].set_xticks(axes, minor=True)

            cbar = fig.colorbar(
                mappable=mplt.cm.ScalarMappable(cmap=color_map),
                ax=plot[y, x],
                orientation="vertical")
            cbar.set_label('Saturation')

            # set the spacing between subplots
            plt.subplots_adjust(left=0.1,
                                bottom=0.1,
                                right=0.9,
                                top=0.9,
                                wspace=0.4,
                                hspace=0.4)

    plt.savefig('./RNN figure number[' + str(figure_num + 1) + '].png')
