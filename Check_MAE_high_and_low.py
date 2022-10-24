import Created_Functions as cf
from model_SRCNN import Create_Model
import os
import numpy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

training, testing = cf.matrix_org_with_timesteps_without_timesteps(
    "./data_extraction_with_timesteps.mat", train_set=450, step_num=11)

filters_values = [64, 64, 16]
kernel = [7, 5, 3]
activation = ['linear', 'relu', 'tanh']

model = Create_Model(filters=filters_values, kernel_size=kernel, activations=activation)
model.load_weights("./model_finale.hdf5")

total_mean_absolute_error = 0
max_error_value = 0
test_num = 0
min_error_value = 50000
for num in range(len(testing[0])):
    modified_pic, truth_pic, input_pic = cf.get_model_and_truth_pic(
        model,
        testing,
        num
    )
    print("calculating error for test num:" + str(num))
    total_mean_absolute_error, error_matrix = cf.MAE_old(
        modified_pic,
        truth_pic,
        num,
        0
    )
    error_value = total_mean_absolute_error

    if max_error_value < error_value:
        max_error_value = error_value
        test_num_max = num

    if min_error_value > error_value:
        min_error_value = error_value
        test_num_min = num

print("\n\n Worst testing data is testing data number: " + str(test_num_max) + "\n highest error is: " + str(
    max_error_value / 2))
print("\n\n best testing data is testing data number: " + str(test_num_min) + "\n lowest error is: " + str(
    min_error_value / 2))