'''
    Once we have a trained neural network in tf lite,
    test that quantization and scaling work

'''

import pandas
import sys,os
from collections import OrderedDict
import argparse
import numpy as np
import datetime

# where the base code is on your machine
SMART_TOOLS_ROOT_DIR = os.environ['SMART_TOOLS_ROOT_DIR']
SCRATCH_DIR = SMART_TOOLS_ROOT_DIR + '/scratch/'

# generic plotting utils - always use and modify these
UTILS_DIR=os.environ['UTILS_DIR']
sys.path.append(UTILS_DIR)
from plotting_utils import *
from textfile_utils import *

import tensorflow as tf
from utils_tensorflow import *

# data is re-organized to size:
# BATCH_SIZE x NUM_SENSORS x NUM_FEATURES
# view this as an image with 1 channel, NUM_SENSORS x NUM_FEATURES size

def load_trained_tflite_model(model_path):

    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    ################################################
    # Test the model on random input data.
    input_shape = input_details[0]['shape']

    return interpreter, input_details, output_details, input_shape


if __name__ == '__main__':

    # now build the CNN model
    base_dir = SMART_TOOLS_ROOT_DIR + '/pretrained_models/tensorflow_classifier/'
    model_base_dir = base_dir + '/tf_model/'

    # Pull arguments from command line.
    parser = argparse.ArgumentParser(description='plot sensor data')

    # where to read data
    parser.add_argument('--train_csv', type=str, default= SMART_TOOLS_ROOT_DIR + '/processed_data/Feature_Processed_Data/OL50_10secframe_Proccessed_Train_Xy_Matrix.csv')

    parser.add_argument('--val_csv', type=str, default= SMART_TOOLS_ROOT_DIR + '/processed_data/Feature_Processed_Data/OL50_10secframe_Proccessed_Validate_Xy_Matrix.csv')

    parser.add_argument('--test_csv', type=str, default=SMART_TOOLS_ROOT_DIR + '/processed_data/Feature_Processed_Data/OL50_10secframe_Proccessed_Test_Xy_Matrix.csv')

    args = parser.parse_args()

    # STEP 0: PANDAS DATA MANIPULATION, SAME AS RANDOM FOREST
    ########################################################################
    # Load into dataframes and do some basic stats
    train_df = pandas.read_csv(args.train_csv)
    val_df = pandas.read_csv(args.val_csv)
    test_df = pandas.read_csv(args.test_csv)

    num_activities = len(set(train_df['Activity']))

    # get a numpy array of X and Y data
    ########################################################################

    # get all columns that are inputs to our model
    x_features_columns = [colname for colname in list(train_df) if colname not in ['Unnamed: 0', 'Activity', 'Subject Number', 'Trial']]

    # try different y columns
    # first, how well can we predict the activity?
    # then, can we predict the subject
    # then, we can predict the trial number
    # the first should be high, second and third should be low

    y_features_columns = 'Activity'

    # repeat key column extraction for train and val data

    train_test_val_df_dict = OrderedDict()
    train_test_val_df_dict['train'] = train_df
    train_test_val_df_dict['val'] = val_df
    train_test_val_df_dict['test'] = test_df

    tf_dataset_dict = OrderedDict()

    # min, max, mean etc.
    num_features = 10

    num_sensors = int(len(x_features_columns)/num_features)

    # SCALE THE DATA and create a Tensorflow DataSet
    # analagous to a Pytorch data loader
    # KEY POINT: we manually re-scale without sklearn so we know how to replicate the scaling in C on the Arduino
    # we have already stored the scaling quantiles in train_quantile_csv

    # retrieve the quantiles from this file for normalization, this was computed during training
    quantile_list = [.001, 0.25, 0.5, 0.75, 0.999]
    train_quantile_csv = base_dir + '/train_normalization_quantiles.csv'
    train_quantile_df = pandas.read_csv(train_quantile_csv)

    train_quantile_df = train_quantile_df.set_index(train_quantile_df.columns[0])

    for data_split, data_df in train_test_val_df_dict.items():

        # load raw data from csvs
        data_x_np, data_y_np, data_x_df, data_y_df = get_xy_numpy(data_df, x_features_columns, y_features_columns=y_features_columns)

        # for all data, scale each column using the same PER-COLUMN scaling as the training data for uniformity
        normalized_data_x_df = data_x_df.copy()
        for feature_name in data_x_df.columns:

            # do not use absolute min, max due to OUTLIERS!
            min_value = train_quantile_df[feature_name][quantile_list[0]]
            max_value = train_quantile_df[feature_name][quantile_list[-1]]

            normalized_data_x_df[feature_name] = (data_x_df[feature_name] - min_value) / (max_value - min_value)

        # now, print the stats of the normalized dataframe, the max should be roughly near 1 always
        print(' ')
        print(' ')
        print('data split: ', data_split)
        print(normalized_data_x_df.describe())
        print(' ')
        print(' ')

        ## now actually transform the training data
        data_x_np_scaled = normalized_data_x_df.to_numpy()

        ## BATCH_SIZE x NUM_SENSORS x NUM_FEATURES
        ## view this as an image with 1 channel, NUM_SENSORS x NUM_FEATURES size

        reshaped_data_x_np_scaled = data_x_np_scaled.reshape([-1, num_sensors, num_features])

        ## get a tensorflow dataset
        tf_dataset = tf.data.Dataset.from_tensor_slices((reshaped_data_x_np_scaled, data_y_np))

        # load the tensorflow dataset
        tf_dataset_dict[data_split] = tf_dataset

    # now we have a numpy array of input data x and labels y
    # let us try a few examples and see if we get correct results from the pretrained network

    tflite_quantized_model_path = model_base_dir + '/model_quantized.tflite'
    tflite_model_path = model_base_dir + '/model.tflite'


    model_type_list = [('TFLite Quantized', tflite_quantized_model_path), ('TFLite Float', tflite_model_path)]

    # now test the model on train, val, test and write results to a file

    total_df = pandas.DataFrame()

    for model_type, model_path in model_type_list:

        interpreter, input_details, output_details, input_shape = load_trained_tflite_model(model_path)

        results_dict = OrderedDict()

        for data_split in ['train', 'val', 'test']:

            # results entry should have: model_name, correct, total, accuracy
            tf_dataset = tf_dataset_dict[data_split]

            correct = 0
            total = 0

            for i, batch_data in enumerate(tf_dataset):

                x_tensor = batch_data[0].numpy().astype(np.float32)
                x_tensor = np.expand_dims(x_tensor, axis=0)
                y_label_tensor = batch_data[1].numpy()

                interpreter.set_tensor(input_details[0]['index'], x_tensor)
                interpreter.invoke()

                # The function `get_tensor()` returns a copy of the tensor data.
                # Use `tensor()` in order to get a pointer to the tensor.
                output_data = interpreter.get_tensor(output_details[0]['index'])
                preds = np.squeeze(output_data)
                argmax_pred = np.argmax(preds)

                if (y_label_tensor == argmax_pred):
                    correct += 1
                total += 1

            accuracy = float(correct)/float(total)
            print(' ')
            print('data_split: ', data_split)
            print('accuracy: ', float(correct)/float(total))
            print(' ')

            results_entry = [model_type, correct, total, accuracy]
            results_dict[data_split] = results_entry

        results_df = pandas.DataFrame.from_dict(results_dict,orient='index')
        results_df.columns = ['Model Type', 'Correct', 'Total', 'Accuracy']
        total_df = total_df.append(results_df)

    total_df.to_csv(base_dir + '/results.csv')

