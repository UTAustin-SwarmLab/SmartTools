'''
    Use a neural network in pytorch to classify different activity styles

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

# where the tensorflow logs are placed
# helper function to extract key columns from the pandas dataframes

# how many samples are taken at once for training
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

epochs = 50

# NOTE: data will be re-organized to size:
# BATCH_SIZE x NUM_SENSORS x NUM_FEATURES
# view this as an image with 1 channel, NUM_SENSORS x NUM_FEATURES size


if __name__ == '__main__':

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

    train_quantile_csv =

    # min, max, mean etc.
    num_features = 10

    num_sensors = int(len(x_features_columns)/num_features)

    # SCALE THE DATA and create a Tensorflow DataSet
    # analagous to a Pytorch data loader

    for data_split, data_df in train_test_val_df_dict.items():

        data_x_np, data_y_np, data_x_df, data_y_df = get_xy_numpy(data_df, x_features_columns, y_features_columns=y_features_columns)

        quantile_list = [.001, 0.25, 0.5, 0.75, 0.999]

        # only for training data, get the above quantiles for ALL COLUMNS and save to a csv
        if data_split == 'train':
            # do not use sklearn, instead save the following quantiles of data to a dataframe and store as a csv
            train_quantile_df = data_x_df.quantile(quantile_list)

            train_quantile_df.to_csv(train_quantile_csv)

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
        print(normalized_data_x_df.describe())
        print(' ')
        print(' ')


        ## now actually transform the training data
        data_x_np_scaled = normalized_data_x_df.numpy()

        ## BATCH_SIZE x NUM_SENSORS x NUM_FEATURES
        ## view this as an image with 1 channel, NUM_SENSORS x NUM_FEATURES size

        reshaped_data_x_np_scaled = data_x_np_scaled.reshape([-1, num_sensors, num_features])

        ## get a tensorflow dataset
        tf_dataset = tf.data.Dataset.from_tensor_slices((reshaped_data_x_np_scaled, data_y_np))

        # load the tensorflow dataset
        tf_dataset_dict[data_split] = tf_dataset
