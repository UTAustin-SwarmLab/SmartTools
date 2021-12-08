'''
    Use a neural network in pytorch to classify different activity styles

'''

import pandas
import sys,os
from collections import OrderedDict
import argparse
import numpy as np

# where the base code is on your machine
SMART_TOOLS_ROOT_DIR = os.environ['SMART_TOOLS_ROOT_DIR']
SCRATCH_DIR = SMART_TOOLS_ROOT_DIR + '/scratch/'

# generic plotting utils - always use and modify these
UTILS_DIR=os.environ['UTILS_DIR']
sys.path.append(UTILS_DIR)
from plotting_utils import *
from textfile_utils import *

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf

# helper function to extract key columns from the pandas dataframes

def get_xy_numpy(df, x_features_columns, y_features_columns='Activity'):
    x_df = df[x_features_columns]
    x_np = x_df.to_numpy()
    # get the output column we want to predict
    y_df = df[y_features_columns]
    y_np = y_df.to_numpy()
    # assert the x and y dataframes do NOT have any null or NaN entries
    assert(x_df.isnull().sum().sum() == 0)
    assert(y_df.isnull().sum().sum() == 0)

    return x_np, y_np, x_df, y_df


BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100



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

    # min, max, mean etc.
    num_features = 10

    for data_split, data_df in train_test_val_df_dict.items():

        data_x_np, data_y_np, data_x_df, data_y_df = get_xy_numpy(data_df, x_features_columns, y_features_columns=y_features_columns)

        if data_split == 'train':
            scaler = MinMaxScaler()

            # fit the params of scaling on TRAIN ONLY
            FittedScaler = scaler.fit(data_x_np)

        # now actually transform the training data
        data_x_np_scaled = FittedScaler.transform(data_x_np)

        num_sensors = int(len(x_features_columns)/num_features)

        # x: data_x_np_scaled
        # y: data_y_np
        print(' ')
        print(' ')
        print('data_split: ', data_split)
        print('data_x_np: ', data_x_np.shape)
        print('data_y_np: ', data_y_np.shape)
        print(' ')
        print(' ')

        # get a tensorflow dataset
        tf_dataset = tf.data.Dataset.from_tensor_slices((data_x_np_scaled, data_y_np))

        # load the tensorflow dataset
        tf_dataset_dict[data_split] = tf_dataset


train_dataset = tf_dataset_dict['train'].shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
val_dataset = tf_dataset_dict['val'].batch(BATCH_SIZE)
test_dataset = tf_dataset_dict['test'].batch(BATCH_SIZE)



