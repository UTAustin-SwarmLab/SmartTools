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

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf

from utils_tensorflow import *

# where the tensorflow logs are placed

logdir = "logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

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


# how many samples are taken at once for training
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

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

    # min, max, mean etc.
    num_features = 10

    num_sensors = int(len(x_features_columns)/num_features)

    # SCALE THE DATA and create a Tensorflow DataSet
    # analagous to a Pytorch data loader

    for data_split, data_df in train_test_val_df_dict.items():

        data_x_np, data_y_np, data_x_df, data_y_df = get_xy_numpy(data_df, x_features_columns, y_features_columns=y_features_columns)

        if data_split == 'train':
            scaler = MinMaxScaler()

            # fit the params of scaling on TRAIN ONLY
            FittedScaler = scaler.fit(data_x_np)

        # now actually transform the training data
        data_x_np_scaled = FittedScaler.transform(data_x_np)

        # BATCH_SIZE x NUM_SENSORS x NUM_FEATURES
        # view this as an image with 1 channel, NUM_SENSORS x NUM_FEATURES size

        reshaped_data_x_np_scaled = data_x_np_scaled.reshape([-1, num_sensors, num_features])

        # x: data_x_np_scaled
        # y: data_y_np
        print(' ')
        print(' ')
        print('data_split: ', data_split)
        print('data_x_np: ', data_x_np_scaled.shape)
        print('reshaped_data_x_np: ', reshaped_data_x_np_scaled.shape)
        print('data_y_np: ', data_y_np.shape)
        print(' ')
        print(' ')

        # get a tensorflow dataset
        tf_dataset = tf.data.Dataset.from_tensor_slices((reshaped_data_x_np_scaled, data_y_np))

        # load the tensorflow dataset
        tf_dataset_dict[data_split] = tf_dataset


    # we now have all the datasets and dataloaders in TENSORFLOW format
    train_data = tf_dataset_dict['train'].shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    val_data = tf_dataset_dict['val'].batch(BATCH_SIZE)
    test_data = tf_dataset_dict['test'].batch(BATCH_SIZE)

    # find the number of batches in the TESTING dataset
    # this is useful later
    test_len = 0
    for batch in test_data:
        test_len += 1

    # now build the CNN model
    base_dir = SCRATCH_DIR + '/tensorflow_classifier/'
    remove_and_create_dir(base_dir)

    model_base_dir = base_dir + '/tf_model/'
    remove_and_create_dir(model_base_dir)

    # 1D CNN model
    model, model_path = build_1D_CNN(model_base_dir, model_name = '1DCNN', num_sensors = num_sensors, num_features = num_features, num_outputs = num_activities)

    # How many KB is the final tensorflow model?
    model_size = calculate_model_size(model)

    # when we train, we save a csv of the training accuracy/loss
    csv_logger = tf.keras.callbacks.CSVLogger(base_dir + '/training.log')

    # how long we train, set up model with loss function
    epochs = 50
    batch_size = 64
    model.compile(optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])

    #test_labels = np.zeros(test_len)
    #idx = 0
    #for data, label in test_data:  # pylint: disable=unused-variable
    #    test_labels[idx] = label.numpy()
    #    idx += 1

    # now, finally start training
    model.fit(train_data,
            epochs=epochs,
            validation_data=val_data,
            callbacks=[tensorboard_callback, csv_logger])

    loss, acc = model.evaluate(test_data)
    pred = np.argmax(model.predict(test_data), axis=1)

    #confusion = tf.math.confusion_matrix(labels=tf.constant(test_labels),
    #                                   predictions=tf.constant(pred),
    #                                   num_classes=4)
    #print(confusion)
    print("Test Loss {}, Test Accuracy {}".format(loss, acc))
