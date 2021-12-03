'''
    Use a neural network in pytorch to classify different activity styles

'''

import pandas
import sys,os
from collections import OrderedDict
import argparse

# generic plotting utils - always use and modify these
# UTILS_DIR=os.environ['UTILS_DIR']
# sys.path.append(UTILS_DIR)
# from plotting_utils import *
# from textfile_utils import *

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch

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


if __name__ == '__main__':

    # Pull arguments from command line.
    parser = argparse.ArgumentParser(description='plot sensor data')

    # where to read data
    parser.add_argument('--train_csv', type=str, default='../processed_data/OL50_10secframe_Proccessed_train_Xy_Matrix.csv')

    parser.add_argument('--val_csv', type=str, default='../processed_data/OL50_10secframe_Proccessed_val_Xy_Matrix.csv')

    args = parser.parse_args()

    ########################################################################
    # Load into dataframes and do some basic stats
    train_df = pandas.read_csv(args.train_csv)
    val_df = pandas.read_csv(args.val_csv)
    # write the shapes of the dataframes
    print(train_df.shape)

	# (2160, 114)

    print(val_df.shape)
	# (720, 114)

    print(train_df.describe())
    # will print stats of each column, clearly see there are outliers

    print(train_df['Activity'].value_counts())
	# >>> train_df['Activity'].value_counts()
	# 0.0    553
	# 1.0    546
	# 2.0    535
	# 3.0    526
	# Name: Activity, dtype: int64

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
    train_x_np, train_y_np, train_x_df, train_y_df = get_xy_numpy(train_df, x_features_columns, y_features_columns=y_features_columns)

    val_x_np, val_y_np, val_x_df, val_y_df = get_xy_numpy(val_df, x_features_columns, y_features_columns=y_features_columns)

    scaler = MinMaxScaler()

    # fit the params of scaling on TRAIN ONLY
    FittedScaler = scaler.fit(train_x_np)

    # now actually transform the training data
    train_x_np_scaled = FittedScaler.transform(train_x_np)

    # use the SAME transformation on the test data
    val_x_np_scaled = FittedScaler.transform(val_x_np)

    # now, create pytorch tensors for the numpy scaled data before feeding it into a DNN
    ########################################################################

    # min, max, mean etc.
    num_features = 10

    num_sensors = int(len(x_features_columns)/num_features)

    # first, convert the SCALED training data to a pytorch tensor
    torch_train_x_tensor = torch.tensor(train_x_np_scaled)
    torch_train_y_tensor = torch.tensor(train_y_np)

    torch_val_x_tensor = torch.tensor(val_x_np_scaled)
    torch_val_y_tensor = torch.tensor(val_y_np)

    print(' ')
    print('training tensors: ')
    print(torch_train_x_tensor.shape, torch_train_y_tensor.shape)
    print(' ')

    print(' ')
    print('val tensors: ')
    print(torch_val_x_tensor.shape, torch_val_y_tensor.shape)
    print(' ')

	# current outputs
	# training tensors:
	# torch.Size([2160, 110]) torch.Size([2160])

	# val tensors:
	# torch.Size([720, 110]) torch.Size([720])

    # KEY next step: now, resize the data so each example is a num_sensor x num_feature matrix
    # so it can nicely be fed into a CNN. the CNN will aggregate features across sensors
    # in our case since we have 11 sensors and 10 features, we should get tensors of size
    # (NUM_EXAMPLES x 11 x 10)
    #####################################################################

    train_x_tensor_resized = torch_train_x_tensor.reshape([-1, num_sensors, num_features])
    val_x_tensor_resized = torch_val_x_tensor.reshape([-1, num_sensors, num_features])

    print(train_x_tensor_resized.shape, val_x_tensor_resized.shape)
    # sanity check: indeed, they have the correct size!
    # torch.Size([2160, 11, 10]) torch.Size([720, 11, 10])

    # now that we have correctly shaped tensors, lets create a pytorch dataloader
    # to cycle through data
    ########################################################################



