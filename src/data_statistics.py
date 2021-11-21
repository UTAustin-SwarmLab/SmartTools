import pandas
import sys,os
from collections import OrderedDict
import argparse

# generic plotting utils - always use and modify these
UTILS_DIR=os.environ['UTILS_DIR']
sys.path.append(UTILS_DIR)

from plotting_utils import *
from textfile_utils import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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

    y_features_columns = ['Activity']

    # repeat key column extraction for train and val data
    train_x_np, train_y_np, train_x_df, train_y_df = get_xy_numpy(train_df, x_features_columns, y_features_columns='Activity')

    val_x_np, val_y_np, val_x_df, val_y_df = get_xy_numpy(train_df, x_features_columns, y_features_columns='Activity')

    # now, let us train a basic random forest classifier WITHOUT data normalization
    ########################################################################
    clf = RandomForestClassifier().fit(train_x_np,train_y_np)
    prediction= clf.predict(val_x_np)
    accuracy_score = accuracy_score(y_val, prediction)*100
    print('accuracy: ', accuracy_score)


