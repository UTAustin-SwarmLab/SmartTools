import pandas
import sys,os
from collections import OrderedDict
import argparse

# generic plotting utils - always use and modify these
UTILS_DIR=os.environ['UTILS_DIR']
sys.path.append(UTILS_DIR)

from plotting_utils import *
from textfile_utils import *

if __name__ == '__main__':

    # Pull arguments from command line.
    parser = argparse.ArgumentParser(description='plot sensor data')

    # where to read data
    parser.add_argument('--train_csv', type=str, default='../processed_data/OL50_10secframe_Proccessed_train_Xy_Matrix.csv')

    parser.add_argument('--val_csv', type=str, default='../processed_data/OL50_10secframe_Proccessed_val_Xy_Matrix.csv')

    args = parser.parse_args()

    # Load into dataframes
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

