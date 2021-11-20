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
    parser.add_argument('--data_dir', type=str, default=UTILS_DIR + '/data/sensor_streams/')

    args = parser.parse_args()

    remove_and_create_dir(args.base_plot_dir)

