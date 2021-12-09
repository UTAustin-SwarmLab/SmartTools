import pandas
import numpy as np
import sys,os

# where the base code is on your machine
SMART_TOOLS_ROOT_DIR = os.environ['SMART_TOOLS_ROOT_DIR']
SCRATCH_DIR = SMART_TOOLS_ROOT_DIR + '/scratch/'

# generic plotting utils - always use and modify these
UTILS_DIR=os.environ['UTILS_DIR']
sys.path.append(UTILS_DIR)
from plotting_utils import *
from textfile_utils import *


if __name__ == '__main__':

    base_dir = SCRATCH_DIR + '/tensorflow_classifier/'

    log_file = base_dir + '/training.log'

    df = pandas.read_csv(log_file)
    train_accuracy_vec = np.array(df['accuracy'])
    val_accuracy_vec = np.array(df['val_accuracy'])

    print(' ')
    print('train_loss')
    print(train_accuracy_vec)
    print(' ')

    print(' ')
    print('val_loss')
    print(val_accuracy_vec)
    print(' ')

    plt.plot(np.array(train_accuracy_vec))
    plt.plot(np.array(val_accuracy_vec))
    plt.legend(["train", "val"])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title('TF Model')
    plt.savefig(base_dir + '/accuracy.pdf')
    plt.close()
