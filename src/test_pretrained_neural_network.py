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
import torch
from torch.utils.data import TensorDataset, DataLoader

# helper functions on neural networks
from neural_network_utils import *


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
    parser.add_argument('--train_csv', type=str, default='../processed_data/Feature_Processed_Data/OL50_10secframe_Proccessed_Train_Xy_Matrix.csv')

    parser.add_argument('--val_csv', type=str, default='../processed_data/Feature_Processed_Data/OL50_10secframe_Proccessed_Validate_Xy_Matrix.csv')

    parser.add_argument('--test_csv', type=str, default='../processed_data/Feature_Processed_Data/OL50_10secframe_Proccessed_Test_Xy_Matrix.csv')

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

    train_test_val_dataloaders = OrderedDict()

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

        # first, convert the SCALED training data to a pytorch tensor
        torch_x_tensor = torch.tensor(data_x_np_scaled.astype(np.float32))
        torch_y_tensor = torch.tensor(data_y_np).type(torch.LongTensor)

        # KEY next step: now, resize the data so each example is a num_sensor x num_feature matrix
        # so it can nicely be fed into a CNN. the CNN will aggregate features across sensors
        # in our case since we have 11 sensors and 10 features, we should get tensors of size
        # (NUM_EXAMPLES x 11 x 10)
        #####################################################################

        data_x_tensor_resized = torch_x_tensor.reshape([-1, num_sensors, num_features])

        # STEP 2: CREATE PYTORCH DATALOADERS TO CYCLE THROUGH DATA
        # from now on, all helper functions are in neural_network_utils.py
        ########################################################################

        # now that we have correctly shaped tensors, lets create a pytorch dataloader
        # to cycle through data. this will print data in chunks of:
        # BATCH_SIZE x num_sensors x num_features, where BATCH_SIZE is a default of 32
        ########################################################################

        data_dataset, data_loader = get_pytorch_dataloader(data_x_tensor_resized, torch_y_tensor, params=DEFAULT_PARAMS, print_mode = False, max_print=5)

        train_test_val_dataloaders[data_split] = data_loader

    # STEP 3: TEST A PRE-TRAINED NEURAL NETWORK MODEL
    ########################################################################

    # create a dictionary that has all the data we care for
    # how many epochs to train, how many batches are fed in, where to plot etc.

    PLOT_DIR = SCRATCH_DIR + '/plots/'
    MODEL_DIR = SCRATCH_DIR + '/models/'

    test_options = {"train_loader": train_test_val_dataloaders['train'],
                     "val_loader": train_test_val_dataloaders['val'],
                     "test_loader": train_test_val_dataloaders['test'],
                     "batch_size": 64,
                     "model_save_path": MODEL_DIR,
                     "num_epochs": 50,
                     "plot_dir": PLOT_DIR,
                     "model_name": 'ConvNet Tool Activity Classifier'}

    model_name = "  ".join((test_options["model_name"],
                            "Epochs: "+str(test_options["num_epochs"])))


    ## set up model (defined in utils)
    model = BasicConvNet(num_classes = num_activities)

    # loss/error function
    loss_func = torch.nn.CrossEntropyLoss()

    # IF we use the model we just trained
    # load in the pretrained weights
    # model_path = test_options["model_save_path"] + model_name + '.pt'

    # ELSE, use the default model Sandeep trained
    model_path = SMART_TOOLS_ROOT_DIR + '/pretrained_models/PyTorch/' + model_name + '.pt'

    model.load_state_dict(torch.load(model_path)['model_state_dict'])

    classes = ('Engrave', 'Sand', 'Cut', 'Route')


    for data_split, data_loader in train_test_val_dataloaders.items():

        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        # again no gradients needed
        with torch.no_grad():
            for data in data_loader:
                images, labels = data
                images_reshaped = images.unsqueeze(1)
                outputs = model(images_reshaped)
                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label_tensor, prediction in zip(labels, predictions):
                    label = label_tensor.item()
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1


        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(' ')
            print('Data Split: ', data_split)
            print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))
            print(' ')


