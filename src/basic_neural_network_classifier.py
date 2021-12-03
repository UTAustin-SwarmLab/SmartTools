'''
    Use a neural network in pytorch to classify different activity styles

'''

import pandas
import sys,os
from collections import OrderedDict
import argparse
import numpy as np

# generic plotting utils - always use and modify these
# UTILS_DIR=os.environ['UTILS_DIR']
# sys.path.append(UTILS_DIR)
# from plotting_utils import *
# from textfile_utils import *

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
    parser.add_argument('--train_csv', type=str, default='../processed_data/OL50_10secframe_Proccessed_train_Xy_Matrix.csv')

    parser.add_argument('--val_csv', type=str, default='../processed_data/OL50_10secframe_Proccessed_val_Xy_Matrix.csv')

    args = parser.parse_args()

    # STEP 0: PANDAS DATA MANIPULATION, SAME AS RANDOM FOREST
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
    train_x_np, train_y_np, train_x_df, train_y_df = get_xy_numpy(train_df, x_features_columns, y_features_columns=y_features_columns)

    val_x_np, val_y_np, val_x_df, val_y_df = get_xy_numpy(val_df, x_features_columns, y_features_columns=y_features_columns)

    scaler = MinMaxScaler()

    # fit the params of scaling on TRAIN ONLY
    FittedScaler = scaler.fit(train_x_np)

    # now actually transform the training data
    train_x_np_scaled = FittedScaler.transform(train_x_np)

    # use the SAME transformation on the test data
    val_x_np_scaled = FittedScaler.transform(val_x_np)

    # STEP 1: CREATE PYTORCH TENSORS
    ########################################################################

    # now, create pytorch tensors for the numpy scaled data before feeding it into a DNN
    ########################################################################

    # min, max, mean etc.
    num_features = 10

    num_sensors = int(len(x_features_columns)/num_features)

    # first, convert the SCALED training data to a pytorch tensor
    torch_train_x_tensor = torch.tensor(train_x_np_scaled.astype(np.float32))
    #torch_train_y_tensor = torch.tensor(train_y_np.astype(np.float32))
    torch_train_y_tensor = torch.tensor(train_y_np).type(torch.LongTensor)

    torch_val_x_tensor = torch.tensor(val_x_np_scaled.astype(np.float32))
    #torch_val_y_tensor = torch.tensor(val_y_np.astype(np.float32))
    torch_val_y_tensor = torch.tensor(val_y_np).type(torch.LongTensor)

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


    # STEP 2: CREATE PYTORCH DATALOADERS TO CYCLE THROUGH DATA
    # from now on, all helper functions are in neural_network_utils.py
    ########################################################################

    # now that we have correctly shaped tensors, lets create a pytorch dataloader
    # to cycle through data. this will print data in chunks of:
    # BATCH_SIZE x num_sensors x num_features, where BATCH_SIZE is a default of 32
    ########################################################################

    train_dataset, train_loader = get_pytorch_dataloader(train_x_tensor_resized, torch_train_y_tensor, params=DEFAULT_PARAMS, print_mode = False, max_print=5)

    val_dataset, val_loader = get_pytorch_dataloader(val_x_tensor_resized, torch_val_y_tensor, params=DEFAULT_PARAMS, print_mode = False, max_print=5)


    # STEP 3: CREATE A SIMPLE NEURAL NETWORK MODEL, LOSS FUNCTION, OPTIMIZER
    ########################################################################

    train_options = {"train_loader": train_loader,
                     "val_loader": val_loader,
                     "num_epochs": 10,
                     "batch_size": 32,
                     "learning_rate": 1e-3,
                     "output_freq": 1,
                     "checkpoint_freq": 5,
                     "save_model": False,
                     "model_save_path": None,
                     "plot_dir": None}


    # set up model (defined in utils)
    model = BasicConvNet(num_classes = num_activities)

    # set up an optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                              lr=train_options["learning_rate"], amsgrad=True)

    # loss/error function
    loss_func = torch.nn.CrossEntropyLoss()

    # save the loss in arrays
    train_loss_vec = []
    val_loss_vec = []

    # loop through the data for NUM_EPOCHS iterations to learn the model
    for i in range(train_options["num_epochs"]):
        train_loss = 0
        for batch_num, (batch_x, batch_y) in enumerate(train_options['train_loader']):

            #batch_x_reshaped, batch_y_reshaped = resize_batch_per_model_type(batch_x, batch_y)
            batch_x_reshaped = batch_x.unsqueeze(1)
            batch_y_reshaped = batch_y

            print(' ')
            print(batch_x.shape, batch_x_reshaped.shape)
            print(batch_y.shape, batch_y_reshaped.shape)
            print(' ')

            # forward pass
            predictions = model(batch_x_reshaped)

            # backwards pass (backpropagate gradients)
            loss = loss_func(predictions, batch_y_reshaped)

            # add to running sum of loss
            train_loss += loss.item()

            # take a gradient descent step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # how often to print loss
        if (i + 1) % train_options["output_freq"] == 0:
            print(" ")
            train_loss_vec.append(train_loss)
            print("Epoch %d: Train loss %0.3f" % (i + 1, train_loss))

            # now freeze the model periodically to test it on validation data
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for batch_num, (val_x, val_y) in enumerate(train_options['val_loader']):

                    val_x_reshaped = val_x.unsqueeze(1)
                    val_y_reshaped = val_y

                    predictions = model(val_x_reshaped)
                    val_loss += loss_func(predictions, val_y_reshaped)

                print("Epoch %d: Val loss %0.3f" % (i + 1, val_loss))
                val_loss_vec.append(val_loss.item())

            print(" ")
            # reset the model for training
            model.train()

    print(' ')
    print('train_loss')
    print(train_loss_vec)
    print(' ')

    print(' ')
    print('val_loss')
    print(val_loss_vec)
    print(' ')

    ## plot the losses and optionally save the model to a file on disk
    #model_name = "_".join((model.name,
    #                       "Epochs"+str(train_options["num_epochs"]),
    #                       "LatentDim"+str(model_params["latent_dim"])))

    #plt.plot(np.array(train_loss[1:]))
    #plt.plot(np.array(val_loss[1:]))
    #plt.legend(["train", "val"])
    #plt.xlabel("Epoch")
    #plt.ylabel("loss")
    #plt.title("Prediction Model")
    #plt.savefig(train_options['plot_dir'] + '/' + model_name)

    #if train_options["save_model"]:
    #    model_path = train_options["model_save_path"] + model_name + '.pt'
    #    print('model_path: ', model_path)

    #    model_save_dict = {'epoch': i, 'model_state_dict': model.state_dict(), \
    #        'optimizer_state_dict': optimizer.state_dict()}

    #    torch.save(model_save_dict, model_path)

    #return model
