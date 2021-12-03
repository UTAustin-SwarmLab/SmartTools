import sys, os
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from forecaster_model import *
import numpy as np


ROBOTICS_CODESIGN_DIR = os.environ['ROBOTICS_CODESIGN_DIR']
sys.path.append(ROBOTICS_CODESIGN_DIR)
sys.path.append(ROBOTICS_CODESIGN_DIR + '/utils/')

SCRATCH_DIR = ROBOTICS_CODESIGN_DIR + '/scratch/'

BASE_DIR = SCRATCH_DIR + '/ball_catch/'
DATALOADER_DIR = BASE_DIR + '/pytorch_dataset/'

from textfile_utils import *
from plotting_utils import *
from load_saved_data import *

device = "cpu"

def train_forecast_model(model_params, train_options):

    train_dataset, train_loader = create_dataloader_from_tensors(train_options['train_data_path'])
    val_dataset, val_loader = create_dataloader_from_tensors(train_options['val_data_path'])

    # x value
    time_horizon = train_dataset[0][0].shape[0]
    x_dim = train_dataset[0][0].shape[1]
    y_dim = train_dataset[0][1].shape[1]

    print(' ')
    print('time_horizon: ', time_horizon)
    print('x_dim: ', x_dim)
    print('y_dim: ', y_dim)
    print(' ')

    input_dim = time_horizon * x_dim
    output_dim = time_horizon * y_dim

    # set up model
    model = Feedforward(input_dim,
                        output_dim,
                        latent_dim = model_params["latent_dim"])

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=train_options["learning_rate"], amsgrad=True)
    loss_func = torch.nn.MSELoss()

    train_RMSE = []
    val_RMSE = []

    for i in range(train_options["num_epochs"]):
        train_loss = 0
        for batch_num, (batch_x, batch_y) in enumerate(train_loader):

            batch_x_reshaped, batch_y_reshaped = resize_batch_per_model_type(batch_x, batch_y)

            # forward pass
            predictions = model(batch_x_reshaped)

            # backwards pass (backpropagate gradients)
            loss = loss_func(predictions, batch_y_reshaped)
            train_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (i + 1) % train_options["output_freq"] == 0:
            print(" ")
            train_RMSE.append(train_loss)
            print("Epoch %d: Train RMSE %0.3f" % (i + 1, train_loss))


            model.eval()
            with torch.no_grad():
                val_loss = 0
                for batch_num, (val_x, val_y) in enumerate(val_loader):
                    val_x_reshaped, val_y_reshaped = resize_batch_per_model_type(val_x, val_y)

                    predictions = model(val_x_reshaped)
                    val_loss += loss_func(predictions, val_y_reshaped)

                print("Epoch %d: Val RMSE %0.3f" % (i + 1, val_loss))
                val_RMSE.append(val_loss)

            print(" ")
            model.train()


    model_name = "_".join((model.name,
                           "Epochs"+str(train_options["num_epochs"]),
                           "LatentDim"+str(model_params["latent_dim"])))

    plt.plot(np.array(train_RMSE[1:]))
    plt.plot(np.array(val_RMSE[1:]))
    plt.legend(["train", "val"])
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Prediction Model")
    plt.savefig(train_options['plot_dir'] + '/' + model_name)

    if train_options["save_model"]:
        model_path = train_options["model_save_path"] + model_name + '.pt'
        print('model_path: ', model_path)

        model_save_dict = {'epoch': i, 'model_state_dict': model.state_dict(), \
            'optimizer_state_dict': optimizer.state_dict()}

        torch.save(model_save_dict, model_path)

    return model

if __name__=="__main__":

    MODEL_DIR = BASE_DIR + '/trained_models/'
    remove_and_create_dir(MODEL_DIR)

    PLOT_DIR = BASE_DIR + '/training_progress_plots/'
    remove_and_create_dir(PLOT_DIR)

    train_options = {"train_data_path": DATALOADER_DIR + '/train/',
                     "val_data_path": DATALOADER_DIR + '/val/',
                     "num_epochs": 50,
                     "batch_size": 32,
                     "learning_rate": 1e-3,
                     "output_freq": 1,
                     "checkpoint_freq": 25,
                     "save_model": True,
                     "model_save_path": MODEL_DIR,
                     "plot_dir": PLOT_DIR}

    model_params = {"latent_dim": 64}

    model = train_forecast_model(model_params, train_options)
