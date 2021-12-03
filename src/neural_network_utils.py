"""
    Neural Network Utility Functions in Pytorch
    Author: Sandeep Chinchali
"""


import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_PARAMS = {'batch_size': 32, 'shuffle': True, 'num_workers': 1}

device = "cpu"

def get_pytorch_dataloader(train_x_tensor, train_y_tensor, params=DEFAULT_PARAMS, print_mode = True, max_print=5):

    dataset = TensorDataset(train_x_tensor, train_y_tensor) # create your datset
    dataloader = DataLoader(dataset, **params) # create your dataloader

    if print_mode:
        for i, (x, y) in enumerate(dataloader):
            print(' ')
            print('batch i: ', i)
            print(x.shape)
            print(y.shape)
            print(' ')
            # stop printing after a few
            if i > max_print:
                break
    return dataset, dataloader


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


"""
    Basic convolutional neural network architecture

"""

class BasicConvNet(nn.Module):

    def __init__(self,num_classes=5):
        super(BasicConvNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        # simply to debug outputs
        # print(x.shape, '0')

        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))

        # simply to debug outputs
        # print(x.shape, '1')

        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        # simply to debug outputs
        # print(x.shape, '2')

        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension

        # simply to print and debug
        # print(x.shape, '3')

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':

    # instantiate the ConvNet

    num_activities = 4
    CNN = BasicConvNet(num_classes = num_activities)

    num_examples = 32
    num_sensors = 11
    num_features = 10

    # create some dummy data just to see if the network compiles
    sample_data = torch.randn(num_examples, 1, num_sensors, num_features)

    # call the CNN on the sample input data
    prediction = CNN(sample_data)

    # print the size of the output prediction
    print(prediction.shape)




