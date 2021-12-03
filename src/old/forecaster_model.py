import torch
import torch.nn as nn
import torch.nn.functional as F

class Feedforward(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dim=32):
        super(Feedforward, self).__init__()

        self.fc1 = nn.Linear(input_dim, latent_dim)
        #self.bn1 = nn.BatchNorm1d(num_features=latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        #self.bn2 = nn.BatchNorm1d(num_features=latent_dim)
        self.fc3 = nn.Linear(latent_dim, output_dim)

        self.name = "ForecastNN"

    def forward(self, z):
        print(z.shape)
        print(self.fc1)
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = self.fc3(z)

        #z = F.relu(self.bn1(self.fc1(z)))
        #z = F.relu(self.bn2(self.fc2(z)))
        #z = self.fc3(z)

        return z

def resize_batch_per_model_type(batch_x, batch_y, model_type = 'linear'):

    batch_sz = batch_x.shape[0]

    input_dim = batch_x.shape[1] * batch_x.shape[2]

    output_dim = batch_y.shape[1] * batch_y.shape[2]

    batch_x_reshaped = batch_x.reshape(batch_sz, input_dim)

    batch_y_reshaped = batch_y.reshape(batch_sz, output_dim)

    return batch_x_reshaped, batch_y_reshaped
