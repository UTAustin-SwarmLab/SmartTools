import numpy as np
import torch
import os, sys

ROBOTICS_CODESIGN_DIR = os.environ['ROBOTICS_CODESIGN_DIR']
sys.path.append(ROBOTICS_CODESIGN_DIR)
sys.path.append(ROBOTICS_CODESIGN_DIR + '/utils/')

SCRATCH_DIR = ROBOTICS_CODESIGN_DIR + '/scratch/'

BASE_DIR = SCRATCH_DIR + '/ball_catch/'
DATALOADER_DIR = BASE_DIR + '/pytorch_dataset/'

from textfile_utils import *
from plotting_utils import *
from torch.utils.data import TensorDataset, DataLoader

DEFAULT_PARAMS = {'batch_size': 32, 'shuffle': True, 'num_workers': 1}


def create_dataloader_from_tensors(base_dir, params = DEFAULT_PARAMS, print_mode = False, train_type = 'train'):

    tensor_x = torch.load(base_dir + '/x_full.pt')

    print('###########################')
    print('train_type: ', train_type)
    print('tensor x shape: ', tensor_x.shape)

    tensor_y = torch.load(base_dir + '/y_full.pt')

    print('tensor y shape: ', tensor_y.shape)
    print('###########################')

    projectile_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
    projectile_dataloader = DataLoader(projectile_dataset, **params) # create your dataloader

    if print_mode:
        for i, (x, y) in enumerate(projectile_dataloader):
            print(' ')
            print('batch i: ', i)
            print(x.shape)
            print(y.shape)
            print(' ')
    return projectile_dataset, projectile_dataloader

if __name__ == '__main__':

    data_types = ['train', 'val', 'test']

    for data_type in data_types:
        SUB_DATALOADER_DIR = DATALOADER_DIR + '/' + data_type

        projectile_dataset, projectile_dataloader = create_dataloader_from_tensors(SUB_DATALOADER_DIR, print_mode = False, train_type = data_type)




