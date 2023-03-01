import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class CustomTrajDataset(Dataset):
    def __init__(self, traj_df):
        positions = torch.from_numpy(np.array(list(traj_df['position']))).type(torch.FloatTensor)
        orientations = torch.from_numpy(np.array(list(traj_df['orientation']))).type(torch.FloatTensor)
        forces = torch.from_numpy(np.array(list(traj_df['net_force']))).type(torch.FloatTensor)
        torques = torch.from_numpy(np.array(list(traj_df['net_torque']))).type(torch.FloatTensor)

        self.input = torch.cat((positions, orientations), 2)
        self.forces = forces
        self.torques = torques

    def __len__(self):
        return len(self.input)

    def __getitem__(self, i):
        return self.input[i], self.forces[i], self.torques[i]


def _get_data_loader(dataset, batch_size, shuffle=True):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)
    return dataloader


def load_datasets(data_path, batch_size):
    train_df = pd.read_pickle(os.path.join(data_path, 'train.pkl'))
    val_df = pd.read_pickle(os.path.join(data_path, 'val.pkl'))
    test_df = pd.read_pickle(os.path.join(data_path, 'test.pkl'))
    train_dataset = CustomTrajDataset(train_df)
    valid_dataset = CustomTrajDataset(val_df)
    test_dataset = CustomTrajDataset(test_df)

    train_dataloader = _get_data_loader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = _get_data_loader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = _get_data_loader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, valid_dataloader, test_dataloader
