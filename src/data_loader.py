import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class CustomTrajDataset(Dataset):
    def __init__(self, traj_df, mode="append"):
        positions = torch.from_numpy(np.array(list(traj_df['position']))).type(torch.FloatTensor)
        orientations = torch.from_numpy(np.array(list(traj_df['orientation']))).type(torch.FloatTensor)
        forces = torch.from_numpy(np.array(list(traj_df['net_force']))).type(torch.FloatTensor)
        torques = torch.from_numpy(np.array(list(traj_df['net_torque']))).type(torch.FloatTensor)

        if mode == "append":
            self.inputs = torch.cat((positions, orientations), dim=-1)
        else:
            padding = torch.zeros((positions.shape[0], positions.shape[1], 1))
            positions = torch.cat([positions, padding], dim=-1)
            self.inputs = torch.stack((positions, orientations), dim=-2)

        self.in_dim = self.inputs.shape[-1]
        self.batch_dim = self.inputs.shape[-2]
        self.forces = forces
        self.torques = torques

        self.input_shape = self.inputs.shape

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        return self.inputs[i], self.forces[i], self.torques[i]


def _get_data_loader(dataset, batch_size, shuffle=True):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)
    return dataloader


def load_datasets(data_path, batch_size, inp_mode="append"):
    train_df = pd.read_pickle(os.path.join(data_path, 'train.pkl'))
    val_df = pd.read_pickle(os.path.join(data_path, 'val.pkl'))
    test_df = pd.read_pickle(os.path.join(data_path, 'test.pkl'))
    train_dataset = CustomTrajDataset(train_df, mode=inp_mode)
    valid_dataset = CustomTrajDataset(val_df, mode=inp_mode)
    test_dataset = CustomTrajDataset(test_df, mode=inp_mode)

    train_dataloader = _get_data_loader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = _get_data_loader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = _get_data_loader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, valid_dataloader, test_dataloader
