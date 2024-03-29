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
            self.input = torch.cat((positions, orientations), 2)
        else:
            orientations  = orientations[:, :, :4]
            self.input = torch.stack((positions, orientations), dim=2)

        self.in_dim = self.input.shape[-1]
        self.forces = forces
        self.torques = torques

    def __len__(self):
        return len(self.input)

    def __getitem__(self, i):
        return self.input[i], self.forces[i], self.torques[i]


def _get_data_loader(dataset, batch_size, shuffle=True):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=5)
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
