import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class CustomTrajDataset(Dataset):
    def __init__(self, traj_df):
        positions = torch.from_numpy(np.array(list(traj_df['position']))).type(torch.FloatTensor)
        orientations = torch.from_numpy(np.array(list(traj_df['orientation']))).type(torch.FloatTensor)
        self.x = torch.cat((positions, orientations), 2)

        forces = torch.from_numpy(np.array(list(traj_df['net_force']))).type(torch.FloatTensor)
        torques = torch.from_numpy(np.array(list(traj_df['net_torque']))).type(torch.FloatTensor)

        self.y = torch.cat((forces, torques), 2)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class SynthesizedTrajDataset(Dataset):
    def __init__(self, traj_df):
        positions = torch.from_numpy(np.array(list(traj_df['position']))).type(torch.FloatTensor)
        orientations = torch.from_numpy(np.array(list(traj_df['orientation']))).type(torch.FloatTensor)
        self.x = torch.cat((positions, orientations), 1)

        forces = torch.from_numpy(np.array(list(traj_df['net_force']))).type(torch.FloatTensor)
        torques = torch.from_numpy(np.array(list(traj_df['net_torque']))).type(torch.FloatTensor)

        self.y = torch.cat((forces, torques), 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def _get_data_loader(dataset, batch_size, shuffle=True):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)
    return dataloader


def load_datasets(data_path, batch_size):
    train_df = pd.read_pickle(os.path.join(data_path, 'train.pkl'))[:20]
    val_df = pd.read_pickle(os.path.join(data_path, 'val.pkl'))
    test_df = pd.read_pickle(os.path.join(data_path, 'test.pkl'))
    train_dataset = SynthesizedTrajDataset(train_df)
    valid_dataset = SynthesizedTrajDataset(val_df)
    test_dataset = SynthesizedTrajDataset(test_df)

    train_dataloader = _get_data_loader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = _get_data_loader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = _get_data_loader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, valid_dataloader, test_dataloader


def get_target_stats(data_path):
    with open(os.path.join(data_path, 'stats.pkl'), 'rb') as fp:
        target_stats = pickle.load(fp)
    return target_stats