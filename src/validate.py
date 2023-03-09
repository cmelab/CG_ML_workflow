import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
import wandb
import numpy as np
import constants
from data_loader import load_datasets
from model import NN
from utils import syn_calculate_error_sum, syn_calculate_error


def get_parameters():
    parser = argparse.ArgumentParser()

    # input & output data params
    parser.add_argument('-data_path',
                        default='/home/marjanalbooyeh/logs/datasets/pps_two_synthesized/',
                        type=str, help="path to data")
    parser.add_argument('-model_path', default='/home/marjanalbooyeh/logs/ML/2023-03-01 15:18:04/model/best_model.pth', type=str, help="path to log data")
    # ML hyper params
    parser.add_argument('-batch', default=64, type=int, help="batch size")
    parser.add_argument('-hidden_dim', default=64, type=int, help="number of hidden dims")
    parser.add_argument('-n_layer', default=2, type=int, help="number of layers")
    parser.add_argument('-inp_mode', default='append', type=str, help="input data mode: append or stack")
    parser.add_argument('-act_fn', default="ReLU", type=str, help="activation func")

    args = parser.parse_args()
    return args

def create_config(args):
    config = {
        "batch_size": args.batch,
        "hidden_dim": args.hidden_dim,
        "n_layer": args.n_layer,
        "model_path": args.model_path,
        "act_fn" : args.act_fn,
        "inp_mode": args.inp_mode
    }
    return config




def validation(model, data_loader, device, criteria):
    model.eval()
    with torch.no_grad():
        error = 0.
        for i, (input, target_force, target_torque) in enumerate(data_loader):
            feature_tensor = input.to(device)
            target_force = target_force.to(device)
            target_torque = target_torque.to(device)

            force_prediction, torque_prediction = model(feature_tensor)
            print('input', feature_tensor[20])
            print('torque pred: ', torque_prediction[20])
            print('target: ', target_torque[2])
            print('force pred: ', force_prediction[20])
            print('target f: ', target_force[2])
            print('*******')
            force_error = criteria(force_prediction, target_force).item()
            torque_error = criteria(torque_prediction, target_torque).item()
            error += force_error + torque_error

        return error / len(data_loader)


def run(config, device):
    # Load datasets
    train_dataloader, valid_dataloader, test_dataloader = load_datasets(args.data_path, config["batch_size"], inp_mode=config["inp_mode"])
    print('Dataset size: \n\t train: {}, \n\t valid: {}, \n\t test:{}'.
          format(len(train_dataloader), len(valid_dataloader), len(test_dataloader)))


    # build model
    model = NN(in_dim=test_dataloader.dataset.in_dim, hidden_dim=config["hidden_dim"], out_dim=constants.OUT_DIM, n_layers=config["n_layer"], act_fn=config["act_fn"])
    model.to(device)

    criteria = nn.L1Loss().to(device)
    # Testing
    print('**************************Testing*******************************')

    model.load_state_dict(torch.load(config["model_path"], map_location=device))

    test_error = validation(model, test_dataloader, device, criteria)
    print('Testing \n\t test error: {}'.
          format(test_error))




if __name__ == '__main__':

    args = get_parameters()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(args)
    config = create_config(args)

    run(config, device)