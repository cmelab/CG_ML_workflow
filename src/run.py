import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
import wandb
import numpy as np
import constants
from data_loader import load_datasets, get_target_stats
from model import NN
from utils import syn_calculate_error_sum, syn_calculate_error


def get_parameters():
    parser = argparse.ArgumentParser()

    # input & output data params
    parser.add_argument('-data_path',
                        default='/home/marjanalbooyeh/code/cme-lab/ML_datasets/pps_synthesized_Jan30/processed_unnormalized/',
                        type=str, help="path to data")
    parser.add_argument('-log_dir', default='/home/marjanalbooyeh/logs/ML/', type=str, help="path to log data")
    parser.add_argument('-project', default='NN_synthesized_data_v2', type=str, help="w&b project name")

    # ML hyper params
    parser.add_argument('-lr', default=0.01, type=float, help="learning rate")
    parser.add_argument('-decay', default=0.001, type=float, help="weight decay (L2 reg)")
    parser.add_argument('-batch', default=64, type=int, help="batch size")
    parser.add_argument('-hidden_dim', default=64, type=int, help="number of hidden dims")
    parser.add_argument('-n_layer', default=4, type=int, help="number of layers")
    parser.add_argument('-optim', default="SGD", type=str, help="optimizer")

    # Run params
    parser.add_argument('-epochs', default=10000, type=int, help="number of epochs")
    parser.add_argument('-mode', default="single", type=str, help="run mode: single or sweep")
    args = parser.parse_args()
    return args


def create_config(args):
    config = {
        "batch_size": args.batch,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "n_layer": args.n_layer,
        "optim": args.optim
    }
    return config


def train(model, data_loader, optimizer, mse_loss, device, criteria):
    model.train()
    train_loss = 0.
    train_force_error = 0.
    train_torque_error = 0.
    error = 0.
    for i, (feature_tensor, target_tensor) in enumerate(data_loader):
        feature_tensor = feature_tensor.to(device)
        target_tensor = target_tensor.to(device)
        if len(target_tensor.shape) == 3:
            target_force = target_tensor[:, :, :3]
            target_torque = target_tensor[:, :, 3:]
        else:
            target_force = target_tensor[:, :3]
            target_torque = target_tensor[:, 3:]

        optimizer.zero_grad()
        force_prediction, torque_prediction = model(feature_tensor)
        force_loss = mse_loss(force_prediction, target_force)
        torque_loss = mse_loss(torque_prediction, target_torque)
        loss = force_loss + torque_loss
        force_error = criteria(force_prediction, target_force).item()
        torque_error = criteria(torque_prediction, target_torque).item()
        error += force_error + torque_error
        train_loss += loss.item()
        # force_error, torque_error = syn_calculate_error(prediction, target_tensor, target_stats, device)
        # train_force_error += force_error
        # train_torque_error += torque_error
        loss.backward()
        optimizer.step()
    train_loss = train_loss / len(data_loader)

    return train_loss, error/len(data_loader)


def validation(model, data_loader, mse_loss, device, criteria):
    model.eval()
    with torch.no_grad():
        valid_loss = 0.
        valid_force_error = 0.
        valid_torque_error = 0.
        error = 0.
        for i, (feature_tensor, target_tensor) in enumerate(data_loader):
            feature_tensor = feature_tensor.to(device)
            target_tensor = target_tensor.to(device)
            if len(target_tensor.shape) == 3:
                target_force = target_tensor[:, :, :3]
                target_torque = target_tensor[:,:,  3:]
            else:
                target_force = target_tensor[:, :3]
                target_torque = target_tensor[:, 3:]

            force_prediction, torque_prediction = model(feature_tensor)
            force_loss = mse_loss(force_prediction, target_force)
            torque_loss = mse_loss(torque_prediction, target_torque)
            loss = force_loss + torque_loss
            valid_loss += loss.item()
            force_error = criteria(force_prediction, target_force).item()
            torque_error = criteria(torque_prediction, target_torque).item()
            error += force_error + torque_error
            # force_error, torque_error = syn_calculate_error(prediction, target_tensor, target_stats, device)
            # valid_force_error += force_error
            # valid_torque_error += torque_error
        return valid_loss / len(data_loader), error / len(data_loader)


def run(config, log_path, model_path):
    # Load datasets
    train_dataloader, valid_dataloader, test_dataloader = load_datasets(args.data_path, config["batch_size"])
    print('Dataset size: \n\t train: {}, \n\t valid: {}, \n\t test:{}'.
          format(len(train_dataloader), len(valid_dataloader), len(test_dataloader)))
    target_stats = get_target_stats(args.data_path)


    # build model
    model = NN(in_dim=constants.IN_DIM, hidden_dim=config["hidden_dim"], out_dim=constants.OUT_DIM, n_layers=config["n_layer"])
    model.to(device)


    mse_loss = nn.L1Loss().to(device)
    criteria = nn.L1Loss().to(device)
    if config["optim"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["decay"])
    if config["optim"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=100, min_lr=0.001)


    wandb.watch(models=model, criterion=mse_loss, log="all")

    # Train
    print('**************************Training*******************************')
    best_val_error = None
    best_model_path = None
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_val_errors = []
    for epoch in range(args.epochs):

        train_loss, train_error = train(model, train_dataloader, optimizer, mse_loss, device, criteria)
        epoch_train_losses.append(train_loss)
        valid_loss, val_error = validation(model, valid_dataloader, mse_loss, device, criteria)
        # epoch_val_losses.append(valid_loss)
        # val_error = val_force_error + val_torque_error
        # epoch_val_errors.append(val_error)
        scheduler.step(val_error)
        print('epoch {}/{}: \n\t train_loss: {}, \n\t val_loss: {}, \n\t train_error: {}, \n\t val_error: {}'.
                      format(epoch + 1, args.epochs, train_loss, valid_loss, train_error, val_error))
        # if epoch % 20 == 0:
        #         print('epoch {}/{}: \n\t train_loss: {}, \n\t valid_loss: {}, \n\t val_error: {}, \n\t val_force_error: {}, \n \t val_torque_error: {}'.
        #               format(epoch + 1, args.epochs, train_loss, valid_loss, val_error, val_force_error, val_torque_error))
        wandb.log({'train_loss': train_loss,
                   'train_error': train_error,
                   'valid loss': valid_loss,
                   'valid error': val_error,
                   "learning rate": optimizer.param_groups[0]['lr']})


        if best_val_error is None:
            best_val_error = val_error

        if val_error <= best_val_error:
            best_val_error = val_error
            best_model_path = os.path.join(model_path, 'best_model.chkpnt')
            torch.save(model.state_dict(), best_model_path)
            # wandb.save(best_model_path)
            print('#################################################################')
            print('best_val_error: {}, best_epoch: {}'.format(best_val_error, epoch))
            print('#################################################################')
            # wandb.log({'best_epoch': epoch + 1,
            #            'best val error': best_val_error})


    # Testing
    print('**************************Testing*******************************')
    if best_model_path:
        model = NN(in_dim=constants.IN_DIM, hidden_dim=config["hidden_dim"], out_dim=constants.OUT_DIM, n_layers=config["n_layer"])
        model.to(device)
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_loss, test_error= validation(model, test_dataloader, mse_loss, device, criteria)
    print('Testing \n\t test error: {}'.
          format(test_error))

    wandb.log({'test error': test_error})

def run_sweep(project):
    sweep_config = {
        "name": "synthesized-sweep",
        "method": "grid",
        "metric": {"goal": "minimize", "name": "test error"},
        "parameters": {
            # dataset parameters
            "batch_size": {
                "values": [32, 64]
            },

            # model parameters
            "hidden_dim": {
                "values": [64, 128, 256]
            },
            "n_layer": {
                "values": [3, 4, 5]
            },
            "act_fn": {
                "values": ["ReLU", "Tanh"]
            },

            # optimizer parameters
            "optim": {
                "values": ["SGD"]
            },
            "lr": {
                "values": [0.01]
            },
            "decay": {
                "values": [0.1, 0.01]
            }
        }
    }
    sweep_id = wandb.sweep(sweep_config, project=project)

    wandb.agent(sweep_id, function=_main_sweep)

def _main_sweep(config=None):
    with wandb.init(config=None):
        config = wandb.config
        args = get_parameters()
        now = datetime.now()
        log_path = os.path.join(args.log_dir, now.strftime("%Y-%m-%d %H:%M:%S"))
        print('Log path: ', log_path)
        if not os.path.exists(log_path):
            os.mkdir(log_path)

        model_path = os.path.join(log_path, "model")
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        run(config, log_path, model_path)


if __name__ == '__main__':

    args = get_parameters()
    now = datetime.now()
    log_path = os.path.join(args.log_dir, now.strftime("%Y-%m-%d %H:%M:%S"))
    print('Log path: ', log_path)
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    model_path = os.path.join(log_path, "model")
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(args)
    config = create_config(args)
    print(config)

    if args.mode == "single":
        wandb.init(project=args.project, group="single_run", tags=["NN", "synthesized"], dir=log_path,
                   config=config)
        run(config, log_path, model_path)
    else:
        run_sweep(args.project)
