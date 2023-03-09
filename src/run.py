import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
import wandb

import constants
from data_loader import load_datasets
from model import NN


def get_parameters():
    parser = argparse.ArgumentParser()

    # input & output data params
    parser.add_argument('-data_path',
                        default='/home/marjanalbooyeh/logs/datasets/pps_two_synthesized/neighbors/',
                        type=str, help="path to data")
    parser.add_argument('-log_dir', default='/home/marjanalbooyeh/logs/ML/', type=str, help="path to log data")
    parser.add_argument('-project', default='NN_synthesized_March1', type=str, help="w&b project name")
    parser.add_argument('-group', default='two_pps', type=str, help="w&b run group")
    parser.add_argument('-notes', default='', type=str, help="w&b run note")
    parser.add_argument('-inp_mode', default='append', type=str, help="input data mode: append or stack")

    # ML hyper params
    parser.add_argument('-lr', default=0.01, type=float, help="learning rate")
    parser.add_argument('-decay', default=0.001, type=float, help="weight decay (L2 reg)")
    parser.add_argument('-batch', default=64, type=int, help="batch size")
    parser.add_argument('-hidden_dim', default=64, type=int, help="number of hidden dims")
    parser.add_argument('-n_layer', default=2, type=int, help="number of layers")
    parser.add_argument('-optim', default="SGD", type=str, help="optimizer")
    parser.add_argument('-act_fn', default="Tanh", type=str, help="activation func")

    # Run params
    parser.add_argument('-epochs', default=50000, type=int, help="number of epochs")
    parser.add_argument('-mode', default="single", type=str, help="run mode: single or sweep")
    args = parser.parse_args()
    return args


def create_config(args):
    config = {
        "batch_size": args.batch,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "n_layer": args.n_layer,
        "optim": args.optim,
        "decay": args.decay,
        "act_fn": args.act_fn
    }
    return config


def train(model, data_loader, optimizer, mse_loss, device, criteria):
    model.train()
    train_loss = 0.
    error = 0.
    for i, (input, target_force, target_torque) in enumerate(data_loader):
        feature_tensor = input.to(device)
        target_force = target_force.to(device)
        target_torque = target_torque.to(device)
        optimizer.zero_grad()
        force_prediction, torque_prediction = model(feature_tensor)
        force_loss = mse_loss(force_prediction, target_force)
        torque_loss = mse_loss(torque_prediction, target_torque)
        loss = force_loss + torque_loss
        force_error = criteria(force_prediction, target_force).item()
        torque_error = criteria(torque_prediction, target_torque).item()
        error += force_error + torque_error
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss = train_loss / len(data_loader)

    return train_loss, error / len(data_loader)


def validation(model, data_loader, mse_loss, device, criteria):
    model.eval()
    with torch.no_grad():
        valid_loss = 0.
        valid_force_error = 0.
        valid_torque_error = 0.
        error = 0.
        for i, (input, target_force, target_torque) in enumerate(data_loader):
            feature_tensor = input.to(device)
            target_force = target_force.to(device)
            target_torque = target_torque.to(device)

            force_prediction, torque_prediction = model(feature_tensor)
            force_loss = mse_loss(force_prediction, target_force)
            torque_loss = mse_loss(torque_prediction, target_torque)
            loss = force_loss + torque_loss
            valid_loss += loss.item()
            force_error = criteria(force_prediction, target_force).item()
            torque_error = criteria(torque_prediction, target_torque).item()
            error += force_error + torque_error

        return valid_loss / len(data_loader), error / len(data_loader)


def run(config, model_path):
    # Load datasets
    train_dataloader, valid_dataloader, test_dataloader = load_datasets(args.data_path, config["batch_size"],
                                                                        inp_mode=config["inp_mode"])
    print('Dataset size: \n\t train: {}, \n\t valid: {}, \n\t test:{}'.
          format(len(train_dataloader), len(valid_dataloader), len(test_dataloader)))
    in_dim = train_dataloader.dataset.in_dim
    wandb.run.summary["input_dim"] = in_dim

    # build model
    model = NN(in_dim=in_dim, hidden_dim=config["hidden_dim"], out_dim=constants.OUT_DIM,
               n_layers=config["n_layer"], act_fn=config["act_fn"], mode=config["inp_mode"])
    print(model)
    model.to(device)

    loss = nn.L1Loss().to(device)
    criteria = nn.L1Loss().to(device)
    if config["optim"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["decay"])
    if config["optim"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=50, min_lr=0.01)

    wandb.watch(models=model, criterion=loss, log="all")

    # Train
    print('**************************Training*******************************')
    best_val_error = None
    best_model_path = None
    epoch_train_losses = []

    for epoch in range(args.epochs):

        train_loss, train_error = train(model, train_dataloader, optimizer, loss, device, criteria)
        epoch_train_losses.append(train_loss)
        valid_loss, val_error = validation(model, valid_dataloader, loss, device, criteria)
        # scheduler.step(val_error)
        if epoch % 20 == 0:
            print('epoch {}/{}: \n\t train_loss: {}, \n\t val_loss: {}, \n\t train_error: {}, \n\t val_error: {}'.
                  format(epoch + 1, args.epochs, train_loss, valid_loss, train_error, val_error))
        wandb.log({'train_loss': train_loss,
                   'train_error': train_error,
                   'valid loss': valid_loss,
                   'valid error': val_error,
                   "learning rate": optimizer.param_groups[0]['lr']})

        if best_val_error is None:
            best_val_error = val_error

        if val_error <= best_val_error:
            best_val_error = val_error
            best_model_path = os.path.join(model_path, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            # wandb.save(best_model_path)
            print('#################################################################')
            print('best_val_error: {}, best_epoch: {}'.format(best_val_error, epoch))
            print('#################################################################')
            wandb.run.summary["best_epoch"] = epoch + 1
            wandb.run.summary["best_val_error"] = best_val_error

    # Testing
    print('**************************Testing*******************************')
    if best_model_path:
        model = NN(in_dim=in_dim, hidden_dim=config["hidden_dim"], out_dim=constants.OUT_DIM,
                   n_layers=config["n_layer"], act_fn=config["act_fn"], mode=config["inp_mode"])
        model.to(device)
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_loss, test_error = validation(model, test_dataloader, loss, device, criteria)
    print('Testing \n\t test error: {}'.
          format(test_error))

    wandb.run.summary['test error'] = test_error


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
        run(config, model_path)


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
    config["log_path"] = log_path
    config["dataset"] = args.data_path
    config["inp_mode"] = args.inp_mode
    print(config)

    if args.mode == "single":
        wandb.init(project=args.project, notes=args.notes, group=args.group,
                   tags=["append_input"], dir=log_path,
                   config=config)
        run(config, model_path)
    else:
        run_sweep(args.project)
