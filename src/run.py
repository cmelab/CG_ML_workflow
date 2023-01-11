import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
import wandb

import constants
from data_loader import load_datasets, get_target_stats
from model import NN
from utils import syn_calculate_error


def get_parameters():
    parser = argparse.ArgumentParser()

    # input & output data params
    parser.add_argument('-data_path',
                        default='/home/marjanalbooyeh/code/cme-lab/ML_datasets/pps_synthesized_Jan10/processed/',
                        type=str, help="path to data")
    parser.add_argument('-log_dir', default='/home/marjanalbooyeh/logs/ML/', type=str, help="path to log data")

    # ML hyper params
    parser.add_argument('-lr', default=0.01, type=float, help="learning rate")
    parser.add_argument('-batch', default=128, type=int, help="batch size")
    parser.add_argument('-hidden_dim', default=128, type=int, help="number of hidden dims")
    parser.add_argument('-n_layer', default=7, type=int, help="number of layers")
    parser.add_argument('-optim', default="Adam", type=str, help="optimizer")

    # Run params
    parser.add_argument('-epochs', default=500, type=int, help="number of epochs")
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


def train(model, data_loader, optimizer, mse_loss, device, target_stats):
    model.train()
    train_loss = 0.
    train_force_error = 0.
    train_torque_error = 0.
    for i, (feature_tensor, target_tensor) in enumerate(data_loader):
        feature_tensor = feature_tensor.to(device)
        target_tensor = target_tensor.to(device)

        optimizer.zero_grad()
        prediction = model(feature_tensor)
        loss = mse_loss(prediction, target_tensor)
        train_loss += loss.item()
        force_error, torque_error = syn_calculate_error(prediction, target_tensor, target_stats, device)
        train_force_error += force_error
        train_torque_error += torque_error
        loss.backward()
        optimizer.step()

    train_loss = train_loss / len(data_loader)

    return train_loss, train_force_error/len(data_loader), train_torque_error/len(data_loader)


def validation(model, data_loader, mse_loss, device, target_stats):
    model.eval()
    with torch.no_grad():
        valid_loss = 0.
        valid_force_error = 0.
        valid_torque_error = 0.
        for i, (feature_tensor, target_tensor) in enumerate(data_loader):
            feature_tensor = feature_tensor.to(device)
            target_tensor = target_tensor.to(device)
            prediction = model(feature_tensor)
            #             print('prediction: ', prediction)
            loss = mse_loss(prediction, target_tensor)
            valid_loss += loss.item()
            force_error, torque_error = syn_calculate_error(prediction, target_tensor, target_stats, device)
            valid_force_error += force_error
            valid_torque_error += torque_error
        return valid_loss / len(data_loader), valid_force_error / len(data_loader), valid_torque_error / len(
            data_loader)


def run():

    args = get_parameters()
    print(args)
    config = create_config(args)
    print(config)

    # create log path
    now = datetime.now()
    log_path = os.path.join(args.log_dir, now.strftime("%Y-%m-%d %H:%M:%S"))
    print('Log path: ', log_path)
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    model_path = os.path.join(log_path, "model")
    if not os.path.exists(model_path):
        os.mkdir(model_path)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load datasets
    train_dataloader, valid_dataloader, test_dataloader = load_datasets(args.data_path, config["batch_size"])
    print('Dataset size: \n\t train: {}, \n\t valid: {}, \n\t test:{}'.
          format(len(train_dataloader), len(valid_dataloader), len(test_dataloader)))
    target_stats = get_target_stats(args.data_path)


    # build model
    model = NN(in_dim=constants.IN_DIM, hidden_dim=config["hidden_dim"], out_dim=constants.OUT_DIM, n_layers=config["n_layer"])
    model.to(device)


    mse_loss = nn.MSELoss().to(device)
    if config["optim"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    if config["optim"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=100, min_lr=0.00001)

    wandb.init(project="NN_synthesized_data_v2", group="single_run", tags=["NN", "synthesized"], dir=log_path, config=config)
    wandb.watch(models=model, criterion=mse_loss, log="all")

    # Train
    print('**************************Training*******************************')
    best_val_error = None
    best_model_path = None
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_val_errors = []
    for epoch in range(args.epochs):

        train_loss, train_force_error, train_torque_error = train(model, train_dataloader, optimizer, mse_loss, device, target_stats)
        epoch_train_losses.append(train_loss)
        valid_loss, val_force_error, val_torque_error = validation(model, valid_dataloader, mse_loss, device, target_stats)
        epoch_val_losses.append(valid_loss)
        val_error = val_force_error + val_torque_error
        epoch_val_errors.append(val_error)
        scheduler.step(val_error)

        if epoch % 20 == 0:
                print('epoch {}/{}: \n\t train_loss: {}, \n\t valid_loss: {}, \n\t val_error: {}, \n\t val_force_error: {}, \n \t val_torque_error: {}'.
                      format(epoch + 1, args.epochs, train_loss, valid_loss, val_error, val_force_error, val_torque_error))
        print(epoch)
        wandb.log({'train_loss': train_loss,
                   'train_force_error': train_force_error,
                   'train_torque_error': train_torque_error,
                   'valid loss': valid_loss,
                   'valid error': val_error,
                   'valid force error': val_force_error,
                   'valid torque error': val_torque_error})
        # wandb.log({"learning rate": optimizer.param_groups[0]['lr']})

        if best_val_error is None:
            best_val_error = val_error

        if val_error <= best_val_error:
            best_val_error = val_error
            best_model_path = os.path.join(model_path, 'best_model.chkpnt')
            torch.save(model.state_dict(), best_model_path)
            wandb.save(best_model_path)
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

    test_loss, test_force_error, test_torque_error = validation(model, test_dataloader, mse_loss, device, target_stats)
    test_error = test_force_error + test_torque_error
    print('Testing \n\t test error: {}, \n\t test_force_error: {}, \n\t test_torque_error: {}'.
          format(test_error, test_force_error, test_torque_error))

    wandb.log({'test error': test_error,
               'test force error': test_force_error,
               'test torque error': test_torque_error})


if __name__ == '__main__':
    run()