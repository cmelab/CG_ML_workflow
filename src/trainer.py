import torch
import torch.nn as nn
import wandb

from data_loader import load_datasets
from model import NN, NNGrow


class MLTrainer:
    def __init__(self, config, job_id):
        self.job_id = job_id
        self.project = config.project
        self.group = config.group
        self.notes = config.notes
        self.tags = config.tags

        # dataset parameters
        self.data_path = config.data_path
        self.inp_mode = config.inp_mode
        self.batch_size = config.batch_size
        self.batch_norm = config.batch_norm

        # model parameters
        self.model_type = config.model_type
        self.hidden_dim = config.hidden_dim
        self.n_layer = config.n_layer
        self.act_fn = config.act_fn
        self.dropout = config.dropout

        # optimizer parameters
        self.optim = config.optim
        self.lr = config.lr
        self.decay = config.decay

        # run parameters
        self.epochs = config.epochs

        # select device (GPU or CPU)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # create data loaders
        self.train_dataloader, self.valid_dataloader, self.test_dataloader = \
            load_datasets(config.data_path, config.batch_size, inp_mode=config.inp_mode)

        self.in_dim = self.train_dataloader.dataset.in_dim
        self.batch_dim = None
        if self.batch_norm:
            self.batch_dim = self.train_dataloader.dataset.batch_dim
        print('batch_dim: ', self.batch_dim)
        # create model
        self.model = self._create_model()

        # create loss, optimizer and scheduler
        self.loss = nn.L1Loss().to(self.device)
        self.criteria = nn.L1Loss().to(self.device)
        if self.optim == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                              weight_decay=self.decay)
        if self.optim == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.decay)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.9, patience=200, threshold=0.3,
                                                                     min_lr=0.001, cooldown=1000)

        self.wandb_config = self._create_config()

        self.wandb_run = wandb.init(project=self.project, notes=self.notes, group=self.group,
                   tags=self.tags, config=self.wandb_config)
        self.wandb_run_name = self.wandb_run.name
        self.wandb_run_path = self.wandb_run.path

    def _create_model(self):
        if self.model_type == "fixed":
            model = NN(in_dim=self.in_dim, hidden_dim=self.hidden_dim, out_dim=3,
                   n_layers=self.n_layer, act_fn=self.act_fn, mode=self.inp_mode, batch_dim=self.batch_dim)
        else:
            model = NNGrow(in_dim=self.in_dim, hidden_dim=self.hidden_dim, out_dim=3,
                   n_layers=self.n_layer, act_fn=self.act_fn, mode=self.inp_mode, batch_dim=self.batch_dim)
        model.to(self.device)

        return model

    def _create_config(self):
        config = {
            "batch_size": self.batch_size,
            "lr": self.lr,
            "hidden_dim": self.hidden_dim,
            "n_layer": self.n_layer,
            "optim": self.optim,
            "decay": self.decay,
            "act_fn": self.act_fn,
            "inp_mode": self.inp_mode,
            "batch_norm": self.batch_norm,
            "model_type": self.model_type
        }
        return config

    def _train(self):
        self.model.train()
        train_loss = 0.
        error = 0.
        for i, (input_feature, target_force, target_torque) in enumerate(self.train_dataloader):
            feature_tensor = input_feature.to(self.device)
            feature_tensor.requires_grad = True

            self.optimizer.zero_grad()
            prediction = self.model(feature_tensor)
            if self.group == "force":
                model_prediction = torch.autograd.grad(prediction, feature_tensor, retain_graph=True, create_graph=True,
                                                       grad_outputs=torch.ones_like(prediction))[0]
                if self.inp_mode == "stack":
                    model_prediction = model_prediction[:, :, 0, :3]
                else:
                    model_prediction = model_prediction[:, :, :3]
                model_prediction = model_prediction.sum(dim=[-2])
                target = target_force.to(self.device)
            else:
                model_prediction = prediction
                target = target_torque.to(self.device)

            _loss = self.loss(model_prediction, target)
            error += self.criteria(model_prediction, target).item()
            train_loss += _loss.item()
            _loss.backward()
            self.optimizer.step()
        train_loss = train_loss / len(self.train_dataloader)
        train_error = error / len(self.train_dataloader)

        return train_loss, train_error

    def _validation(self, data_loader, print_output=False):
        self.model.eval()
        # with torch.no_grad():
        error = 0.
        for i, (input_feature, target_force, target_torque) in enumerate(data_loader):
            feature_tensor = input_feature.to(self.device)
            feature_tensor.requires_grad = True
            prediction = self.model(feature_tensor)
            if self.group == "force":
                model_prediction = torch.autograd.grad(prediction, feature_tensor, retain_graph=True, create_graph=True,
                                                       grad_outputs=torch.ones_like(prediction))[0]
                if self.inp_mode == "stack":
                    model_prediction = model_prediction[:, :, 0, :3]
                else:
                    model_prediction = model_prediction[:, :, :3]
                model_prediction = model_prediction.sum(dim=[-2])
                target = target_force.to(self.device)
            else:
                model_prediction = prediction
                target = target_torque.to(self.device)
            error += self.criteria(model_prediction, target).item()
            if print_output and i % 10 == 0:
                print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
                print("prediction: ", model_prediction[0:5])
                print("target: ", target[0:5])
        return error / len(data_loader)

    def run(self):
        self.wandb_run.summary["job_id"] = self.job_id
        self.wandb_run.summary["data_path"] = self.data_path
        self.wandb_run.summary["input_shape"] = self.train_dataloader.dataset.input_shape

        wandb.watch(models=self.model, criterion=self.loss, log="all")
        print('**************************Training*******************************')
        self.best_val_error = None

        for epoch in range(self.epochs):

            train_loss, train_error = self._train()
            val_error = self._validation(self.valid_dataloader)
            self.scheduler.step(val_error)
            if epoch % 10 == 0:
                print('epoch {}/{}: \n\t train_loss: {}, \n\t train_error: {}, \n\t val_error: {}'.
                      format(epoch + 1, self.epochs, train_loss, train_error, val_error))
                
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        print(name, param.data)

            wandb.log({'train_loss': train_loss,
                       'train_error': train_error,
                       'valid error': val_error,
                       "learning rate": self.optimizer.param_groups[0]['lr']})

            if self.best_val_error is None:
                self.best_val_error = val_error

            if val_error <= self.best_val_error:
                self.best_val_error = val_error
                torch.save(self.model.state_dict(), "best_model.pth")
                print('#################################################################')
                print('best_val_error: {}, best_epoch: {}'.format(self.best_val_error, epoch))
                print('#################################################################')
                self.wandb_run.summary["best_epoch"] = epoch + 1
                self.wandb_run.summary["best_val_error"] = self.best_val_error

        # Testing
        print('**************************Testing*******************************')
        self.test_error = self._validation(self.test_dataloader, print_output=True)
        print('Testing \n\t test error: {}'.
              format(self.test_error))

        self.wandb_run.summary['test error'] = self.test_error
        wandb.finish()

