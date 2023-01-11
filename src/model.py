import torch.nn as nn


class NN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers, act_fn="ReLU"):
        super(NN, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.act_fn = act_fn

        self.net = nn.Sequential(*self._get_net())

    def _get_act_fn(self):
        act = getattr(nn, self.act_fn)
        return act()

    def _get_net(self):
        layers = [nn.Linear(self.in_dim, self.hidden_dim), self._get_act_fn()]
        for i in range(self.n_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.Dropout(p=0.25))
            layers.append(self._get_act_fn())
        layers.append(nn.Linear(self.hidden_dim, self.out_dim))
        return layers

    def forward(self, x):
        return self.net(x)