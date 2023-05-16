import torch.nn as nn


class NN(nn.Module):
    def __init__(self, in_dim,  hidden_dim, out_dim, n_layers, act_fn="ReLU",
                 mode="append", batch_dim=None, dropout=0.3):
        super(NN, self).__init__()
        self.in_dim = in_dim
        self.batch_dim = batch_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.act_fn = act_fn
        self.mode = mode
        self.dropout = dropout

        self.net = nn.Sequential(*self._get_net(out_dim=self.out_dim))

    def _get_act_fn(self):
        act = getattr(nn, self.act_fn)
        return act()

    def _get_net(self, out_dim):
        layers = [nn.Linear(self.in_dim, self.hidden_dim), self._get_act_fn()]
        for i in range(self.n_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            # if self.batch_dim:
            #     layers.append(nn.BatchNorm1d(self.batch_dim))
            layers.append(self._get_act_fn())
            layers.append(nn.Dropout(p=self.dropout))
        layers.append(nn.Linear(self.hidden_dim, out_dim))
        return layers

    def forward(self, x):
        out = self.net(x)

        if self.mode == "stack":
            out = out.mean(dim=-2)

        # sum over all neighbor particle contributions for each particle
        out = out.sum(dim=-2)
        return out


class NNGrow(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers, act_fn="ReLU",
                 mode="append", batch_dim=None, dropout=0.5):
        super(NNGrow, self).__init__()
        self.in_dim = in_dim
        self.batch_dim = batch_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.act_fn = act_fn
        self.mode = mode
        self.dropout = dropout

        self.net = nn.Sequential(*self._get_net(out_dim=self.out_dim))

    def _get_act_fn(self):
        act = getattr(nn, self.act_fn)
        return act()

    def _get_net(self, out_dim):
        layers = [nn.Linear(self.in_dim, self.hidden_dim[0]), self._get_act_fn()]
        for i in range(1, len(self.hidden_dim)):
            layers.append(nn.Linear(self.hidden_dim[i-1], self.hidden_dim[i]))
            # if self.batch_dim:
            #     layers.append(nn.BatchNorm1d(self.batch_dim))
            layers.append(self._get_act_fn())
            # layers.append(nn.Dropout(p=self.dropout))
        layers.append(nn.Linear(self.hidden_dim[-1], out_dim))
        return layers

    def forward(self, x):
        out = self.net(x)

        if self.mode == "stack":
            out = out.mean(dim=-2)

        # sum over all neighbor particle contributions for each particle
        out = out.sum(dim=-2)
        return out
