import torch.nn as nn


class NN(nn.Module):
    def __init__(self, pos_in_dim, orient_in_dim,  hidden_dim, energy_out_dim, torque_out_dim, n_layers, act_fn="ReLU",
                 mode="append"):
        super(NN, self).__init__()
        self.pos_in_dim = pos_in_dim
        self.orient_in_dim = orient_in_dim
        self.hidden_dim = hidden_dim
        self.energy_out_dim = energy_out_dim
        self.torque_out_dim = torque_out_dim
        self.n_layers = n_layers
        self.act_fn = act_fn
        self.mode = mode

        self.energy_net = nn.Sequential(*self._get_net(in_dim=self.pos_in_dim, out_dim=self.energy_out_dim))
        self.torque_net = nn.Sequential(*self._get_net(in_dim=self.orient_in_dim, out_dim=self.torque_out_dim))

    def _get_act_fn(self):
        act = getattr(nn, self.act_fn)
        return act()

    def _get_net(self, in_dim, out_dim):
        layers = [nn.Linear(in_dim, self.hidden_dim), self._get_act_fn()]
        for i in range(self.n_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.Dropout(p=0.5))
            layers.append(self._get_act_fn())
        layers.append(nn.Linear(self.hidden_dim, out_dim))
        return layers

    def forward(self, pos, orient):
        energy = self.energy_net(pos)
        torque = self.torque_net(orient)

        # sum over all neighbor particle contributions for each particle
        # energy = energy.sum(dim=-2)
        torque = torque.sum(dim=-2)
        return energy, torque
