import torch.nn as nn
import torch


class twin_nn(nn.Module):
    def __init__(self, n_hidden_layers, n_units, input_dim, differential=True):
        super(twin_nn, self).__init__()
        self.differential = differential
        self.n_hidden_layers = n_hidden_layers
        self.activation = nn.Softplus()
        # Initialize feed-forward graph
        net = []
        net.append(nn.Linear(input_dim, n_units))
        for i in range(1, n_hidden_layers):
            net.append(nn.Linear(n_units, n_units))
        net.append(nn.Linear(n_units, 1))
        self.net = nn.ModuleList(net)

    def feedforward_net(self, x):
        # Input layer to first hidden layer (no activation)
        z = self.net[0](x)
        # Hidden layers (and output)
        for i in range(1, self.n_hidden_layers+1):
            z = self.net[i](self.activation(z))
        return z

    def backprop(self, z, x):
        # Backpropogate through initial feed-forward network to obtain delta=dy/dx
        # i.e. derivative of output (price) wrt input (spot)
        xbar = torch.autograd.grad(z, x, grad_outputs=torch.ones_like(z), create_graph=True)
        return xbar[0]

    def forward(self, x):
        # Combine feed-forward and differentiation step
        x.requires_grad = True
        y = self.feedforward_net(x)
        if self.differential:
            xbar = self.backprop(y, x)
        else:
            xbar = None
        return y, xbar
