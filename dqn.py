import torch.nn as nn
import torch.nn.functional as F


def resolve_activation(name_or_fn):
    if callable(name_or_fn):
        return name_or_fn
    mapping = {
        "ReLU": F.relu,
        "Sigmoid": F.sigmoid,
        "Softmax": lambda x: F.softmax(x, dim=1),
    }
    if name_or_fn not in mapping:
        raise ValueError(f"Unsupported activation: {name_or_fn}")
    return mapping[name_or_fn]


class DQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, layers, activation):
        super(DQNetwork, self).__init__()
        self.layers = nn.ModuleList()
        last_dim = input_dim

        for layer_size in layers:
            self.layers.append(nn.Linear(last_dim, layer_size))
            last_dim = layer_size

        self.output_layer = nn.Linear(last_dim, output_dim)
        self.activation = resolve_activation(activation)

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.output_layer(x)