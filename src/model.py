"""MLP architecture matching Eqs. (1)-(2) of Bahubalindruni et al. (2015):

    yh = tanh(x . wh + bh)      hidden layer, tanh activation
    y  = yh . wo + bo           output layer, linear activation
"""
import torch
import torch.nn as nn


class TFTNet(nn.Module):
    def __init__(self, n_inputs: int = 4, n_hidden: int = 22, n_outputs: int = 1):
        super().__init__()
        self.hidden = nn.Linear(n_inputs, n_hidden)   # wh, bh
        self.tanh = nn.Tanh()                         # Sig(.)
        self.output = nn.Linear(n_hidden, n_outputs)  # wo, bo

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        yh = self.tanh(self.hidden(x))
        y = self.output(yh)
        return y


class TFTNet2(nn.Module):
    """Two-hidden-layer MLP (tanh hidden layers, linear output), a deeper
    variant of TFTNet used for the parasitic-capacitance ANNs:

        h1 = tanh(x  . w1 + b1)
        h2 = tanh(h1 . w2 + b2)
        y  = h2 . wo + bo        (linear output)
    """

    def __init__(self, n_inputs: int = 4, n_hidden1: int = 10, n_hidden2: int = 10,
                 n_outputs: int = 1):
        super().__init__()
        self.hidden1 = nn.Linear(n_inputs, n_hidden1)    # w1, b1
        self.hidden2 = nn.Linear(n_hidden1, n_hidden2)   # w2, b2
        self.tanh = nn.Tanh()
        self.output = nn.Linear(n_hidden2, n_outputs)    # wo, bo

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.tanh(self.hidden1(x))
        h2 = self.tanh(self.hidden2(h1))
        y = self.output(h2)
        return y
