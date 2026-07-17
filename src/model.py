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
