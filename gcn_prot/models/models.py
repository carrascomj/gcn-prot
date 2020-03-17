"""Custom layers for graph convolutional neural networks."""
import torch.nn as nn
import torch.nn.functional as F

from .layers import GraphConvolution


class GCN_simple(nn.Module):
    """Simplest GCN model."""

    def __init__(
        self, input, hidden, label, dropout=0.1, bias=False, act=F.relu
    ):
        """Initialize GCN model.

        Parameters
        ----------
        input: int
            dimension of inputs
        hidden: Iterable[int]
            a vector of dimensions of the hidden graph convolutional layers
        label: int
            dimension of output
        dropout: float (0.1)
        bias: bool (False)
        act: function
            activation function. Default: F.relu

        """
        super(GCN_simple, self).__init__()
        hidden = [hidden] if isinstance("hidden", int) else hidden
        gc_layers = [
            GraphConvolution(in_dim, out_dim, dropout, bias, act)
            for in_dim, out_dim in zip([input] + hidden, hidden + [label])
        ]
        self.layers = nn.Sequential(*gc_layers)

    def forward(self, input):
        """Pass forward GCN model.

        Parameters
        ----------
        input:
            v: torch.Tensor
                3D Tensor containing the features of nodes
            adj: torch.Tensor
                3D tensor with the values of the adjacency matrix

        """
        x, _ = self.layers.forward(input)
        return F.log_softmax(x, dim=1)
