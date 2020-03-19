"""Custom layers for graph convolutional neural networks."""
import torch
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
            for in_dim, out_dim in zip([input] + hidden[:-1], hidden)
        ]
        self.hidden_layers = nn.Sequential(*gc_layers)
        self.out_layer = nn.Sequential(nn.Linear(hidden[-1], label))

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
        v, adj = input
        input = [v, sparsize(adj)]
        x, _ = self.hidden_layers.forward(input)
        x = x.sum(axis=-2)
        return self.out_layer(x)


def sparsize(adj):
    """Transform R^BxNxN `adj` tensor to sparse matrix.

    N is the number of nodes, B is the batch size.

    Parameters
    ----------
    adj: torch.Tensor (BxNxN)
        B stacked square matrices (NxN)

    Returns
    -------
    torch.Tensor (S*S)
        where S is B*N

    """
    if len(adj.shape) < 3:
        return adj
    batch = adj.shape[0]
    return torch.cat(
        [
            torch.cat(
                [
                    torch.zeros(adj[i].shape) if j != i else adj[i]
                    for j in range(batch)
                ],
                axis=1,
            )
            for i in range(batch)
        ]
    ).to_sparse()
