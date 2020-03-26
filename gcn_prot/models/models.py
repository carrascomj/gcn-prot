"""Custom layers for graph convolutional neural networks."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import GraphConvolution


class GCN_simple(nn.Module):
    """Simplest GCN model."""

    def __init__(
        self,
        feats,
        hidden,
        label,
        nb_nodes,
        dropout,
        bias=False,
        act=F.relu,
        cuda=False,
    ):
        """Initialize GCN model.

        Parameters
        ----------
        feats: int
            dimension of inputs
        hidden: Iterable[int]
            a vector of dimensions of the hidden graph convolutional layers
        label: int
            dimension of output
        nb_nodes: int
            number of aminoacids. Used for last layer.
        dropout: float
        bias: bool (False)
        act: function
            activation function. Default: F.relu
        cuda: bool
            important to correctly sparsize

        """
        super(GCN_simple, self).__init__()
        hidden = [hidden] if isinstance("hidden", int) else hidden
        gc_layers = [
            GraphConvolution(in_dim, out_dim, dropout, bias, act)
            for in_dim, out_dim in zip([feats] + hidden[:-1], hidden)
        ]
        self.hidden_layers = nn.Sequential(*gc_layers)
        self.out_layer = nn.Linear(nb_nodes, label)
        self.in_cuda = cuda

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
        input = [v, sparsize(adj, self.in_cuda)]
        x, _ = self.hidden_layers.forward(input)
        x = x.sum(axis=-1)
        x = self.out_layer(x)
        return x


def sparsize(adj, cuda=False):
    """Transform R^BxNxN `adj` tensor to sparse matrix.

    N is the number of nodes, B is the batch size.

    Parameters
    ----------
    adj: torch.Tensor (BxNxN)
        B stacked square matrices (NxN)
    cuda: bool
        if working with gpu. Default: False

    Returns
    -------
    torch.Tensor (S*S)
        where S is B*N

    """
    if len(adj.shape) < 3:
        return adj
    batch = adj.shape[0]
    if cuda:
        out = torch.cat(
            [
                torch.cat(
                    [
                        torch.zeros(adj[i].shape).cuda() if j != i else adj[i]
                        for j in range(batch)
                    ],
                    axis=1,
                ).cuda()
                for i in range(batch)
            ]
        ).to_sparse()
    else:
        out = torch.cat(
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
    return out
