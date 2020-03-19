"""Custom layers for graph convolutional neural networks."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """Simple GCN layer.

    Adaped from https://github.com/CrivelliLab/Protein-Structure-DL/
    """

    def __init__(
        self, in_features, out_features, dropout=0.1, bias=False, act=F.relu,
    ):
        """Initialize layer."""
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.linear = nn.Linear(in_features, out_features, bias)
        self.act = act

    def forward(self, input):
        """Pass forward features `v` and sparse adjacency matrix `adj`."""
        v, adj = input
        v_shape = v.shape
        if len(v_shape) > 2:
            # stacked matrices (tensor) to concatenated matrices
            v = torch.cat([matrix for matrix in v])
        # apply each adj_i to each aminoacid_i
        support = torch.spmm(adj, v)
        # return to tensor
        support = support.reshape(v_shape)
        output = self.act(self.linear(support))
        return F.dropout(output, training=self.training), adj

    def __repr__(self):
        """Stringify as typical torch layer."""
        return (
            f"{self.__class__.__name__} " f"({self.in_features} -> {self.out_features})"
        )
