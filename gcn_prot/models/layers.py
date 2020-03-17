"""Custom layers for graph convolutional neural networks."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """Simple GCN layer.

    Adapted from https://github.com/HazyResearch/hgcn/
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

    def forward(self, v, adj):
        """Pass forward features `v` and adjacency matrix `adj`."""
        support = self.linear.forward(v)
        support = F.dropout(support, self.dropout, training=self.training)
        output = torch.spmm(adj, support)
        return self.act(output), adj

    def __repr__(self):
        """Stringify as typical torch layer."""
        return (
            f"{self.__class__.__name__} " f"({self.in_features} -> {self.out_features})"
        )
