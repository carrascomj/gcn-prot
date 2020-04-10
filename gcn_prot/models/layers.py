"""Custom layers for graph convolutional neural networks."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import sparsize


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
        output = F.dropout(output, training=self.training)
        return output, adj

    def __repr__(self):
        """Stringify as typical torch layer."""
        return (
            f"{self.__class__.__name__} "
            f"({self.in_features} -> {self.out_features})"
        )


class NormalizationLayer(nn.Module):
    """Normalization layer for the adjacency matrix.

    Adaped from https://github.com/CrivelliLab/Protein-Structure-DL/
    """

    def __init__(
        self,
        in_features,
        out_features,
        dropout=0.1,
        bias=False,
        act=F.relu,
        D=100.0,
        cuda=False,
    ):
        """Initialize layer."""
        super(NormalizationLayer, self).__init__()
        self.weight_feat = out_features
        # Define trainable parameters
        self.weight1 = nn.Linear(in_features, self.weight_feat, bias)
        self.weight2 = nn.Linear(in_features, self.weight_feat, bias)
        self.in_feat = in_features
        self.d = D
        self.in_cuda = cuda

    def forward(self, input):
        """Normalize sparse adjacency matrix `adj` in terms of `v`."""
        v, adj = input
        adj = adj.to_dense()
        c1 = self.weight1(v)
        c2 = self.weight2(v)
        c_shape = c2.shape
        if len(c_shape) > 2:
            c = (
                torch.sigmoid(
                    c1.bmm(c2.view(c_shape[0], c_shape[2], c_shape[1]))
                )
                * self.d
            ) + 0.00001  # As to not divide by zero
            c = 1 / (2 * c * c)
            c = sparsize(c, self.in_cuda).to_dense()
        else:
            c = (
                torch.sigmoid(c1.mm(c2.T)) * self.d
            ) + 0.00001  # As to not divide by zero

        norm_adj = torch.exp(-((adj * adj) * c))
        return v, norm_adj.to_sparse()
