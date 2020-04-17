"""Custom layers for graph convolutional neural networks."""
import math

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
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        """Reset params."""
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        """Pass forward features `v` and sparse adjacency matrix `adj`."""
        v, adj = input
        support = torch.matmul(v, self.weight)
        s_shape = support.shape
        in_batch = len(s_shape) > 2
        if in_batch and adj.is_sparse:
            support = torch.cat([matrix for matrix in support])
        if adj.is_sparse:
            output = torch.spmm(adj, support)
        else:
            output = torch.matmul(adj, support)
        output = output.reshape(s_shape)
        if self.bias is not None:
            output = output + self.bias
        return output, adj

    def __repr__(self):
        """Stringify as typical torch layer."""
        return (
            f"{self.__class__.__name__} " f"({self.in_features} -> {self.out_features})"
        )


class NormalizationLayer(nn.Module):
    """Normalization layer for the adjacency matrix.

    Adaped from https://github.com/CrivelliLab/Protein-Structure-DL/
    """

    def __init__(self, in_features, bias=False, D=100.0, apply_mask=None):
        """Initialize layer."""
        super(NormalizationLayer, self).__init__()
        self.in_features = in_features
        self.weight_feat = 1
        # Define trainable parameters
        self.weight1 = nn.Linear(in_features, 1, bias)
        self.weight2 = nn.Linear(in_features, 1, bias)
        self.in_feat = in_features
        self.d = D

    def forward(self, input):
        """Normalize sparse adjacency matrix `adj` in terms of `v`."""
        v, adj = input
        c1 = self.weight1(v)
        c2 = self.weight2(v)
        if len(c2.shape) > 2:
            c = c2.permute(0, 2, 1) + c1
        #     c = (
        #         torch.sigmoid(c1.bmm(c2.permute(0, 2, 1))) * self.d
        #     ) + 0.00001  # As to not divide by zero
        else:
            c = c2.T + c1
        #     c = (
        #         torch.sigmoid(c1.mm(c2.T)) * self.d
        #     ) + 0.00001  # As to not divide by zero
        c = 1 / (2 * c * c + 0.00001)

        norm_adj = torch.exp(-((adj * adj) * c))
        return v, norm_adj

    def __repr__(self):
        """Stringify as typical torch layer."""
        return (
            f"{self.__class__.__name__} " f"({self.in_features} -> {self.weight_feat})"
        )
