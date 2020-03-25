"""Test adjacency matrix operations."""

import torch

from gcn_prot.data import get_datasets
from gcn_prot.features import euclidean_dist, transform_input


def test_euclidean():
    """Test euclidean distance."""
    adj = torch.FloatTensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    res = euclidean_dist(adj)
    return (res == torch.FloatTensor([[0, 1, 2], [1, 0, 1], [2, 1, 0]])).all()


def test_transform(data_path, batch):
    """Test transformation of input."""
    input, y = transform_input(batch)
    v, adj = input
    assert adj.shape == torch.Size([25, 185, 185])
