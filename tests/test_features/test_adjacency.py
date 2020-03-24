"""Test adjacency matrix operations."""

import torch

from gcn_prot.features import euclidean_dist


def test_euclidean():
    """Test euclidean distance."""
    adj = torch.FloatTensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    res = euclidean_dist(adj)
    return (res == torch.FloatTensor([[0, 1, 2], [1, 0, 1], [2, 1, 0]])).all()


if __name__ == "__main__":
    test_euclidean()
