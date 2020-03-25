"""Test adjacency matrix operations."""

import torch

from gcn_prot.data import get_datasets
from gcn_prot.features import euclidean_dist, transform_input


def test_euclidean():
    """Test euclidean distance."""
    adj = torch.FloatTensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    res = euclidean_dist(adj)
    return (res == torch.FloatTensor([[0, 1, 2], [1, 0, 1], [2, 1, 0]])).all()


def test_transform(data_path):
    """Test transformation of input."""
    train, _, _ = get_datasets(
        data_path=data_path,
        nb_nodes=185,
        task_type="classification",
        nb_classes=2,
        split=None,
        k_fold=None,
        seed=1234,
    )
    for batch in torch.utils.data.DataLoader(
        train, shuffle=False, batch_size=25, drop_last=False
    ):
        some_batch = batch
        break
    input, y = transform_input(some_batch)
    v, adj = input
    assert adj.shape == torch.Size([25, 185, 185])
