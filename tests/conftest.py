"""Instantiate fixtures."""

from os.path import dirname, join, pardir

import pytest
import torch

from gcn_prot.data import get_datasets
from gcn_prot.models import GCN_simple


@pytest.fixture
def data_path(scope="session"):
    """Path to KrasHras experiment data."""
    return join(dirname(__file__), pardir, "new_data")


@pytest.fixture
def pdb_path(scope="session"):
    """Path to KrasHras experiment data."""
    return join(dirname(__file__), pardir, "new_data", "pdb")


@pytest.fixture
def graph_path(scope="session"):
    """Path to KrasHras experiment data."""
    return join(dirname(__file__), pardir, "new_data", "graph")


@pytest.fixture(scope="function")
def adj_batch():
    """Define stacked adjacency matrix for 2 proteins."""
    return torch.Tensor([[[1, 3], [3, 1]], [[7, 8], [8, 7]]])


@pytest.fixture(scope="function")
def batch(data_path):
    """Define a batch for the KrasHras dataset."""
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
        return batch


@pytest.fixture(scope="module")
def nn_kras():
    """Define simple GCN for the KrasHras dataset."""
    return GCN_simple(29, [3], 2, 185, dropout=0)
