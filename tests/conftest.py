"""Instantiate fixtures."""

from os.path import dirname, join, pardir

import pytest
import torch


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
