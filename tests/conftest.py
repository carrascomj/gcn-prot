"""Instantiate fixtures."""

from os.path import dirname, join, pardir

import pytest


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
