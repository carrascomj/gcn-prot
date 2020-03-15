"""Test dataset building and splitting."""

import pytest

from gcn_prot.data import get_datasets, get_longest


def test_longest(graph_path):
    """Test get longest length in directory."""
    assert 189 == get_longest(graph_path)


def test_get_dataset(data_path):
    """Test splitting with default size of datasets."""
    train, test, valid = get_datasets(
        data_path=data_path,
        nb_nodes=185,
        task_type="classification",
        nb_classes=2,
        split=None,
        k_fold=None,
        seed=1234,
    )
    assert 164 == len(train)
    assert 24 == len(test)
    assert 48 == len(valid)


def test_indexing(data_path):
    """Test random access the generated graph dataset."""
    train, _, _ = get_datasets(
        data_path=data_path,
        nb_nodes=185,
        task_type="classification",
        nb_classes=2,
        split=None,
        k_fold=None,
        seed=1234,
    )
    prot = train[0]
    prot_dims = [len(tr) for tr in prot]
    # v, c, m, y
    assert prot_dims == [185, 185, 185, 2]


@pytest.mark.skip(reason="have to figure out how it works.")
def test_kfold(data_path):
    """Test kfold splitting."""
    train, test, valid = get_datasets(
        data_path=data_path,
        nb_nodes=185,
        task_type="classification",
        nb_classes=2,
        split=None,
        k_fold=[7, 1, 2],
        seed=1234,
    )
    assert 164 == len(train)
    assert 24 == len(test)
    assert 48 == len(valid)
