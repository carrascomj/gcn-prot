"""Test dataset building and splitting."""

import pytest
import torch

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


def test_dataloading_batch(data_path):
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
    trainloader = torch.utils.data.DataLoader(
        train, shuffle=False, batch_size=25, drop_last=False
    )
    for batch in trainloader:
        batch_dims = [len(tr) for tr in batch]
        break

    v = batch[0]
    assert batch_dims == [25, 25, 25, 2]
    assert v.shape == torch.Size([25, 185, 29])


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
