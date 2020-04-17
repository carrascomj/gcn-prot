"""Test functions of the training loop."""
import torch

from gcn_prot.data import get_datasets
from gcn_prot.models import fit_network, forward_step


def test_forward_step(batch, nn_kras):
    """Test the forward pass."""
    forward_step(batch, nn_kras, False)


def test_fit_epoch(data_path, nn_kras):
    """Fit one epoch of train + test."""
    train, test, _ = get_datasets(
        data_path=data_path,
        nb_nodes=7,
        task_type="classification",
        nb_classes=2,
        split=None,
        k_fold=None,
        seed=1234,
    )
    optimizer = torch.optim.Adam(nn_kras.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    fit_network(
        nn_kras, train, test, optimizer, criterion, 20, epochs=1, plot_every=2000,
    )
