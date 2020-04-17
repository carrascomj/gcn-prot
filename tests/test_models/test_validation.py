"""Test validation pass."""
import torch

from gcn_prot.data import get_datasets
from gcn_prot.models import Validation, fit_network


def test_validation(data_path, nn_kras):
    """Fit one epoch of train + test."""
    train, test, valid = get_datasets(
        data_path=data_path,
        nb_nodes=185,
        task_type="classification",
        nb_classes=2,
        split=None,
        k_fold=None,
        seed=1234,
    )
    optimizer = torch.optim.Adam(nn_kras.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    nn_kras = fit_network(
        nn_kras, train, test, optimizer, criterion, 20, epochs=1, plot_every=2000,
    )
    validator = Validation(nn_kras, valid)
    validator.validate()
    stats = validator.compute_stats()
    set(stats.keys()) == {
        "recall",
        "precision",
        "accuracy",
        "f_score",
    }
