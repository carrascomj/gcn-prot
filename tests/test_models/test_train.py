"""Test functions of the training loop."""
from copy import deepcopy

import torch

from gcn_prot.data import get_datasets
from gcn_prot.models import fit_network, forward_step


def test_forward_step(batch, nn_kras):
    """Test the forward pass."""
    forward_step(batch, nn_kras)


# def test_fit_epoch(data_path, nn_kras):
#     """Fit one epoch of train + test."""
#     train, test, _ = get_datasets(
#         data_path=data_path,
#         nb_nodes=185,
#         task_type="classification",
#         nb_classes=2,
#         split=None,
#         k_fold=None,
#         seed=1234,
#     )
#     net_params = nn_kras.parameters()
#     optimizer = torch.optim.Adam(net_params)
#     criterion = torch.nn.CrossEntropyLoss()
#     old_params = deepcopy(net_params[0])
#     fit_network(nn_kras, train, test, optimizer, criterion, 20, epochs=1)
#     assert old_params != nn_kras.parameters()[0]
