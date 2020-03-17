"""Test GCN models."""

import torch

from gcn_prot.models import GCN_simple


def test_simple_init():
    """Test that layers are properly built."""
    nnet = GCN_simple(100, [20, 20, 20, 20], 2)
    assert len(nnet.layers) == 5


def test_simple_forward():
    """Test forward pass with different stacking method."""
    nnet = GCN_simple(3, [20, 20, 20, 20], 2)
    v = torch.FloatTensor([[1, 8, 3], [10, 2, 3], [1, 2, 3]])
    adj = torch.FloatTensor([[1, 0, 0], [0, 0, 3], [0, 0, 0]]).to_sparse()
    out = nnet.forward([v, adj])
    assert out.shape[1] == 2
