"""Test GCN models."""

import torch

from gcn_prot.models import GCN_simple, GCN_normed
from gcn_prot.models.utils import sparsize


def test_simple_init():
    """Test that layers are properly built."""
    nnet = GCN_simple(10, [20, 20, 20, 20], 2, 100, dropout=0)
    assert len(nnet.hidden_layers) == 4


def test_sparse(adj_batch):
    """Test sparse matrix reconstruction."""
    batch = len(adj_batch)
    nodes = len(adj_batch[0])
    sp_dim = batch * nodes
    adj = sparsize(adj_batch)
    assert adj.shape == torch.Size([sp_dim, sp_dim])


def test_simple_forward_online():
    """Test forward pass with one instance."""
    nnet = GCN_simple(3, [20, 20, 20, 20], 2, 3, dropout=0)
    # 3 aminoacids with 3 featuresz
    v = torch.FloatTensor([[23.0, 0.0, 2.0], [4.0, 2.0, 0.0], [0.0, 2.0, 0.0]])
    adj = torch.FloatTensor([[0, 1, 2], [1, 2, 0], [2, 1, 0]]).to_sparse()
    out = nnet.forward([v, adj])
    # 1 instance + 2 classes
    assert out.shape == torch.Size([2])


def test_simple_forward_batch(adj_batch):
    """Test forward pass with different stacking method."""
    nnet = GCN_simple(3, [20, 20, 20, 20], 2, 2, dropout=0)
    # 2 proteins with 2 aminoacids and 3 features each
    v = torch.FloatTensor(
        [[[23.0, 0.0, 2.0], [4.0, 2.0, 0.0]], [[1.0, 1.0, 24.0], [2.0, 1.0, 0.0]],]
    )
    out = nnet.forward([v, adj_batch])
    # 2 instances + 2 classes
    assert out.shape == torch.Size([2, 2])


def test_normalized_forward_batch(adj_batch):
    """Test forward pass with different stacking method."""
    nnet = GCN_normed(3, [20, 20, 20, 20], 4, 2, 2, dropout=0)
    # 2 proteins with 2 aminoacids and 3 features each
    v = torch.FloatTensor(
        [[[23.0, 0.0, 2.0], [4.0, 2.0, 0.0]], [[1.0, 1.0, 24.0], [2.0, 1.0, 0.0]],]
    )
    out = nnet.forward([v, adj_batch])
    # 2 instances + 2 classes
    assert out.shape == torch.Size([2, 2])
