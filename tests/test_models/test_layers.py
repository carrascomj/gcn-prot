"""Test layers of the GCN model where testing the model is not enough."""
import torch

from gcn_prot.models.layers import NormalizationLayer


def test_normalization():
    """Test adjancency normalization online."""
    v = torch.FloatTensor([[23.0, 0.0, 2.0], [4.0, 2.0, 0.0], [0.0, 2.0, 0.0]])
    adj = torch.FloatTensor([[0, 1, 2], [1, 2, 0], [2, 1, 0]])
    norm_layer = NormalizationLayer(3, 2)
    _, out = norm_layer([v, adj])
    out = out
    assert (out <= 1).all() and (out >= 0).all()


def test_normalization_batch(adj_batch):
    """Test adjancency normalization in batch."""
    v = torch.FloatTensor(
        [[[23.0, 0.0, 2.0], [4.0, 2.0, 0.0]], [[1.0, 1.0, 24.0], [2.0, 1.0, 0.0]],]
    )
    norm_layer = NormalizationLayer(3, 8)
    _, out = norm_layer([v, adj_batch])
    out = out
    assert (out <= 1).all() and (out >= 0).all()
