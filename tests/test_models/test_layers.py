"""Test layers of the GCN model where testing the model is not enough."""
import torch

from gcn_prot.models.layers import NormalizationLayer
from gcn_prot.models.utils import sparsize


def test_normalization():
    """Test adjancency normalization online."""
    v = torch.FloatTensor([[23.0, 0.0, 2.0], [4.0, 2.0, 0.0], [0.0, 2.0, 0.0]])
    adj = torch.FloatTensor([[0, 1, 2], [1, 2, 0], [2, 1, 0]]).to_sparse()
    norm_layer = NormalizationLayer(3, 2)
    _, out = norm_layer([v, adj])
    out = out.to_dense()
    assert (out <= 1).all() and (out >= 0).all()


def test_normalization_batch(adj_batch):
    """Test adjancency normalization in batch."""
    v = torch.FloatTensor(
        [[[23.0, 0.0, 2.0], [4.0, 2.0, 0.0]], [[1.0, 1.0, 24.0], [2.0, 1.0, 0.0]],]
    )
    adj_batch = sparsize(adj_batch)
    norm_layer = NormalizationLayer(3, 8)
    _, out = norm_layer([v, adj_batch])
    out = out.to_dense()
    assert (out <= 1).all() and (out >= 0).all()


if __name__ == "__main__":
    test_normalization_batch(torch.Tensor([[[1, 3], [3, 1]], [[7, 8], [8, 7]]]))
