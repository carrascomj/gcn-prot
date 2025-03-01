"""Custom matrix operations."""
import torch


def sparsize(adj, cuda=False, sparsed=True):
    """Transform R^BxNxN `adj` tensor to sparse matrix.

    N is the number of nodes, B is the batch size.

    Parameters
    ----------
    adj: torch.Tensor (BxNxN)
        B stacked square matrices (NxN)
    cuda: bool
        if working with gpu. Default: False

    Returns
    -------
    torch.Tensor (S*S)
        where S is B*N

    """
    if len(adj.shape) < 3:
        return adj
    batch = adj.shape[0]
    if cuda:
        out = torch.cat(
            [
                torch.cat(
                    [
                        torch.zeros(adj[i].shape).cuda() if j != i else adj[i]
                        for j in range(batch)
                    ],
                    axis=1,
                ).cuda()
                for i in range(batch)
            ]
        )
    else:
        out = torch.cat(
            [
                torch.cat(
                    [
                        torch.zeros(adj[i].shape) if j != i else adj[i]
                        for j in range(batch)
                    ],
                    axis=1,
                )
                for i in range(batch)
            ]
        )
    if sparsed:
        out = out.to_sparse()
    return out


def calc_accuracy(out, true):
    """Compute total accuracy of output of NN (CrossEntropyLoss-like)."""
    prediction = out.min(axis=1).indices.flatten()
    return (prediction == true.flatten()).sum().item() / len(prediction)
