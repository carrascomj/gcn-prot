"""Adjancency matrix operations.

Created on Fri Mar 20 13:34:38 2020

@author: Bjorn

"""

import torch as t


def euclidean_dist(c):
    """Calculate euclidean distance.

    Parameters
    ----------
    c : Rank 3 array defining coordinates of nodes in n-euclidean space

    Returns
    -------
    a - Rank 3 tensor defining pairwise adjacency matrix of nodes.

    """
    adj_mat = t.zeros(len(c), len(c)).float()
    for i in range(0, len(c)):
        node1 = c[i]
        for j in range(0, len(c)):
            node2 = c[j]
            dist = t.dist(node1, node2)
            adj_mat[i, j] = dist

    return adj_mat


def batched_eucl(coord):
    """Compute adjacency matrix for a batched coordinates Tensor `c`.

    https://github.com/pytorch/pytorch/issues/9406#issuecomment-472269698
    """
    B, M, N = coord.shape
    return t.pairwise_distance(
        coord[:, :, None].expand(B, M, M, N).reshape((-1, N)),
        coord[:, None].expand(B, M, M, N).reshape((-1, N)),
    ).reshape((B, M, M))


def binary_dist(c):
    """Calculate euclidean distance.

    Parameters
    ----------
    c : Rank 3 array defining coordinates of nodes in n-euclidean space

    Returns
    -------
    a - Rank 3 tensor defining pairwise adjacency matrix of nodes.

    """
    adj_mat = t.zeros(len(c), len(c)).float()
    for i in range(0, len(c)):
        for j in range(0, len(c)):
            dist = t.dist(c[i], c[j])
            adj_mat[i, j] = 1 if dist < 16 else 0

    return adj_mat


def transform_input(input_nn, training=True):
    """Get adjancecy matrix from the inputs and apply mask."""
    v, c, m, y = input_nn
    # adj = [euclidean_dist(c[prot]) * m[prot] for prot in range(c.shape[0])]
    v = t.autograd.Variable(v.float())
    y = t.stack(y).T
    y = t.autograd.Variable(t.LongTensor([t.where(label == 1)[0][0] for label in y]))
    return [v, c.float()], y
