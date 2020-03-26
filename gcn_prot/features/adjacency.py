"""Adjancency matrix operations.

Created on Fri Mar 20 13:34:38 2020

@author: Bjorn

"""

import torch as t


def euclidean_dist(c, namespace="euclidean_dist_a"):
    """Calculate euclidean distance.

    Parameters
    ----------
    c : Rank 3 array defining coordinates of nodes in n-euclidean space

    Returns
    -------
    a - Rank 3 tensor defining pairwise adjacency matrix of nodes.

    """
    adj_mat = t.FloatTensor(len(c), len(c))
    for i in range(0, len(c)):
        node1 = c[i]
        for j in range(0, len(c)):
            node2 = c[j]
            dist = t.dist(node1, node2)
            adj_mat[i, j] = dist

        return adj_mat


def transform_input(input, training=True):
    """Get adjancecy matrix from the inputs and apply mask."""
    v, c, m, y = input
    adj = [euclidean_dist(c[prot]) * m[prot] for prot in range(c.shape[0])]
    v = t.nn.Parameter(v.float(), requires_grad=training)
    y = t.stack(y).T
    y = t.LongTensor([t.where(label == 1)[0][0] for label in y])
    return [v, t.stack(adj).float()], y
