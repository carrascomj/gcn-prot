# -*- coding: utf-8 -*-
"""
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
