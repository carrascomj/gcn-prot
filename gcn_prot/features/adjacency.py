# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 13:34:38 2020

@author: Bjorn
"""

import torch as t
import numpy as np

def euclidean_dist(c, namespace="euclidean_dist_a"):

   '''   
   Parameters
   ----------
   c : Rank 3 array defining coordinates of nodes in n-euclidean space


   Returns
   -------
   a - Rank 3 tensor defining pairwise adjacency matrix of nodes.
   '''
   
   adj_mat = t.ones(())
   adj_mat = adj_mat.new_empty((len(c), len(c)))
   for i in range(0, len(c)):
      node1 = c[i]
      node1 = t.from_numpy(node1)
      for j in range(0, len(c)):
         node2 = c[j]
         node2 = t.from_numpy(node2)
         dist = t.dist(node1, node2)
         adj_mat[i,j] = dist
      
      return(adj_mat)


#%% 
   
# Get some example coordinates from the training data
      
c = train[0][1]

euclidean_dist(c)
print(adj_mat)
