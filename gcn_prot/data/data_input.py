# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 09:40:35 2020

@author: Bjorn
"""
import torch as t
import torch.nn.functional as f


def VCAInput(n_nodes, n_coord, n_feat, mask=None):
   
   '''
    Method returns Pytorch placeholders for graph convolutional networks.
    Params:
        n_nodes - int; number of nodes in graphs
        n_coord - int; number of spatial dimensions for points
        n_feat - int; number of features per node
        mask - np.array(); mask to apply on adjacency matrix
    Returns:
        v - Rank 3 tensor defining the node features; BATCHxNxF
        c - Rank 3 tensor defining coordinates of nodes in n-euclidean space; BATCHxNxC
        a - Rank 3 tensor defining L2 distances between nodes according to tensor c; BATCHxNxN
        m - mask placeholder
   '''
   v = t.Tensor(n_nodes, n_feat) 
   #c = tf.placeholder(tf.float32, [None, nb_nodes, nb_coords])
   # No placeholder exists in torch
   c = t.Tensor(0, n_nodes, n_coord) 
   m = t.Tensor(0, n_nodes, n_nodes)
   
   a_euc = euclidean_dist(c)
   a_cos = cos_dist(c, m)
   a = [a_euc, a_cos]
   
   return v, c, a, m
   
def euclidean_dist(c, namespace="euclidean_dist_a"):

   '''   
   Parameters
   ----------
   c : Rank 3 tensor defining coordinates of nodes in n-euclidean space
      DESCRIPTION.
   namespace : TYPE, optional
      DESCRIPTION. The default is "euclidean_dist_a".

   Returns
   -------
   a - Rank 3 tensor defining pairwise adjacency matrix of nodes.
   '''
   l2 = t.sum(c*c, dim=1)
   l2 = t.reshape(l2, [-1, 1, l2.shape[-1]])
   #a = l2 - 2*t.matmul(c, t.transpose(c, 0,2,1)) + t.transpose(l2, 0,2,1)
   # Results in c=tensor([], size=(0, 29, 3)) and l2 = 
   a = l2 - 2*t.matmul(c, c.permute(0,2,1) + l2.permute(0,2,1))
   a = t.abs(a, name=namespace)

   return a

def cos_dist(c, mask=None, namespace="cost_dist_a"):
  
   '''
   Parameters
   ----------
   c : Rank 3 tensor defining coordinates of nodes in n-euclidean space.
   mask : The default is None.

   Returns
   -------
   a : rank 3 tensor defining cosine pairwise adjacency matrix of nodes.
   '''
   normalized = f.normalize(c, dim=-1,p=2) # p=2 mean l2 normalization, dim=0 is for axis
   #normalized = tf.nn.l2_normalize(c, axis=-1)
   prod = t.matmul(normalized, normalized, adjoint_b=True)
   a = (1 - prod) / 2.0
   if mask is not None:
       a = 1 - a
       a = a*mask # Do elementwise multiplication
       #a = tf.multiply(a, mask, name=namespace)
   else:
      a = t.add(-a, 1, name=namespace)
      #a = tf.add(-a, 1, name=namespace)

   return a
   
  
#%%
# input_shape for the kinase data set: [344,29,3]
# V, C, A, M = VCAInput(input_shape[0], input_shape[2], input_shape[1])
# VCAInput(nb_nodes, nb_coords, nb_features, a_mask=None)
input_shape = [344,29,3]

V, C, A, M = VCAInput(input_shape[0], input_shape[2], input_shape[1])

