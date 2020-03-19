# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 19:23:20 2020

@author: Bjorn
"""

class Node:
    def __init__(self, node):
        self.name = node
        self.neighbors = []
        
    def add_neighbor(self, neighbor):
        if isinstance(neighbor, Node):
           if neighbor.name not in self.neighbors:
                self.neighbors.append(neighbor.name)
                neighbor.neighbors.append(self.name)
                self.neighbors = sorted(self.neighbors)
                neighbor.neighbors = sorted(neighbor.neighbors)
        else:
            return False
        
    def add_neighbors(self, neighbors):
        for neighbor in neighbors:
            if isinstance(neighbor, Node):
                if neighbor.name not in self.neighbors:
                    self.neighbors.append(neighbor.name)
                    neighbor.neighbors.append(self.name)
                    self.neighbors = sorted(self.neighbors)
                    neighbor.neighbors = sorted(neighbor.neighbors)
            else:
                return False
        
    def __repr__(self):
        return str(self.neighbors)


class Graph:
    def __init__(self):
        self.nodes = {}
    
    def add_node(self, node):
        if isinstance(node, Node):
            self.node[node.name] = node.neighbors

            
    def add_nodes(self, nodes):
        for node in nodes:
            if isinstance(node, Node):
                self.nodes[node.name] = node.neighbors

            
    def add_edge(self, node_from, node_to):
        if isinstance(node_from, Node) and isinstance(node_to, Node):
            node_from.add_neighbor(node_to)
            if isinstance(node_from, Node) and isinstance(node_to, Node):
                self.nodes[node_from.name] = node_from.neighbors
                self.nodes[node_to.name] = node_to.neighbors
                
    def add_edges(self, edges):
        for edge in edges:
            self.add_edge(edge[0],edge[1])          
    
    def adjacencyList(self):
        if len(self.nodes) >= 1:
                return [str(key) + ":" + str(self.nodes[key]) for key in self.nodes.keys()]  
        else:
            return dict()
        
    def adjacencyMatrix(self):
        if len(self.nodes) >= 1:
            self.node_names = sorted(g.nodes.keys())
            self.node_indices = dict(zip(self.node_names, range(len(self.node_names)))) 
            import numpy as np
            self.adjacency_matrix = np.zeros(shape=(len(self.nodes),len(self.nodes)))
            # Calculate euclidean distance here
            for i in range(len(self.node_names)):
                for j in range(i, len(self.nodes)):
                    for el in g.nodes[self.node_names[i]]:
                        j = g.node_indices[el]
                        self.adjacency_matrix[i,j] = 1
            return self.adjacency_matrix
        else:
            return dict()              
                        

def graph(g):
    """ Function to print a graph as adjacency list and adjacency matrix. """
    return str(g.adjacencyList()) + '\n' + '\n' + str(g.adjacencyMatrix())
 
   
#%%
    
a = Node('A')
b = Node('B')
c = Node('C')
d = Node('D')
e = Node('E')

a.add_neighbors([b,c,e]) 
b.add_neighbors([a,c])
c.add_neighbors([b,d,a,e])
d.add_neighbor([c])
e.add_neighbors([a,c])

g = Graph()
g.add_nodes([a,b,c,d,e])
g.add_edge(b,d)

print(graph(g))