# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 20:58:54 2018

@author: Anonymous for review
"""

import os
import numpy as np
import pandas as pd
import time
import random
import pickle
import math

import networkx as nx
from networkx.generators.classic import empty_graph, path_graph, complete_graph

import torch
from torch_geometric.data import Data
import scipy
from networkx.convert_matrix import to_scipy_sparse_matrix, to_numpy_array

from data_loader import node_labels_dic, graph_label_list, compute_adjency, graph_indicator
from graph import Graph

def to_sparse(x):
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def generate_triangles():
    i=0
    while i < 1000:
        G1 = nx.fast_gnp_random_graph(100,0.07)
        A = nx.adjacency_matrix(G1)
        triangles = (A@A@A).diagonal().sum()/6
        S1 = to_numpy_array(G1)
        T1 = torch.from_numpy(S1)
        TS1 = to_sparse(T1)
        connect = nx.edge_connectivity(G1)
        if connect > 0:
            data = Data(x=torch.ones((100,1)),edge_index=TS1._indices(),y=torch.ones(1).long()*triangles)
            torch.save(data,"../data/triangle/triangle_" + str(i) + ".pt")
            i += 1


def generate_four_cycles():
    i =0 
    while i < 1000:
        G1 = nx.fast_gnp_random_graph(100,0.07)
        A = nx.adjacency_matrix(G1)
        dense_A = np.array(nx.to_numpy_matrix(G1))        

        four_cycles = (A@A@A@A).diagonal().sum() 
        H1 =  dense_A.sum()
        H2 = 0
        
        for j in range(100):
            H2 = H2 + dense_A[j][j] * (dense_A[j][j] - 1) / 2
        
        four_cycles = (four_cycles - 4 * H2 - 2 * H1) / 8 
        
        S1 = to_numpy_array(G1)
        T1 = torch.from_numpy(S1)
        TS1 = to_sparse(T1)
        connect =  nx.edge_connectivity(G1)
        if connect > 0:
            data = Data(x=torch.ones((100,1)),edge_index=TS1._indices(),y=torch.ones(1).long()*four_cycles)
            torch.save(data,"../data/4_cycle/4_cycle_" + str(i) + ".pt")
            i += 1


def build_NCI1_dataset(path):
    node_dic = node_labels_dic(path,'NCI1_node_labels.txt')
    node_dic2 = {}
    for k,v in node_dic.items():
        node_dic2[k] = v-1
    node_dic = node_dic2
    graphs = graph_label_list(path,'NCI1_graph_labels.txt')
    adjency = compute_adjency(path,'NCI1_A.txt')
    data_dict = graph_indicator(path,'NCI1_graph_indicator.txt')
    data = []
    for i in graphs:
        g = Graph()
        for node in data_dict[i[0]]:
            g.name = i[0]
            g.add_vertex(node)
            g.add_one_attribute(node,node_dic[node])
            
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        
        D1 = nx.get_node_attributes(g.nx_graph,'attr_name')
        X1 = np.eye(40)[np.array(list(D1.values()))]
        S1 = to_numpy_array(g.nx_graph)
        T1 = torch.from_numpy(S1)
        TS1 = to_sparse(T1)
        nnode = g.nx_graph.number_of_nodes()
        data = Data(x=torch.from_numpy(X1).float(), edge_index=TS1._indices(), y=torch.ones(1).long() * (i[1]))
        data.num_nodes = nnode
        torch.save(data,"../data/nci1/nci1_" + str(i[0]) + ".pt")               
        print(i[0])

