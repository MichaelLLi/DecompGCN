# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 20:58:54 2018

@author: Michael
"""

import os
import numpy as np
import pandas as pd
import time
import random
import pickle
import networkx as nx
from networkx.generators.classic import empty_graph, path_graph, complete_graph
import math
import torch
from torch_geometric.data import Data
import scipy
import torch.nn.functional as F
from networkx.convert_matrix import to_scipy_sparse_matrix, to_numpy_array
from torch_geometric.nn import GCNConv
from torch_geometric.data import Dataset

from data_loader import node_labels_dic, graph_label_list, compute_adjency, graph_indicator
from graph_new import Graph

def to_sparse(x):
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())

# connectivity
def generate_connectivity():
    for m in range(1,3):
        i=0
        while i<500:
            G1=nx.fast_gnp_random_graph(100,0.07)
            S1=to_numpy_array(G1)
            T1=torch.from_numpy(S1)
            TS1=to_sparse(T1)
            connect=nx.edge_connectivity(G1)
            if connect==m:
                data=Data(x=torch.ones((100,1)),edge_index=TS1._indices(),y=torch.ones(1).long()*(connect-1))
                torch.save(data,"./.data/connected/connect_" + str(m) + "_" + str(i) + ".pt")
                i+=1

# clique
def generate_clique():
    for m in range(3,5):
        i=0
        while i<500:
            G1=nx.fast_gnp_random_graph(100,0.07)
            S1=to_numpy_array(G1)
            T1=torch.from_numpy(S1)
            TS1=to_sparse(T1)
            connect=nx.edge_connectivity(G1)
            clique=nx.graph_clique_number(G1)
            if connect>0 and clique==m:
                data=Data(x=torch.ones((100,1)),edge_index=TS1._indices(),y=torch.ones(1).long()*(m-3))
                torch.save(data,"./.data/clique/clique_" + str(m) + "_" + str(i) + ".pt")
                i+=1

# tree and cycle
def generate_tree_cycle():
    num_tree, num_cycle = 0, 0
    i = 0
    while num_tree < 500:
        if i % 1000 == 0:
            print(i, num_tree, num_cycle)
        G = nx.random_powerlaw_tree(100, seed=i, tries=10000)
        S = to_numpy_array(G)
        T = torch.from_numpy(S)
        TS = to_sparse(T)

        data=Data(x=torch.ones((100,1)),edge_index=TS._indices(),y=torch.zeros(1).long())
        torch.save(data,"../data/tree_cycle/tree_" + str(num_tree) + ".pt")
        num_tree += 1
        i += 1

    while num_cycle < 500:
        if i % 1000 == 0:
            print(i, num_tree, num_cycle)
        G = nx.fast_gnp_random_graph(100,0.07)
        S = to_numpy_array(G)
        T = torch.from_numpy(S)
        TS = to_sparse(T)
        connect = nx.edge_connectivity(G)

        tree = nx.is_forest(G)
        if connect > 0 and tree == False:
            data=Data(x=torch.ones((100,1)),edge_index=TS._indices(),y=torch.ones(1).long())
            torch.save(data,"../data/tree_cycle/cycle_" + str(num_cycle) + ".pt")
            num_cycle += 1
        i += 1

def generate_triangles():
    i=0
    while i<500:
        G1=nx.fast_gnp_random_graph(100,0.07)
        A = nx.adjacency_matrix(G1)
        triangles = (A@A@A).diagonal().sum()/6
        S1=to_numpy_array(G1)
        T1=torch.from_numpy(S1)
        TS1=to_sparse(T1)
        connect=nx.edge_connectivity(G1)
        if connect>0:
            data=Data(x=torch.ones((100,1)),edge_index=TS1._indices(),y=torch.ones(1).long()*triangles)
            torch.save(data,"../data/triangle/triangle_" + str(i) + ".pt")
            i+=1
            
def generate_planar():
    original_graphs=nx.read_graph6('../data/list_2040_graphs.g6')
    i=0
    for graph in original_graphs:
        if nx.edge_connectivity(graph)>0 and graph.number_of_nodes()>30 and graph.number_of_nodes()<100 and graph.number_of_edges()>graph.number_of_nodes()*1.3:
            nnode=graph.number_of_nodes()
            nedges=graph.number_of_edges()
            a=True
            connect=0
            while a or (connect==0):
                G1=nx.gnm_random_graph(nnode,nedges)
                a, _ = nx.check_planarity(G1)
                connect=nx.edge_connectivity(G1)
            S1=to_numpy_array(G1)
            T1=torch.from_numpy(S1)
            TS1=to_sparse(T1)
            S0=to_numpy_array(graph)
            T0=torch.from_numpy(S0)
            TS0=to_sparse(T0)        
            data=Data(x=torch.ones((nnode,1)),edge_index=TS1._indices(),y=torch.ones(1).long())
            data1=Data(x=torch.ones((nnode,1)),edge_index=TS0._indices(),y=torch.zeros(1).long())
            torch.save(data,"../data/planar/planar_" + str(i) + ".pt")
            i+=1
            torch.save(data1,"../data/planar/planar_" + str(i) + ".pt")
            i+=1        

def generate_COLLAB_dataset(path):
    graphs=graph_label_list(path,'COLLAB_graph_labels.txt')
    adjency=compute_adjency(path,'COLLAB_A.txt')
    data_dict=graph_indicator(path,'COLLAB_graph_indicator.txt')
    
    for i in graphs:
        g=Graph()
        
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            for node2 in adjency[node]:
                g.add_edge((node,node2))

        S1=to_numpy_array(g.nx_graph)
        T1=torch.from_numpy(S1)
        TS1=to_sparse(T1)
        nnode = g.nx_graph.number_of_nodes()

        data=Data(x=torch.ones((nnode, 1)), edge_index=TS1._indices(), y=torch.ones(1).long() * (i[1]-1))
        torch.save(data,"../data/collab/collab_" + str(i[0]) + ".pt")

generate_COLLAB_dataset("COLLAB/")
