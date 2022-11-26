# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
import scipy.io

N_nodes=25
totalNoOfGraphs=10
graphArr=np.zeros([totalNoOfGraphs,N_nodes*N_nodes])
count=0
D_max=0
count1=0

while count<totalNoOfGraphs:
#    G = nx.fast_gnp_random_graph(N_nodes,0.05, directed=True)
    G = nx.fast_gnp_random_graph(N_nodes,0.1, directed=False)
#    G = nx.random_geometric_graph(N_nodes,.14,p=5)
    #G = nx.random_powerlaw_tree(N_nodes,gamma=3)
#    if nx.is_strongly_connected(G) == False:
    if nx.is_connected(G) == False:
#        print("Graph is not connected")
        count1=count1+1
        continue   
    A = nx.adjacency_matrix(G)
    A_np=nx.to_numpy_array(G)
    A_np_arr=A_np.reshape(N_nodes*N_nodes)
    graphArr[count]=A_np_arr
#    nx.draw_networkx_labels(G, pos=nx.spring_layout(G))
#    P = np.eye(N_nodes)+nx.to_numpy_array(G)
#    P = np.divide(P, np.sum(P,axis=0)) 
    D = nx.diameter(G)
    D_max=max(D_max,D)
    print("Diameter is ",D)
    count=count+1
    
scipy.io.savemat('graphArray.mat', mdict={'arr': graphArr,'D_max':D_max})
#f = open("graphArray.txt", "w")
#f.write(graphArr)
#f.close()