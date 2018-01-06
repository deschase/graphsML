import sys

import matplotlib.pyplot as plt
import networkx as nx
import graphs as gr
import community
import community_detection as com
import numpy as np




# gr2 = create_line_graph(gr1)
# draw_graph(gr2)

original= gr.Graph_community([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21])
#original= gr.Graph_community([1,2])
#original.draw_graph()
original.compute_D()
original.compute_E()
original.symmetrize_E()
algo = com.Louvain_algorithm_directed(original.E)
algo.run_algo(100)
algo.memory.print_com()
algo.memory.plot_evolution_community()
com = algo.memory.compute_total_community(len(algo.memory.stock) - 1)
for c in com:
    print c
    print ""
print len(com)
original.draw_graph_com(D=True,E=True, commu=com, total= True)
original.draw_graph_com(D=True, E=True, commu=com, circular=True, total=True)
original.draw_graph_com(D=True, E=True, commu=com, circular=True)


# to test the evolution of the coordinate of pi (power method to calculate the biggest eigenvector)
"""algo = com.Louvain_algorithm_directed(original.E)
list_pi = algo.compute_pi(100)
list_norm = [[] for j in range(10)]
for i in range(len(list_pi)):
    print list_pi[i][0,:]
    for j in range(10):
        list_norm[j].append(np.asscalar(list_pi[i][j,:]))
for j in range(10):
    plt.plot(list_norm[j])
plt.show()"""