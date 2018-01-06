import sys

import matplotlib.pyplot as plt
import networkx as nx
import graphs as gr
import community
import community_detection as com

# gr2 = create_line_graph(gr1)
# draw_graph(gr2)

tome_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
original= gr.Graph_community(tome_list)
original.draw_graph()
original.compute_D()
original.compute_E()
#original.draw_graph_com(D = True, E = False)
original.symmetrize_E()

"""
weight = list(original.E_sym.edges(data=True))
for w in weight:
    print w

"""

#original.draw_graph_com(D = False, E = True)
algo = com.Louvain_algorithm_directed(original.E)
algo.run_algo(100)
algo.memory.print_com()
algo.memory.plot_evolution_community()
com = algo.memory.compute_total_community(len(algo.memory.stock) - 1)
for c in com:
    print c
    print ""
original.draw_graph_com(False,False,com)
