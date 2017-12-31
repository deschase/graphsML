import sys

import matplotlib.pyplot as plt
import networkx as nx
import graphs as gr
import community
import community_detection as com




# gr2 = create_line_graph(gr1)
# draw_graph(gr2)

original= gr.Graph_community([1,2,3,4,5,6,7,8,9,10])
original.draw_graph()
original.compute_D()
original.compute_E()
#original.draw_graph_com(D = True, E = False)
original.symmetrize_E()

"""weight = list(original.E_sym.edges(data=True))
for w in weight:
    print w"""

original.draw_graph_com(D = False, E = True)
