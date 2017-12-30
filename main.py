import sys

import matplotlib.pyplot as plt
import networkx as nx
import graphs as gr
import community
import community_detection as com

colors = ['blue', 'orange', 'purple', 'green', 'black', 'pink', 'brown', 'magenta', 'grey', 'cyan']
def create_line_graph(graph):
    line_graph = nx.Graph()
    list_edges = list(graph.edges)
    for link in list_edges:
        if link[0] != link[1]:
            line_graph.add_node(link)
    lg_node = list(line_graph.nodes)
    for nd in lg_node:
        extr1 = nd[0]
        extr2 = nd[1]
        for ngbr1 in graph.neighbors(extr1):
            if (extr1, ngbr1) in lg_node:
                line_graph.add_edge(nd, (extr1, ngbr1))
            elif (ngbr1, extr1) in lg_node:
                line_graph.add_edge(nd, (ngbr1, extr1))
        for ngbr2 in graph.neighbors(extr2):
            if (extr2, ngbr2) in lg_node:
                line_graph.add_edge(nd, (extr2, ngbr2))
            elif (ngbr2, extr2) in lg_node:
                line_graph.add_edge(nd, (ngbr2, extr2))
    return line_graph

# gr2 = create_line_graph(gr1)
# draw_graph(gr2)

original= gr.Graph_community([1])
original.draw_graph()
original.compute_D()
original.compute_E()
"""weight = list(original.E.edges(data=True))
for w in weight:
    print w
original.draw_E()"""

#first compute the best partition
partition = community.best_partition(original.E.to_undirected()) # algo de louvain with weighted original graph
#partition = community.best_partition(original.D) # algo de louvain with unweighted original graph

#drawing : we draw each community one by one
"""size = float(len(set(partition.values())))
pos = nx.circular_layout(original.graph)
count = 0
for com in set(partition.values()) :
    count = count + 1.
    list_edges = [edge for edge in partition.keys()
                                if partition[edge] == com]
    nx.draw_networkx_nodes(original.graph, pos, node_size=150, node_color='orange', alpha=0.3)
    nx.draw_networkx_labels(original.graph, pos, font_size=10, font_family='sans-serif')
    nx.draw_networkx_edges(original.graph, pos, edgelist = list_edges, alpha = 1, edge_color=colors[int(count)])
    plt.show()"""




