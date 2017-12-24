import sys

import matplotlib.pyplot as plt
import networkx as nx

import graphs as gr
import community_detection as com


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

original= gr.Graph_community([1,2,3,4,5,6,7,8])
print(incidence_matrix(original))
original.draw_graph()
original.compute_D()
label = com.spectral_clustering(original.D, 'rw')
original.draw_graph_com( label, original.Dlist)
