import sys

import matplotlib.pyplot as plt
import networkx as nx

def get_associations(namefile):
    dico = {}
    with open(namefile, "rb") as f:
        list_lines = f.readlines()

        for line in list_lines:
            line = line.decode("utf_8")
            line = line.replace('\n', '')
            line = line.replace('\r', '')
            line = line.split(',')
            dico[int(line[1])] = str(line[0])

    return dico


def get_matrix(namefile):
    matrix = list()
    with open(namefile, "rb") as f:
        list_lines = f.readlines()

        for line in list_lines:
            line = line.decode("utf_8")
            line = line.replace('\n', '')
            line = line.replace('\r', '')
            line = line.split(',')
            matrix.append([int(elem) for elem in line])

    return matrix


def draw_graph(graph):
    graph_pos = nx.spring_layout(graph)

    # draw nodes, edges and labels
    nx.draw_networkx_nodes(graph, graph_pos, node_size=100, node_color='orange', alpha=0.3)
    # we can now added edge thickness and edge color
    nx.draw_networkx_edges(graph, graph_pos, width=1, alpha=0.3, edge_color='blue')
    nx.draw_networkx_labels(graph, graph_pos, font_size=6, font_family='sans-serif')

    # show graph
    plt.show()


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

dico_ex = get_associations("data/1_2_3_4_5_correspondances.csv")
matrix_ex = get_matrix("data/1_2_3_4_5_matrix.csv")

Ex_graph = nx.Graph()

for name in dico_ex.values():
    Ex_graph.add_node(name)

for i in dico_ex.keys():
    for j in dico_ex.keys():
        if j < i and matrix_ex[i][j] != 0:
            Ex_graph.add_edge(dico_ex[i], dico_ex[j], weight=matrix_ex[i][j])

draw_graph(Ex_graph)
draw_graph(create_line_graph(Ex_graph))

# gr1 = nx.Graph()
# gr1.add_node(0)
# gr1.add_node(1)
# gr1.add_node(2)
# gr1.add_node(3)
# gr1.add_node(4)
# gr1.add_node(5)
# gr1.add_edge(0,1)
# gr1.add_edge(1,2)
# gr1.add_edge(1,3)
# gr1.add_edge(1,4)
# gr1.add_edge(2,4)
# gr1.add_edge(4,5)
# gr1.add_edge(2,5)
# draw_graph(gr1)
#
# gr2 = create_line_graph(gr1)
# draw_graph(gr2)
