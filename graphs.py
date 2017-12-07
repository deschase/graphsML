import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

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

class Graph_community(object):
    def __init__(self, tomes):
        nam = str(tomes[0])
        for t in tomes[1:len(tomes)]:
            nam += "_" + str(t)

        self.dico = get_associations("data/" + nam + "_correspondances.csv")
        self.matrix = np.asarray(get_matrix("data/" + nam + "_matrix.csv"))
        self.graph = nx.Graph()

        for name in self.dico.values():
            self.graph.add_node(name)

        for i in self.dico.keys():
            for j in self.dico.keys():
                if j < i and self.matrix[i][j] != 0:
                    self.graph.add_edge(self.dico[i], self.dico[j], weight=self.matrix[i, j])

        self.D = nx.Graph() # Line graph that corresponds to the graph, no consideration of the weight of the original graph
        self.Dlist = []
        self.E = nx.DiGraph() # Idem but with consideration of the weight

    def draw_graph(self):
        """ This function is going to draw the original graph """
        graph_pos = nx.spring_layout(self.graph)

        # draw nodes, edges and labels
        nx.draw_networkx_nodes(self.graph, graph_pos, node_size=100, node_color='orange', alpha=0.3)
        # we can now added edge thickness and edge color
        nx.draw_networkx_edges(self.graph, graph_pos, width=1, alpha=0.3, edge_color='blue')
        nx.draw_networkx_labels(self.graph, graph_pos, font_size=6, font_family='sans-serif')

        # show graph
        plt.show()

    def compute_E(self):
        "To complete : add direction"
        list_edges = list(self.graph.edges)
        for link in list_edges:
            if link[0] != link[1]: # check not self loop
                self.E.add_node(link)
        lg_node = list(self.E.nodes)
        for nd in lg_node:
            extr1 = nd[0]
            extr2 = nd[1]

            for ngbr1 in self.graph.neighbors(extr1):
                if (extr1, ngbr1) in lg_node:
                    self.E.add_edge(nd, (extr1, ngbr1))
                elif (ngbr1, extr1) in lg_node:
                    self.E.add_edge(nd, (ngbr1, extr1))
            for ngbr2 in self.graph.neighbors(extr2):
                if (extr2, ngbr2) in lg_node:
                    self.E.add_edge(nd, (extr2, ngbr2))
                elif (ngbr2, extr2) in lg_node:
                    self.E.add_edge(nd, (ngbr2, extr2))

    def compute_D(self):
        list_edges = list(self.graph.edges)

        for link in list_edges:
            if link[0] != link[1]: # check not self loop
                self.D.add_node(link)
                self.Dlist.append(link)

        lg_node = list(self.D.nodes)

        for nd in lg_node:
            extr1 = nd[0]
            extr2 = nd[1]
            k1 = len(list(self.graph.neighbors(extr1)))
            k2 = len(list(self.graph.neighbors(extr2)))
            if k1 != 1:
                for ngbr1 in self.graph.neighbors(extr1):
                    if (extr1, ngbr1) in lg_node:
                        self.D.add_edge(nd, (extr1, ngbr1), weight = 1./(k1 -1) )
                    elif (ngbr1, extr1) in lg_node:
                        self.D.add_edge(nd, (ngbr1, extr1), weight = 1./(k1 -1))
            if k2 != 1:
                for ngbr2 in self.graph.neighbors(extr2):
                    if (extr2, ngbr2) in lg_node:
                        self.D.add_edge(nd, (extr2, ngbr2), weight = 1./(k2 -1))
                    elif (ngbr2, extr2) in lg_node:
                        self.D.add_edge(nd, (ngbr2, extr2), weight = 1./(k2 -1))

    def return_total_weight(self, D, E):
        weight = 0
        if E:
            list_edges = list(self.E.edges)
            for e in list_edges:
                weight +=self.E[e]["weight"]
        elif D:
            list_edges = list(self.D.edges)
            for e in list_edges:
                weight +=self.D[e]["weight"]
        else:
            list_edges = list(self.graph.edges)
            for e in list_edges:
                weight +=self.graph[e]["weight"]
        return weight



    def draw_graph_com(self, label, list_edge):
        colors = ['b', 'r', 'yellow', 'green', 'purple', 'black', 'orange', 'magenta', 'grey', 'cyan', 'pink']
        """ This function is going to draw the original graph but with the community color obtained on it """
        graph_pos = nx.spring_layout(self.graph)
        dico = dict()
        # We get the different edge with their label to draw them separately
        for i in range(len(label)):
            if label[i] in dico.keys():
                dico[label[i]].append(list_edge[i])
            else:
                dico[label[i]] = [list_edge[i]]

        for j in dico.keys():
            # draw nodes, edges and labels
            nx.draw_networkx_nodes(self.graph, graph_pos, node_size=100, node_color='orange', alpha=0.3)
            # we can now added edge thickness and edge color
            nx.draw_networkx_edges(self.graph, graph_pos, edgelist= dico[j], width=2, alpha=0.6, edge_color=colors[j])
            nx.draw_networkx_labels(self.graph, graph_pos, font_size=10, font_family='sans-serif')
            plt.show()