import matplotlib.pyplot as plt
import networkx as nx
import community
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
    """ This class is going to enable us to compute the directed (knowing the weight) and undirected (ignoring the weight)
    line graph corresponding to a graph respresenting links between people"""
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


        self.C = nx.Graph() # Simple line graph just to understand what it is
        self.D = nx.Graph() # Line graph that corresponds to the graph, no consideration of the weight of the original graph
        self.Dlist = []
        self.E = nx.DiGraph() # Idem but with consideration of the weight
        self.E_sym = nx.Graph() # symmetric version of E (undirected)

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

    def draw_D(self):
        """ This function is going to draw D"""
        graph_pos = nx.spring_layout(self.D)

        # draw nodes, edges and labels
        nx.draw_networkx_nodes(self.D, graph_pos, node_size=100, node_color='orange', alpha=0.3)
        # we can now added edge thickness and edge color
        nx.draw_networkx_edges(self.D, graph_pos, width=1, alpha=0.3, edge_color='b')
        nx.draw_networkx_labels(self.D, graph_pos, font_size=6, font_family='sans-serif')

        # show graph
        plt.show()

    def draw_E(self):
        """ This function is going to draw E with arrows indicating their orientations"""
        graph_pos = nx.spring_layout(self.E)

        # draw nodes, edges and labels
        nx.draw_networkx_nodes(self.E, graph_pos, node_size=100, node_color='orange', alpha=0.3)
        # we can now added edge thickness and edge color
        nx.draw_networkx_edges(self.E, graph_pos, width=1, alpha=0.3, edge_color='b', arrows = True)
        nx.draw_networkx_labels(self.E, graph_pos, font_size=6, font_family='sans-serif')

        # show graph
        plt.show()

    def compute_C(self):
        """ Compute the simple line graph corespondant to the original graph"""
        list_edges = list(self.graph.edges)
        for link in list_edges:
            if link[0] != link[1]:
                self.C.add_node(link)
        lg_node = list(self.C.nodes)
        for nd in lg_node:
            extr1 = nd[0]
            extr2 = nd[1]
            for ngbr1 in self.graph.neighbors(extr1):
                if (extr1, ngbr1) in lg_node:
                    self.C.add_edge(nd, (extr1, ngbr1))
                elif (ngbr1, extr1) in lg_node:
                    self.C.add_edge(nd, (ngbr1, extr1))
            for ngbr2 in self.graph.neighbors(extr2):
                if (extr2, ngbr2) in lg_node:
                    self.C.add_edge(nd, (extr2, ngbr2))
                elif (ngbr2, extr2) in lg_node:
                    self.C.add_edge(nd, (ngbr2, extr2))

    def compute_E(self):
        """Compute the exact directed graph described in the article"""
        list_edges = list(self.graph.edges)
        for link in list_edges:
            if link[0] != link[1]: # check not self loop
                self.E.add_node(link)
        lg_node = list(self.E.nodes)

        # Calculation of the strength of each node
        s = dict()
        for pre_node in list(self.graph.nodes):
            s[pre_node] = 0
            for ngh in list(self.graph.neighbors(pre_node)):
                if ngh != pre_node:
                    s[pre_node] += self.graph[pre_node][ngh]['weight']

        # Creation of the edges
        for nd in lg_node:
            extr1 = nd[0]
            extr2 = nd[1]
            wgt_alpha = self.graph[extr1][extr2]['weight']
            for ngbr1 in self.graph.neighbors(extr1):
                if (extr1, ngbr1) != (extr1, extr2) and (ngbr1, extr1) != (extr2, extr1):  # to avoid self loop
                    if (extr1, ngbr1) in lg_node:
                        wgt_beta = self.graph[extr1][ngbr1]['weight']
                        self.E.add_edge(nd, (extr1, ngbr1),
                                        weight = float(wgt_alpha)/float(s[extr1] - wgt_beta))
                    elif (ngbr1, extr1) in lg_node:
                        wgt_beta = self.graph[ngbr1][extr1]['weight']
                        self.E.add_edge(nd, (ngbr1, extr1),
                                        weight = float(wgt_alpha)/float(s[extr1] - wgt_beta))
            for ngbr2 in self.graph.neighbors(extr2):
                if (ngbr2, extr2) != (extr1, extr2) and (extr2, ngbr2) != (extr2, extr1):  # to avoid self loop
                    if (extr2, ngbr2) in lg_node:
                        wgt_beta = self.graph[extr2][ngbr2]['weight']
                        self.E.add_edge(nd, (extr2, ngbr2),
                                        weight = float(wgt_alpha)/float(s[extr2] - wgt_beta))
                    elif (ngbr2, extr2) in lg_node:
                        wgt_beta = self.graph[ngbr2][extr2]['weight']
                        self.E.add_edge(nd, (ngbr2, extr2),
                                        weight = float(wgt_alpha)/float(s[extr2] - wgt_beta))

    def symmetrize_E(self):
        """ Create a symmetrized version of E"""
        lg_node = list(self.E.nodes)
        for nd in lg_node: # we add the same nodes than in self.E
            self.E_sym.add_node(nd)

        list_edges = list(self.E.edges)
        list_already_seen = []
        for edge in list_edges:
            extr1 = edge[0]
            extr2 = edge[1]
            if not ((extr2, extr1) in list_already_seen):
                w1 = self.E[extr1][extr2]['weight']
                w2 = self.E[extr2][extr1]['weight']
                self.E_sym.add_edge(extr1, extr2, weight = float(w1 + w2)/ 2.)
                list_already_seen.append((extr1, extr2))

    def compute_D(self):
        list_edges = list(self.graph.edges) #get the list of edges of the original graph

        for link in list_edges: # create the list of nodes of D (the edges of the original graph)
            if link[0] != link[1]: # check not self loop
                self.D.add_node(link)
                self.Dlist.append(link)

        lg_node = list(self.D.nodes) # get the list of nodes of the graph D (line graph)

        for nd in lg_node: # for each node of D
            extr1 = nd[0] # get the 2 vertex of the original edge
            extr2 = nd[1]
            k1 = len(list(self.graph.neighbors(extr1))) # get the degree of each original node
            k2 = len(list(self.graph.neighbors(extr2)))
            if k1 != 1:
                for ngbr1 in self.graph.neighbors(extr1): # get the neighbours of one of the original vertex
                    if (extr1, ngbr1) != (extr1, extr2) and (ngbr1, extr1) != (extr2, extr1): # to avoid self loop
                        if (extr1, ngbr1) in lg_node:
                            self.D.add_edge(nd, (extr1, ngbr1), weight = 1./(k1 -1) )
                        elif (ngbr1, extr1) in lg_node:
                            self.D.add_edge(nd, (ngbr1, extr1), weight = 1./(k1 -1))
            if k2 != 1:
                for ngbr2 in self.graph.neighbors(extr2):
                    if (ngbr2, extr2) != (extr1, extr2) and (extr2, ngbr2) != (extr2, extr1):  # to avoid self loop
                        if (extr2, ngbr2) in lg_node:
                            self.D.add_edge(nd, (extr2, ngbr2), weight = 1./(k2 -1))
                        elif (ngbr2, extr2) in lg_node:
                            self.D.add_edge(nd, (ngbr2, extr2), weight = 1./(k2 -1))

    def return_total_and_max_weight(self, D= False, E=False):
        weight = 0
        maxi = 0.
        if E:
            list_edges = list(self.E.edges)
            for e in list_edges:
                weight +=self.E[e[0]][e[1]]["weight"]
                if self.E[e[0]][e[1]]["weight"] > maxi:
                    maxi = self.E[e[0]][e[1]]["weight"]
        elif D:
            list_edges = list(self.D.edges)
            for e in list_edges:
                weight +=self.D[e[0]][e[1]]["weight"]
                if self.D[e[0]][e[1]]["weight"] > maxi:
                    maxi = self.D[e[0]][e[1]]["weight"]
        else:
            list_edges = list(self.graph.edges)
            for e in list_edges:
                weight +=self.graph[e[0]][e[1]]["weight"]
                if self.graph[e[0]][e[1]]["weight"] > maxi:
                    maxi = self.graph[e[0]][e[1]]["weight"]
        return weight, maxi

    def draw_graph_com(self, D, E, com = []):
        """ This function is going to draw the original graph but with the community color obtained on it """
        colors = ['blue', 'orange', 'purple', 'green', 'black', 'pink', 'brown',
                  'magenta', 'grey', 'cyan', 'blue', 'orange', 'purple',
                  'green', 'black', 'pink', 'brown', 'magenta', 'grey', 'cyan']
        if com==[]:
            # first compute the best partition
            if E:
                partition = community.best_partition(self.E_sym)  # louvain algo with weighted original graph (with a E symmetrised)
            if D:
                partition = community.best_partition(self.D) # louvain algo with unweighted original graph
            w, maxi= self.return_total_and_max_weight()
            # drawing : we draw each community one by one
            size = float(len(set(partition.values())))
            pos = nx.circular_layout(self.graph)
            count = 0
            for com in set(partition.values()):
                count = count + 1.
                list_edges = [edge for edge in partition.keys() if partition[edge] == com]
                list_weight = [20.*float(self.graph[edge[0]][edge[1]]["weight"])/float(maxi) for edge in partition.keys() if partition[edge] == com]
                nx.draw_networkx_nodes(self.graph, pos, node_size=150, node_color='orange', alpha=0.3)
                nx.draw_networkx_labels(self.graph, pos, font_size=10, font_family='sans-serif')
                nx.draw_networkx_edges(self.graph, pos, edgelist=list_edges, alpha=1., width = list_weight, edge_color=colors[int(count)])
                plt.show()
        else:
            w, maxi = self.return_total_and_max_weight()
            pos = nx.circular_layout(self.graph)
            count = 0
            for c in com:
                count = count + 1.
                list_weight = [20. * float(self.graph[edge[0]][edge[1]]["weight"]) / float(maxi) for edge in c]
                nx.draw_networkx_nodes(self.graph, pos, node_size=150, node_color='orange', alpha=0.3)
                nx.draw_networkx_labels(self.graph, pos, font_size=10, font_family='sans-serif')
                nx.draw_networkx_edges(self.graph, pos, edgelist=c, alpha=1., width=list_weight,
                                       edge_color=colors[int(count)])
                plt.show()
