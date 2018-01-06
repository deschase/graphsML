import matplotlib.pyplot as plt
import networkx as nx
import community
import numpy as np
import matplotlib.colors as mpc
import colorsys

def get_associations(namefile):
    """
    This function read a correspondance file and returns a diccionnary that
    associate every number with his associated character.
    :param namefile: A string, the name of the csv file with the relation
    between the characters and the number that will be used in the algorithm.
    :return: A diccionnary associating a number with a name
    """
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
    """
    This function read a matrix file and returns a matrix that gathers the
    weights of the edges between all the nodes (representing characters) of
    the data graph.
    :param namefile: A string, the name of the csv file with the matix data.
    :return: the weighted matrix of the graph encoded in the file.
    """
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
    """
    This class is going to enable us to compute the directed (knowing the weight) and undirected (ignoring the weight)
    line graph corresponding to a graph respresenting links between people
    """

    def __init__(self, tomes):
        """
        This function initalise the Graph_community object and gather the data
        corrsponding to the tomes number given as arguments
        :param tomes: list of the tomes you want to use (Int)
        :return: An instance of the Graph_community object
        """
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

    def draw_graph(self, weight = False):
        """
        This function is going to draw the original graph
        :param tomes: None
        :return: None
        """
        graph_pos = nx.spring_layout(self.graph, k=1.,iterations=20)

        # draw nodes, edges and labels
        # we can now added edge thickness and edge color
        if weight:
            w, maxi = self.return_total_and_max_weight()
            list_weight = [(float(self.graph[e[0]][e[1]]["weight"])/float(maxi))*200. for e in list(self.graph.edges)]
            print list_weight
            nx.draw_networkx_edges(self.graph, graph_pos, width=list_weight, alpha=0.3, edge_color='blue')
        else:
            nx.draw_networkx_edges(self.graph, graph_pos, width=1, alpha=0.3, edge_color='blue')
        nx.draw_networkx_nodes(self.graph, graph_pos, node_size=300, node_color='orange', alpha=0.9)
        nx.draw_networkx_labels(self.graph, graph_pos, font_size=10, font_family='sans-serif')

        # show graph
        plt.show()

    def draw_D(self):
        """
        This function is going to draw the graph D (defined in the article)
        :param tomes: None
        :return: None
        """
        graph_pos = nx.spring_layout(self.D, k=1.,iterations=20)

        # draw nodes, edges and labels

        # we can now added edge thickness and edge color
        nx.draw_networkx_edges(self.D, graph_pos, width=0.1, alpha=0.1, edge_color='orange')
        nx.draw_networkx_nodes(self.D, graph_pos, node_size=100, node_color='blue', alpha=0.9)
        nx.draw_networkx_labels(self.D, graph_pos, font_size=6, font_family='sans-serif')

        # show graph
        plt.show()

    def draw_E(self):
        """
        This function is going to draw E (defined in the article) with
        arrows indicating their orientations
        :param tomes: None
        :return: None
        """
        graph_pos = nx.spring_layout(self.E, k=1.,iterations=20)

        # draw nodes, edges and labels
        # we can now added edge thickness and edge color
        nx.draw_networkx_edges(self.E, graph_pos, width=1, alpha=0.3, edge_color='b', arrows = True)
        nx.draw_networkx_nodes(self.E, graph_pos, node_size=100, node_color='orange', alpha=0.9)
        nx.draw_networkx_labels(self.E, graph_pos, font_size=6, font_family='sans-serif')

        # show graph
        plt.show()

    def compute_C(self):
        """
        Compute the simple line graph corespondant to the original graph
        :param tomes: None
        :return: None
        """
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
        """
        Compute the exact directed graph E described in the article
        :param tomes: None
        :return: None
        """
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
        """
        Create a symmetrized version of E
        :param tomes: None
        :return: None
        """
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
        """
        Compute the exact directed graph E described in the article
        :param tomes: None
        :return: None
        """
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
        """
        Function that returns the maximum weight and the sum of all weights in
        the graph. If D and E are False, the function will return the weights
        of the original graph.
        :param D: Boolean, True if you want the weights of the graph D
                False (default)
        :param E: Boolean, True if you want the weights of the graph E
                False (default)
        :return weight: the summ of all weights of the graphs.
        :return maxi: the maximum of the weights of the graph.
        """
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

    def draw_graph_com(self, D=False, E=False, commu = [], circular = False, total = False):
        """
        This function is going to draw the original graph but with the
        community color obtained on it.
        :param D: Boolean, True if you want to plot the communities created by D
                  False (default)
        :param E: Boolean, True if you want the plot of the community created by the symmetrised E
                  False (default)
        :param com: list of list of the edges belonging to a community.
                    Default value: []
        :param circular: True if you want a circular graph, False otherwise
                    Default value : False
        :param total: boolean True if you want all the communities in the same graph, False othwise
                    Default value: False
        Note: you can draw all of them just by filling the parameters
        :return: None
        """
        colors = mpc.cnames.keys()
        if circular:
            pos = nx.circular_layout(self.graph)
        else:
            pos = nx.spring_layout(self.graph, k=1., iterations=20)
        w, maxi = self.return_total_and_max_weight()


        # first compute the best partition
        if E:
            partition = community.best_partition(self.E_sym)  # louvain algo with weighted original graph (with a E symmetrised)
            # drawing : we draw each community one by one
            size = float(len(set(partition.values())))
            if not total:
                count = 0
                for com in set(partition.values()):
                    count = count + 1.
                    list_edges = [edge for edge in partition.keys() if partition[edge] == com]
                    list_weight = [50. * float(self.graph[edge[0]][edge[1]]["weight"]) / float(maxi) for edge in
                                   partition.keys() if partition[edge] == com]
                    nx.draw_networkx_nodes(self.graph, pos, node_size=150, node_color='orange', alpha=0.9)
                    nx.draw_networkx_labels(self.graph, pos, font_size=10, font_family='sans-serif')
                    nx.draw_networkx_edges(self.graph, pos, edgelist=list_edges, alpha=0.5, width=list_weight,
                                           edge_color=colors[int(count)])
                    plt.title("Communities with E symmetric")
                    plt.show()
                    plt.clf()

            else:
                count = 0
                for com in set(partition.values()):
                    count = count + 1.
                    list_edges = [edge for edge in partition.keys() if partition[edge] == com]
                    list_weight = [50. * float(self.graph[edge[0]][edge[1]]["weight"]) / float(maxi) for edge in
                                   partition.keys() if partition[edge] == com]
                    nx.draw_networkx_edges(self.graph, pos, edgelist=list_edges, alpha=0.5, width=list_weight,
                                           edge_color=colors[int(count)])
                nx.draw_networkx_nodes(self.graph, pos, node_size=150, node_color='orange', alpha=0.9)
                nx.draw_networkx_labels(self.graph, pos, font_size=10, font_family='sans-serif')
                plt.title("Communities with E symmetric")

                plt.show()
                plt.clf()

        if D:
            partition = community.best_partition(self.D) # louvain algo with unweighted original graph
            # drawing : we draw each community one by one
            size = float(len(set(partition.values())))
            if not total:
                count = 0
                for com in set(partition.values()):
                    count = count + 1.
                    list_edges = [edge for edge in partition.keys() if partition[edge] == com]
                    list_weight = [50.*float(self.graph[edge[0]][edge[1]]["weight"])/float(maxi) for edge in
                                   partition.keys() if partition[edge] == com]
                    nx.draw_networkx_nodes(self.graph, pos, node_size=150, node_color='orange', alpha=0.9)
                    nx.draw_networkx_labels(self.graph, pos, font_size=10, font_family='sans-serif')
                    nx.draw_networkx_edges(self.graph, pos, edgelist=list_edges, alpha=0.5, width = list_weight,
                                           edge_color=colors[int(count)])
                    plt.title("Communities with D (unweighted original graph)")
                    plt.show()
                    plt.clf()
            else:
                count = 0
                for com in set(partition.values()):
                    count = count + 1.
                    list_edges = [edge for edge in partition.keys() if partition[edge] == com]
                    list_weight = [50. * float(self.graph[edge[0]][edge[1]]["weight"]) / float(maxi) for edge in
                                   partition.keys() if partition[edge] == com]
                    nx.draw_networkx_edges(self.graph, pos, edgelist=list_edges, alpha=0.5, width=list_weight,
                                           edge_color=colors[int(count)])
                nx.draw_networkx_nodes(self.graph, pos, node_size=150, node_color='orange', alpha=0.9)
                nx.draw_networkx_labels(self.graph, pos, font_size=10, font_family='sans-serif')
                plt.title("Communities with D")

                plt.show()
                plt.clf()

        if commu != []:

            if not total:
                count = 0
                for c in commu:
                    count = count + 1.
                    list_weight = [50. * float(self.graph[edge[0]][edge[1]]["weight"]) / float(maxi) for edge in c]
                    nx.draw_networkx_nodes(self.graph, pos, node_size=150, node_color='orange', alpha=0.3)
                    nx.draw_networkx_labels(self.graph, pos, font_size=10, font_family='sans-serif')
                    nx.draw_networkx_edges(self.graph, pos, edgelist=c, alpha=0.5, width=list_weight,
                                           edge_color=colors[int(count)])
                    plt.title("Communities with E directed")
                    plt.show()
                    plt.clf()
            else:
                count = 0
                for c in commu:
                    count = count + 1.
                    list_weight = [50. * float(self.graph[edge[0]][edge[1]]["weight"]) / float(maxi) for edge in c]
                    nx.draw_networkx_edges(self.graph, pos, edgelist=c, alpha=0.5, width=list_weight,
                                           edge_color=colors[int(count)])
                nx.draw_networkx_nodes(self.graph, pos, node_size=150, node_color='orange', alpha=0.3)
                nx.draw_networkx_labels(self.graph, pos, font_size=10, font_family='sans-serif')
                plt.title("Communities with E directed")

                plt.show()
                plt.clf()
