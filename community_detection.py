from graphs import Graph_community
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans

def spectral_clustering(line_graph, lap_param):
    W = nx.to_numpy_matrix(line_graph)
    diago = np.diag(np.diag(W))
    if lap_param == "unn":
        L = np.subtract(diago, W)
    elif lap_param == "sym":
        L = np.linalg.inv(np.sqrt(diago)).dot(np.subtract(diago, W)).dot(np.linalg.inv(np.sqrt(diago)))
    elif lap_param == "rw":
        L = np.linalg.inv(diago).dot(np.subtract(diago, W))

    eigenvalues, eigenvectors = np.linalg.eig(L)
    sort = np.argsort(eigenvalues)
    nbselected = 30
    eigenval_selected = [eigenvalues[sort[i]] for i in range(nbselected)]
    print eigenval_selected
    indbig = 1
    maxi = 0
    for k in range(nbselected - 1):
        val = abs(eigenval_selected[k + 1] - eigenval_selected[k])
        if (maxi <= val):
            indbig = k
            maxi = val
    print indbig
    print maxi
    indbig = 10
    eigen_vector_selected = eigenvectors[sort[0]]
    for i in range(1, indbig + 1):
        eigen_vector_selected = np.concatenate((eigen_vector_selected, eigenvectors[sort[i]]), axis=0)
    eigen_vector_selected = eigen_vector_selected.T
    kmeans = KMeans(n_clusters=indbig + 1, random_state=0).fit(eigen_vector_selected)
    label = kmeans.labels_
    return label

class Community_undirected(object):
    def __init__(self):
        self.nodes = [] # list of the nodes that belong to the community
        self.inner_weight = 0 # the weights within the nodes of the community
        self.weight_tot = 0 # weight of the edges incident to the vertexes of the community

    def add_new_node(self, node, graph):
        list_neigh = graph.neighbors(node)
        for n in list_neigh:
            if n in self.nodes:
                self.inner_weight += graph[node][n]["weight"]
            self.weight_tot += graph[node][n]["weight"]

        self.nodes.append(node)

    def get_weight_to_node(self, node, graph):
        list_neigh = graph.neighbors(node)
        inner_weight = 0
        weight_tot = 0
        for n in list_neigh:
            if n in self.nodes:
                inner_weight += graph[node][n]["weight"]
            weight_tot += graph[node][n]["weight"]
        return weight_tot, inner_weight

    def get_weight_to_community(self, community, graph):
        weight = 0
        list_edges = list(graph.edges)
        for n1 in self.nodes:
            for n2 in community.nodes:
                if (n1, n2) in list_edges:
                    weight += graph[n1][n2]["weight"]
        return weight


class Stock_communities(object):
    def __init__(self):
        self.stock = []

    def add_communities(self, list_com):
        dico = dict()
        for i in range(len(list_com)):
            dico[i] = list_com[i]
        self.stock.append(dico)