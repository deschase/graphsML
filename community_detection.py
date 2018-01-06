from graphs import Graph_community
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
#from networkx import graphviz_layout

def spectral_clustering(line_graph, lap_param):
    """ Apply a spectral clustering on line_graph following the lap_param = "rw", "unn" or "sym" """
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

class Community_directed(object):
    """ This class is going to represent a communtiy in the Louvain algorithm
    for directed graph"""
    def __init__(self):
        self.nodes = [] # list of the nodes that belong to the community
        self.inner_graph = nx.DiGraph() # the inner graph of the community

    def is_empty(self):
        """
        Fubction that returns if the original graph is empty or not.
        :return: Boolean returning the emptiness or not
        """
        if self.nodes == []:
            return True
        else:
            return False


    def add_new_node(self, node, graph):
        """
        Function that adds a new node to the inner_graph.
        :param node: node to add
        :param graph: Graph from which we take the node to add (contains the
        data about the edges we need to )
        :return: None
        """
        self.inner_graph.add_node(node) # we add the node to the inner graph
        list_neigh = graph.neighbors(node)
        for n in list_neigh:
            if n in self.nodes:
                self.inner_graph.add_edge(n, node, weight = graph[n][node]["weight"])
                self.inner_graph.add_edge(node, n, weight = graph[node][n]["weight"])
        self.nodes.append(node) # we add the node to the list of nodes

    def remove_node(self, node):
        """
        Function that removes a node from the inner_graph.
        :param node: node to remove
        :return: None
        """
        self.inner_graph.remove_node(node) # we remove the node from the inner graph
        self.nodes = [n for n in self.nodes if n!=node] # idem from the list of nodes

    def gain_adding(self, node, graph, pi_vector, s):
        """
        :param node: node to test
        :param graph: original graph we are applying louvain algo on
        :param pi_vector: dominant eigenvector of transition matrix E/sout (as dictionnary)
        :param s: sout (as dictionnary)
        :return: modularity gain from adding node from the community
        """
        gain = 0.
        list_neigh = graph.neighbors(node)
        for n in list_neigh:
            if n in self.nodes:
                gain += graph[n][node]["weight"]/s[n]*pi_vector[n] - pi_vector[n]*pi_vector[node]
                gain += graph[node][n]["weight"]/s[node]*pi_vector[node] -  pi_vector[n]*pi_vector[node]
        return gain

    def gain_removing(self, node, graph, pi_vector, s):
        """
        :param node: node to test
        :param graph: original graph we are applying louvain algo on
        :param pi_vector: dominant eigenvector of transition matrix E/sout (as dictionnary)
        :param s: sout (as dictionnary)
        :return: modularity gain from removing node from the community
        """
        gain = 0.
        list_neigh = self.inner_graph.neighbors(node)
        for n in list_neigh:
            gain += -(graph[n][node]["weight"] / s[n] * pi_vector[n] - pi_vector[n] * pi_vector[node])
            gain += -(graph[node][n]["weight"] / s[node] * pi_vector[node] - pi_vector[n] * pi_vector[node])
        return gain

    def get_weight_from_community(self, community, graph):
        """ Return weight of edges to community from self"""
        weight = 0
        list_edges = list(graph.edges)
        for n1 in self.nodes:
            for n2 in community.nodes:
                if (n1, n2) in list_edges:
                    weight += graph[n1][n2]["weight"]
        return weight

    def get_weight_to_community(self, community, graph):
        """ Return weight of edges from community to self"""
        weight = 0
        list_edges = list(graph.edges)
        for n1 in self.nodes:
            for n2 in community.nodes:
                if (n2, n1) in list_edges:
                    weight += graph[n2][n1]["weight"]
        return weight


class Stock_communities(object):
    """ Class to stock layers of communities and compute the real global communities"""
    def __init__(self):
        self.stock = []

    def add_communities(self, list_com):
        dico = dict()
        for i in range(len(list_com)):
            dico[str(i)] = list_com[i]
        self.stock.append(dico)

    def plot_evolution_community(self):
        G = nx.Graph()
        for i in range(len(self.stock)):
            for j in self.stock[i].keys():
                for n in self.stock[i][j].nodes:
                    if i ==0:
                        G.add_node(n)
                    else:
                        G.add_node("b_"+ n + "_" + str(i))
                        for k in self.stock[i - 1][n].nodes:
                            if i==1:
                                G.add_edge("b_"+ n + "_" + str(i), k)
                            else:
                                G.add_edge("b_"+n + "_" + str(i),"b_"+ k + "_" + str(i - 1))

        pos = nx.spring_layout(G)#graphviz_layout(G, prog='twopi', args='')
        plt.figure(figsize=(15, 15))
        list_nd = list(G.nodes)
        real_list = [n for n in list_nd if n[0] != "b"]
        list_ignore = [n for n in list_nd if n[0] == "b"]
        label = dict()
        for n in real_list:
            label[n]= n
        nx.draw_networkx_nodes(G,pos,nodelist=real_list, node_color = "red", nodesize = 20, alpha = 0.3)
        nx.draw_networkx_nodes(G,pos, nodelist = list_ignore, node_color = "blue", alpha=0.3)
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif', labels=label)
        nx.draw_networkx_edges(G, pos, width=1, alpha=0.3, edge_color='b')
        plt.axis('equal')
        plt.show()

    def compute_total_community(self, level):
        if level >= len(self.stock):
            return "nope"
        else:
            nbCom = len(self.stock[level].keys())
            com = [[] for i in range(nbCom)]
            for k in self.stock[level].keys():
                com[int(k)] = self.stock[level][k].nodes
            new_com = [[] for i in range(nbCom)]
            if level !=0:
                for j in range(level):
                    lv = level - 1 - j
                    for j in range(nbCom):
                        for k in com[j]:
                            new_com[j].extend(self.stock[lv][k].nodes)
                    com = new_com
                    new_com = [[] for i in range(nbCom)]
            return com



    def print_com(self):
        for j in range(len(self.stock)):
            for i in self.stock[j].keys():
                print i
                print self.stock[j][i].nodes


class Louvain_algorithm_directed(object):
    def __init__(self, graph):
        self.list_communities = [] # actual list of communities
        self.correspondance = dict() # actual correspondance between node of the graph and community
        self.graph = graph
        i = 0
        for n in list(graph.nodes):
            c = Community_directed()
            c.add_new_node(n, self.graph)
            self.list_communities.append(c)
            self.correspondance[n] = i
            i+=1
        self.memory = Stock_communities()
        self.pi_vector = dict() # is going to contain the main eignevector of the transition matrix corresponding to self.graph
        self.s = dict() # is going to contain the sum on beta of A(alpha,beta) with A the adjacecy matrix of self.graph

    def compute_s(self):
        for n in list(self.graph.nodes):
            list_neigh = self.graph.neighbors(n)
            for n2 in list_neigh:
                if not n in self.s.keys():
                    self.s[n] = self.graph[n2][n]["weight"]
                else:
                    self.s[n] += self.graph[n2][n]["weight"]

    def compute_pi(self, num_simulations):
        """ Use power method to compute pi"""
        W = nx.to_numpy_matrix(self.graph)
        # We compute the transition matrix
        for j in range(W.shape[1]):
            W[:,j] = W[:,j]/np.sum(W[:,j])
        # Ideally choose a random vector
        # To decrease the chance that our vector
        # Is orthogonal to the eigenvector
        b_k = np.random.rand(W.shape[0]).reshape([W.shape[0], 1])


        for _ in range(num_simulations):
            # calculate the matrix-by-vector product Ab
            b_k1 = np.dot(W, b_k)

            # calculate the norm
            b_k1_norm = np.linalg.norm(b_k1)

            # re normalize the vector
            b_k = b_k1 / b_k1_norm
        i = 0
        for n in list(self.graph.nodes):
            self.pi_vector[n] = np.asscalar(b_k[i])
            i+=1

    def get_neighbour_community(self, node):
        """ Return the list of community in the neighbourhood of node"""
        com_neigh = []
        list_neigh = self.graph.neighbors(node)
        for n in list_neigh:
            if not self.correspondance[n] in com_neigh and self.correspondance[n] != self.correspondance[node]:
                com_neigh.append(self.correspondance[n])
        return com_neigh


    def step1(self):
        """ We try each node to see if te modularity can be increased"""
        print "entering step 1"
        test_total = False
        still_increasing = True
        k = 0
        list_nodes = list(self.graph.nodes)
        #new_list_nodes = list_nodes
        while still_increasing:
            k+= 1
            if k%2==0:
                print k
            still_increasing = False
            #list_nodes = new_list_nodes
            for n in list_nodes:
                # we get the community that are in te neighbourhood of n
                neigh = self.get_neighbour_community(n)
                if neigh != []:
                    # we compute the gain of removing n from its community and adding it to each of its neigh communities
                    lost = self.list_communities[self.correspondance[n]].gain_removing(node = n, graph = self.graph, pi_vector = self.pi_vector, s = self.s)
                    gains = np.asarray([self.list_communities[c].gain_adding(node = n, graph = self.graph, pi_vector = self.pi_vector, s = self.s) + lost for c in neigh])
                    maxi = np.max(gains)
                    maxcom = neigh[int(np.argmax(gains))]
                    if maxi > 0.:
                        # modularity is increasing
                        still_increasing = True
                        test_total = True
                        # we remove the node from its old community
                        self.list_communities[self.correspondance[n]].remove_node(n)
                        # we add it to the new one
                        self.list_communities[maxcom].add_new_node(n, self.graph)
                        self.correspondance[n] = maxcom
                    #else:
                    #    new_list_nodes = [l for l in new_list_nodes if l!=n]
        return test_total

    def step2(self):
        """ We create a new graph with each community being a node"""
        # We sort the communities to keep only the nonempty one
        print "entering step 2"
        final_list = []
        for com in self.list_communities:
            if not com.is_empty():
                final_list.append(com)
        print "final com selected"
        # We add this list to the memory, to keep all the correspondances
        self.memory.add_communities(final_list)
        # we create a new graph
        temp = nx.DiGraph()
        for i in range(len(final_list)):
            temp.add_node(str(i))
        for i in range(len(final_list)):
            for j in range(len(final_list)):
                if i!=j:
                    w = final_list[i].get_weight_from_community(final_list[j], self.graph) #we calculate the total weight from i to j
                    if w > 0:
                        temp.add_edge(str(i), str(j), weight = w)
        print "new graph created"
        # we reinitialize all the variables
        self.graph = temp
        self.list_communities = []  # actual list of communities
        self.correspondance = dict()  # actual correspondance between node of the graph and community
        i = 0
        for n in list(self.graph.nodes):
            c = Community_directed()
            c.add_new_node(n, self.graph)
            self.list_communities.append(c)
            self.correspondance[n] = i
            i += 1
        print "new values created"
        self.pi_vector = dict()  # is going to contain the main eignevector of the transition matrix corresponding to self.graph
        self.s = dict()  # is going to contain the sum on beta of A(alpha,beta) with A the adjacecy matrix of self.graph

    def run_algo(self, num_simulation):
        test = True
        i = 0
        while test:
            print "running", i
            i+=1
            self.compute_s()
            self.compute_pi(num_simulation)
            print "beginning step 1"
            test = self.step1()
            print "beginning step 2"
            self.step2()
