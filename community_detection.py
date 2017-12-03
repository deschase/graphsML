from graphs import Graph_community
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

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
    # to continue to do spectral clustering correctly