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

