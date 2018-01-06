import sys

import matplotlib.pyplot as plt
import networkx as nx
import graphs as gr
import community
import community_detection as com
import numpy as np
import matplotlib.colors as mpc



# We create the graph
original= gr.Graph_community([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21])
# We draw it
original.draw_graph()
# We compute the different line graphs
original.compute_D()
original.compute_E()
original.symmetrize_E()
# We use the louvain algorithm on E
algo = com.Louvain_algorithm_directed(original.E)
algo.run_algo(100)
algo.memory.print_com()
# We plot the evolution of the communities
algo.memory.plot_evolution_community()
# We compute the communities obtained on the last level (biggest community)
com = algo.memory.compute_total_community(len(algo.memory.stock) - 1)
for c in com:
    print c
    print ""
print len(com)
# We draw the communities obtained with different manners
original.draw_graph_com(D=True,E=True, commu=com, total= True)
original.draw_graph_com(D=True, E=True, commu=com, circular=True, total=True)
original.draw_graph_com(D=True, E=True, commu=com, circular=True)"

# To apply Louvain algo directly on the original graph uncomment
"""colors = mpc.cnames.keys()
pos = nx.spring_layout(original.graph, k=1., iterations=20)
w, maxi = original.return_total_and_max_weight()
partition = community.best_partition(original.graph) # louvain algo with original graph
# drawing : we draw each community one by one
size = float(len(set(partition.values())))
count = 0
for com in set(partition.values()):
    count = count + 1.
    list_nodes = [node for node in partition.keys() if partition[node] == com]
    nx.draw_networkx_nodes(original.graph, pos,node_size=200, node_color=colors[int(count)] ,alpha=0.9, nodelist=list_nodes)


nx.draw_networkx_labels(original.graph, pos, font_size=10, font_family='sans-serif')
list_weight = [50. * float(original.graph[edge[0]][edge[1]]["weight"]) / float(maxi) for edge in
               list(original.graph.edges)]
nx.draw_networkx_edges(original.graph, pos, alpha=0.5, width=list_weight,
                           edge_color= "blue")
plt.title("Communities of vertexes obtained with Louvain algorithm")

plt.show()"""


# to test the evolution of the coordinate of pi (power method to calculate the biggest eigenvector) uncomment
"""algo = com.Louvain_algorithm_directed(original.E)
list_pi = algo.compute_pi(100)
list_norm = [[] for j in range(10)]
for i in range(len(list_pi)):
    print list_pi[i][0,:]
    for j in range(10):
        list_norm[j].append(np.asscalar(list_pi[i][j,:]))
for j in range(10):
    plt.plot(list_norm[j])
plt.show()"""