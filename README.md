## graphsML

This github contains what we have done for a MVA project. It is supposed to be a part of the validation of the course "Graph In Machine Learning", taught by Michal Valko.
This aim of this code is to implement and to test the method of community detection developped in the article "Line Graphs of Weighted Networks for Overlapping Communities" written by T.S. Evans and R. Lambiotte. The article can be found here: https://arxiv.org/abs/0912.4389.

## General code organization

### Data gathering

In order to test communnty detection, the frist thong to have is data.
As we wanted to do a community detection over the caracters in One piece we had to create a database we could use. To do that, we created two files:
- script_creation_data.py that launches a script you have to follow in order to create a database formatted in the proper way.
- script_creation_matix.py that launches a script creating the matrix representative of the commnuity from the data created by the previous script.

In order to use those two files, you only have to execute
  "python script_creation_data.py"
to create the data of one of the volume of one Piece
then
  "python script_creation_matix.py"
with the volume numbers of volumes you have already entered in the database.
That way, you will be able to use a graph that gather the data from the volumes you want.

### Graph Creation

The creation of the different line graphs, and the plot of those graphs are situatedd in the file
"graph.py".
Here is a brief description of what this fils contains:
- get_associations and get_matrix that are function reading the data file created before, in order to create the diccionnaries.
- Graph_community which is a class able to manage the creations of the line graphs described in the article.

#### Graph_community class

The class is composed of the following attributes:
- self.dico which is the diccionnay keeping track of the correspondances between the node numbers and the names
- self.matrix which is the matrix of weights of the original graph
- self.graph which is th original graph.
- self.C the simple line graph asssociated to the original graph, with no weight considered at all
- self.D the line graph that corresponds to the graph with no consideration of the weight of the original graph. This is the graph D described in the article
- self.E the oriented and weightes line-graph associated with the original graph. This is the graph E described in the article.
- self.E_sym a symetric version of E, created from the self.E graph.

The function of this class are the following ones:
- compute_C() that create the C line-graph associated with the original graph.
- compute_D() that create the C line-graph associated with the original graph.
- compute_E() that create the E line-graph associated with the original graph.
- symmetrize_E() that create the symmetric version of the E line-graph associated with the original graph.
- return_total_and_max_weight(D, E) that returns the sum of all weights and the maximal weight of the graph wanted.
- draw_graph() that plot the original graph.
- draw_D() that plot the D graph.
- draw_E() that plot the E graph.
- draw_graph_com(D, E, com) that plot the reults of the community detection for the graph wanted.


### Community detection

In order to docommunity detection in the line graphs created we had use community detection algorithms over those graphs. During this project, we implemented two main algorithms, contained in the file community_detection.py:
- The spectral clustering that is contained in the function spectral_clustering(graph, lap_param)
- The Louvain algorithms mainly contained in the class Louvain_algorithm_directed

#### Louvain_algorithm_directed class

The Louvain algorithm class is composed of the following attributes:
-
The functions of the class are the following:
- compute_s() that computes the out-strength of the nodes in self.graph
- compute_pi(num_simulations) that computes the principal eigenvector of the transition matrix
- get_neighbour_community(node) that returns the communities of the neigbours of the node given as argument.
- step1() step 1 of the Louvain algorithm, which select the different nodes to include in every community, according the modularities.
- step2() step 2 of the Louvain algorithm, which create a new graph based on the previous step 1.
- run_algo() function to launch the whole Louvain algorithm

It uses other classes that were created in order to ease a bit the creation of the intermediate graphs created in the Louvain algorithm. The class Community_directed is there to be used during the step one of the Louvain algorithm for the management of the communities, and to make the calculations about the modularity. The class Stock_communities was used to keep track of the communities a each iteration of the Louvain algorithm, and keep the different layers of the community creation during the algorithm.
Here is a detail of the class Community_directed.
First the attributes of this class are:
- self.nodes which is the list of the nodes that belong to the community encoded in the instace of the class
- self.inner_graph which is a directed graph reprsenting the of the community, we can call it the inner graph.
The different functions used in this class are the following:
- is_empty() returns if the original graph is empty or not.
- add_new_node(node, graph) adds a node to the graph
- remove_node(node) which remove a node from the inner graph.
- gain_adding(node, graph, pi_vector, s) which calculates the gain created when adding a node in the community.
- gain_removing(node, graph, pi_vector, s) which calculates the gain created when removing a node from the community.
- get_weight_from_community(community, graph) which returns the weights of the edges that goes from the community toward the graph given as argument.
- get_weight_to_community(community, graph) which returns the weights of the edges that goes from the graph in argument toward the graph of the community.

Here is a detail of the class Stock_communities. The only attribute of this class is self.stock that is a list of diccionnaries containing all the data about the communities created at each layer of the algorithm. The function of this class are
- add_communities(list_com) that add the diccionnay containing all the data of the communities given in argument (list_com is a list of list, representing a gathering of communities).
- plot_evolution_community() which plot the evolution of the communities during the steps of Louvain algorithm.
- compute_total_community(level) that create the communities using Louvain algorithm at the step level wanted.
- print_com() which print the evolution of the communities during the steps of Louvain algorithm.

## Motivation

The goal of this project is to detect community within a graph of people linked to each other. The idea is to detect community of edges (links) instead of detecting community of people. That way, some person can belong to several communities (which is what happens most of the time)
We decided to test this idea on the manga One Piece : we suppose that each time two characters appear on the same page they have a "+1" on the weight of the edge that links them together.

## Installation

To test the code we created, you will need networkx, python-louvain, matplotlib, numpy and sklearn

## Reference

As mentionned above, all the project is based on an article written by T.S. Evans and R. Lambiotte : "Line Graphs of Weighted Networks for Overlapping Communities" (https://arxiv.org/abs/0912.4389)
We wrote a report reflecting our understanding and our experiments about the subject. See https://www.overleaf.com/read/qgvzjxsxsnkx to read the report we are making on the project.
More generally, this project was done for the cours "Graphs in Machine Learning" taught by Michal Valko:
http://researchers.lille.inria.fr/~valko/hp/ .

## Use of the code

In order to use the code, you first have to create the data files if they are not cerated yet.
Then you have to enter the list of tomes you wanted to in the main file (the tome_list variable).
Just launch
"python main.py"
to get the results. 
