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

The Louvain algorithm class

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
