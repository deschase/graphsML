## graphsML

This github contains what we have done for a MVA project. It is supposed to be a part of the validation of the course "Graph In Machine Learning", taught by Michal Valko.
This aim of this code is to implement and to test the method of community detection developped in the article "Line Graphs of Weighted Networks for Overlapping Communities" written by T.S. Evans and R. Lambiotte. The article can be found here: https://arxiv.org/abs/0912.4389.

## Code Example

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



## Motivation

The goal of this project is to detect community within a graph of people linked to each other. The idea is to detect community of edges (links) instead of detecting community of people. That way, some person can belong to several communities (which is what happens most of the time)
We decided to test this idea on the manga One Piece : we suppose that each time two characters appear on the same page they have a "+1" on the weight of the edge that links them together.

## Installation

To test the code we created, you will need networkx, python-louvain, matplotlib, numpy and sklearn

## Reference

As mentionned above, all the project is based on an article written by T.S. Evans and R. Lambiotte : "Line Graphs of Weighted Networks for Overlapping Communities" (https://arxiv.org/abs/0912.4389)
We wrote a report reflecting our understanding and our experiments about the subject. See https://www.overleaf.com/read/qgvzjxsxsnkx to read the report we are making on the project.
More generally, this project was done for the cours "Graphs in Machine Learning" taught by Michal Valko:
http://researchers.lille.inria.fr/~valko/hp/ . Our

## Tests
