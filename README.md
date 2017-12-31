## graphsML

This github contains what we have done for a MVA project. It is supposed to be a part of the validation of the course "Graph In Machine Learning", taught by Michal Valko

## Code Example

## Motivation

The goal of this project is to detect community within a graph of people linked to each other. The idea is to detect community of edges (links) instead of detecting community of people. That way, some person can belong to severa communities (which is what happen mst of the time)
We decided to test this idea on the manga One Piece : we suppose that each time two characters appear on the same page they have a "+1" on the weight of the edge that links them together.

## Installation

To test the code we created, you will need networkx, python-louvain, matplotlib, numpy and sklearn

## Reference

All the project is based on an article written by T.S. Evans and R. Lambiotte : "Line Graphs of Weighted Networks for Overlapping Communities", see https://www.overleaf.com/read/qgvzjxsxsnkx to read the report we are making on the project.

## Tests


