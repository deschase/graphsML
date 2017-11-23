from graphs import Graph

g = { "a" : ["d"],
      "b" : ["c"],
      "c" : ["b", "c", "d", "e"],
      "d" : ["a", "c"],
      "e" : ["c"],
      "f" : []
    }

graph = Graph(g)
print(graph)

for node in graph.vertices():
    print(graph.vertex_degree(node))

print("List of isolated vertices:")
print(graph.find_isolated_vertices())

print("""A path from "a" to "e":""")
print(graph.find_path("a", "e"))

print("""All pathes from "a" to "e":""")
print(graph.find_all_paths("a", "e"))

print("The maximum degree of the graph is:")
print(graph.Delta())

print("The minimum degree of the graph is:")
print(graph.delta())

print("Edges:")
print(graph.edges())

print("Degree Sequence: ")
ds = graph.degree_sequence()
print(ds)

fullfilling = [ [2, 2, 2, 2, 1, 1],
                     [3, 3, 3, 3, 3, 3],
                     [3, 3, 2, 1, 1]
                   ]
non_fullfilling = [ [4, 3, 2, 2, 2, 1, 1],
                    [6, 6, 5, 4, 4, 2, 1],
                    [3, 3, 3, 1] ]

for sequence in fullfilling + non_fullfilling :
    print(sequence, Graph.erdoes_gallai(sequence))

print("Add vertex 'z':")
graph.add_vertex("z")
print(graph)

print("Add edge ('x','y'): ")
graph.add_edge(('x', 'y'))
print(graph)

print("Add edge ('a','d'): ")
graph.add_edge(('a', 'd'))
print(graph)
