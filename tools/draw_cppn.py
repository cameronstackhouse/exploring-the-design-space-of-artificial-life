"""
Module to draw a CPPN using graphviz
"""

from graphviz import Digraph

def draw_cppn(
    cppn, 
    show_disabled_connections: bool = False, 
    show_weights: bool = False, 
    out_dir: str = "analysis"
    ) -> None:   
    """
    Function to draw a CPPN using graphviz

    :param cppn: CPPN to draw
    :param show_disabled_connections: Option if diabled connections should be shown
    :param show_weights: Option if weights should be shown on edges
    :param out_dir: Directory to save image produced
    """
    dot = Digraph() #Creates a new digraph

    counter = 0

    #Iterates through all nodes in a CPPN
    for layer in cppn.nodes:
        for node in layer:
            node.name = f"node{counter}" #Sets the name of the node
            counter+=1

    #Iterates through CPPN layer by layer
    for layer in cppn.nodes:
        with dot.subgraph() as s: #Creates a new subgraph at each layer
            s.attr(rank="same") #Sets the subgraph rank so that every node in a layer is shown on the same layer
            #Iterates through each node in the layer
            for node in layer:
                s.node(node.name, str(node.activation_function.__name__)) #Adds the node to the subgraph
    
    #Gets a list of connections to be shown 
    valid_connections = []
    for connection in cppn.connections: #Goes through all connections
        if show_disabled_connections or connection.enabled: #Checks if connection is enabled or show_diabled_connections is enabled
            valid_connections.append(connection) #Adds connection to the connections to be shown

    #Iterates through valid connections
    for connection in valid_connections:
        if show_weights:
            dot.edge(connection.out.name, connection.input.name, label=str(round(connection.weight, 2))) #Adds connection with weight to the graph
        else:
            dot.edge(connection.out.name, connection.input.name) #Adds connection to the graph

    dot.format = "png"
    dot.render("Graph", view=True, directory=f"{out_dir}/drawings", outfile=f"{out_dir}/drawings/graph.png") #Draws and saves the graph as a png file