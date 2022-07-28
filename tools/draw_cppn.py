"""

"""

from graphviz import Digraph

#TODO Add comments

def draw_cppn(cppn, show_disabled_connections = False, show_weights = False, out_dir = "analysis") -> None:   
    """
    Function to draw a CPPN using graphviz

    :param cppn:
    :param show_disabled_connections:
    :param show_weights:
    :param out_dir:
    """
    dot = Digraph()

    counter = 0

    for layer in cppn.nodes:
        for node in layer:
            node.name = f"node{counter}"
            counter+=1

    for layer in cppn.nodes:
        with dot.subgraph() as s:
            s.attr(rank="same")
            for node in layer:
                s.node(node.name, str(node.activation_function.__name__))
    
    valid_connections = []
    for connection in cppn.connections:
        if show_disabled_connections or connection.enabled:
            valid_connections.append(connection)

    #TODO Add weight to egdes

    for connection in valid_connections:
        if show_weights:
            dot.edge(connection.out.name, connection.input.name, label=str(round(connection.weight, 2)))
        else:
            dot.edge(connection.out.name, connection.input.name)

    dot.format = "png"
    dot.render("Graph", view=True, directory=f"{out_dir}/drawings", outfile=f"{out_dir}/drawings/graph.png") #TODO Change where files are saved