"""

"""

from graphviz import Digraph

#TODO Add comments
#TODO Try and find a way to make it output defined layers

def draw_cppn(cppn, show_disabled_connections = False, out_dir = "analysis"):   
    dot = Digraph()

    counter = 33

    for layer in cppn.nodes:
        for node in layer:
            if counter == 34:
                counter+=1
            node.name = chr(counter) #TODO Try and find better solution to labelling problem 
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
    try:
        dot.edges([str(connection.out.name) + str(connection.input.name) for connection in valid_connections])
    except:
        #TODO Fix this error!
        #TODO Fix blank node error
        print(f"out: {connection.out.name} in: {connection.input.name}")

    dot.format = "png"
    dot.render("Graph", view=True, directory=f"{out_dir}/drawings", outfile=f"{out_dir}/drawings/graph.png") #TODO Change where files are saved