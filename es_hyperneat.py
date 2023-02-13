"""
Module to simulate ES-HyperNEAT on a population of 
CPPNs for the designing of Xenobots.
Implemented as described: http://eplex.cs.ucf.edu/ESHyperNEAT/
Based on https://github.com/ukuleleplayer/pureples/tree/master/pureples
"""

import numpy as np
from tools.activation_functions import sigmoid
import neat 

def get_weights(p):
    weights = []
    
    if p is not None and all(child is not None for child in p.cs):
        for c in p.cs:
            get_weights(c)
    else:
        if p is not None:
            weights.append(p.w)
    
    return weights

def variance(p):
    if not p:
        return 0
    else:
        return np.var(get_weights(p))

class Substrate:
    def __init__(self, input_coords, output_coords, resolution=10) -> None:
        self.input_coords = input_coords
        self.output_coords = output_coords
        self.hidden_coords = ()
        self.resolution = resolution
        
class QuadPoint:
    """
    Class describing a Quad Point, an area in the quadtree.
    """
    def __init__(self, x, y, width, level):
        self.x = x
        self.y = y
        self.width = width
        self.level = level
        self.cs = [None] * 4
        self.w = None
    
class Connection:
    def __init__(self, x1, y1, x2, y2, weight) -> None:
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.weight = weight

class EvolvableSubstrateNetwork:
    def __init__(self, substrate, cppn) -> None:
        self.substrate = substrate
        self.cppn = cppn
        self.initial_depth = 0 
        self.max_depth = 4
        self.variance_threshold = 0.03
        self.band_threshold = 0.3
        self.iteration_level = 1
        self.division_threshold = 0.03
    
    def assemble_neural_net(self):
        input_coords = self.substrate.input_coordinates
        output_coords = self.substrate.output_coordinates
        
        inputs = [range(len(input_coords))]
        outputs = [range(len(output_coords))]
        hidden_index = len(input_coords) + len(output_coords)
        
    def division_and_initilization(self, a, b, outgoing):
        root = QuadPoint(0,0,1,1)
        queue = [root]

        while len(queue) > 0:
            p = queue.pop(0)

            p.cs[0] = QuadPoint(p.x - p.width/2, p.y - p.width/2, p.width/2, p.level+1)
            p.cs[1] = QuadPoint(p.x - p.width/2, p.y + p.width/2, p.width/2, p.level+1)
            p.cs[2] = QuadPoint(p.x + p.width/2, p.y + p.width/2, p.width/2, p.level+1)
            p.cs[3] = QuadPoint(p.x + p.width/2, p.y - p.width/2, p.width/2, p.level+1)

            for c in p.cs:
                if outgoing:
                    c.w = query(a, b, c.x, c.y, self.cppn)
                else:
                    c.w = query(c.x, c.y, a, b, self.cppn)
                    
            if p.level < self.initial_depth or (p.level < self.max_depth and variance(p) > self.division_threshold):
                for child in p.cs:
                    queue.append(child)
        
        return root


    def pruning_and_extraction(self, a, b, connections, p, outgoing):
        for c in p.cs:
            if variance(c) >= self.variance_threshold:
                self.pruning_and_extraction(a, b, connections, c, outgoing)
            else:
                if outgoing:
                    d_left = abs(c.value - query(a, b, c.x - p.width, c.y, self.cppn))
                    d_right = abs(c.value - query(a, b, c.x + p.width, c.y, self.cppn))
                    d_top = abs(c.value - query(a, b, c.y - p.width, c.y, self.cppn))
                    d_bottom = abs(c.value - query(a, b, c.y + p.width, c.y, self.cppn))
                else:
                    d_left = abs(c.value - query(c.x - p.width, c.y, a, b, self.cppn))
                    d_right = abs(c.value - query(c.x + p.width, c.y, a, b, self.cppn))
                    d_top = abs(c.value - query(c.x, c.y - p.width, a, b, self.cppn))
                    d_bottom = abs(c.value - query(c.x, c.y + p.width, a, b, self.cppn))
                
            if max(min(d_top, d_bottom), min(d_left, d_right)) > self.band_threshold:
                if outgoing:
                    con = Connection(a, b, c.x, c.y, c.w, 5)
                else:
                    con = Connection(c.x, c.y, a, b, c.w, 5)
                
                if con not in connections:
                    connections.append(con)
            
            
    def es_hyperneat(self, InputPositions, OutputPositions):
        HiddenNodes = []
        connections1 = []
        connections2 = []
        connections3 = []
        for input in InputPositions:
            root = self.division_and_initilization(input.x, input.y, True)
            self.pruning_and_extraction(input.x, input.y, connections1, root, True)
            for connection in connections1:
                node = (connection.x2, connection.y2)
                if node not in HiddenNodes:
                    HiddenNodes.append(node)
        
        unexplored_hidden = HiddenNodes
        for i in range(self.iteration_level):
            for hidden in unexplored_hidden:
                root = self.division_and_initilization(hidden.x, hidden.y, True)
                self.pruning_and_extraction(hidden.x, hidden.y, connections2, root, True)
                for connection in connections2:
                    node = (connection.x2, connection.y2)
                    if node not in HiddenNodes:
                        HiddenNodes.append(node)
            
            unexplored_hidden = HiddenNodes - unexplored_hidden
    
        for output in OutputPositions:
            root = self.division_and_initilization(output.x, output.y, False)
            self.pruning_and_extraction(output.x, output.y, connections3, root, False)
        
        
        connections = connections1 + connections2 + connections3
        return connections, HiddenNodes

def query(x1, y1, x2, y2, cppn, max_weight=5.0):
    i = [x1, y1, x2, y2, 1]
    w = cppn.activate(i)[0]
    
    if abs(w) > 0.2:
        if w > 0:
            w = (w - 0.2) / 0.8
        else:
            w = (w + 0.2) / 0.8
        return w * max_weight
    else:
        return 0

if __name__ == "__main__":
    net = EvolvableSubstrateNetwork()