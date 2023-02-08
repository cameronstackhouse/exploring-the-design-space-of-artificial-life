"""
Module to simulate Hyper-NEAT on a population of 
CPPNs for the designing of Xenobots.
Implemented as described: http://eplex.cs.ucf.edu/ESHyperNEAT/
"""

from networks import CPPN

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

def division_and_initilization(a, b, outgoing):
    root = QuadPoint(0,0,1,1)
    queue = [root]

    while len(queue) > 0:
        p = queue.pop(0)

        p.cs[0] = QuadPoint(p.x - p.width/2, p.y - p.width/2, p.width/2, p.level+1)
        p.cs[1] = QuadPoint(p.x - p.width/2, p.y + p.width/2, p.width/2, p.level+1)
        p.cs[2] = QuadPoint(p.x + p.width/2, p.y + p.width/2, p.width/2, p.level+1)
        p.cs[3] = QuadPoint(p.x + p.width/2, p.y - p.width/2, p.width/2, p.level+1)

        for c in p.cs:
            # if outgoing:
                # c.w <- CPPN(a, b, c.x, c.y)
            # else:
                # c.w <- CPPN(c.x, c.y, a, b)
            pass
        
        # if p.level < initialDepth or p.level < maxDepth & variance(p) > divThr:
            # foreach child in p.cs:
                # q.enqueue(child)


def pruning_and_extraction(a, b, connections, p, outgoing):
    pass

def es_hyperneat():
    pass


