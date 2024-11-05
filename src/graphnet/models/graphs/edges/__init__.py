"""Modules for constructing graphs.

´GraphDefinition´ defines the nodes and their features,  and contains general
graph-manipulation.´EdgeDefinition´ defines how edges are drawn between nodes
and their features.
"""

from .edges import EdgeDefinition, KNNEdges, RadialEdges, EuclideanEdges
from .minkowski import MinkowskiKNNEdges
