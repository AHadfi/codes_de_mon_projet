"""
Prepares the network and opinions.
"""

import numpy as np
import networkx as nx


def get_graph(n, p=0.3):
    """Returns a connected Erdős-Rényi graph with exactly n nodes."""
    g = nx.erdos_renyi_graph(n, p)
    while not nx.is_connected(g):
        g = nx.erdos_renyi_graph(n, p)
    # Relabel nodes to 0, 1, ..., n - 1
    g = nx.convert_node_labels_to_integers(g)
    return g


def get_data(n):
    """
    Returns a randomly generated vector with values in the range [0, 1],
    representing data on the graph's nodes.
    """
    return np.random.rand(n)
