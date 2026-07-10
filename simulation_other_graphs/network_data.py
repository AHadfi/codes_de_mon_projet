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


def get_complete_graph(n=20):
    """Complete graph K_n: every agent is connected to every other agent."""
    return nx.complete_graph(n)


def get_clustered_graph(n=20, cluster_size=10):
    """
    Two cliques of size `cluster_size` (K_10 each by default), joined by a
    single bridge edge. Models two tightly-knit communities that only
    communicate through one weak link.
    """
    assert 2 * cluster_size == n, "n must equal 2 * cluster_size"

    g1 = nx.complete_graph(cluster_size)
    g2 = nx.complete_graph(cluster_size)
    g2 = nx.relabel_nodes(g2, {i: i + cluster_size for i in range(cluster_size)})

    g = nx.compose(g1, g2)
    # Single bridge connecting the two clusters
    g.add_edge(cluster_size - 1, cluster_size)
    return g


def get_path_graph(n=20):
    """Path graph P_n: agents arranged in a line, each linked to its neighbors."""
    return nx.path_graph(n)