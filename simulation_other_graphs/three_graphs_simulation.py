"""
Three-topology consensus simulation: K20, Clustered (2xK10+bridge), P20.
"""

import numpy as np

from network_data import get_complete_graph, get_clustered_graph, get_path_graph
from opinion_dynamics_models import animate_three_graphs

# Parameters
N = 20      # number of agents
T = 300     # simulation time (iterations)
DELTA = 0.06  # Euler step size

# Same initial opinions for all three topologies
np.random.seed(0)  # remove/change for a fresh random draw each run
x0 = np.random.rand(N)

# Three topologies to compare
graphs = {
    "Complete Graph (K20)": get_complete_graph(N),
    "Clustered Graph (2xK10 + bridge)": get_clustered_graph(N, cluster_size=10),
    "Path Graph (P20)": get_path_graph(N),
}

# Simulation + animation
anim, histories = animate_three_graphs(graphs, x0, T=T, delta=DELTA, a=0.06)

# Quick numerical check: all three should converge to the same consensus
# value (average of x0), just at different speeds.
for name, hist in histories.items():
    print(f"{name:35s} final mean = {hist[-1].mean():.4f}, "
          f"spread (max-min) = {hist[-1].max() - hist[-1].min():.2e}")