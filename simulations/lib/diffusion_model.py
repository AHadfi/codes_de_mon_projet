"""
Opinion dynamics simulation.

"""

from network_data import get_graph, get_data
from opinion_dynamics_models import DiffusionModel

# Parameters
N = 20  # number of agents
T = 10  # simulation time

# Data preparation
G = get_graph(N)
x = get_data(N)

# Simulation
model = DiffusionModel(G, x)
model.animate(T)
