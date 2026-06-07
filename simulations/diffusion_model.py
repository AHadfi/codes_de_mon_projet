"""
Opinion dynamics simulation.
"""

import lib.network_data
from lib.opinion_dynamics_models import DiffusionModel

# Parameters
N = 20  # number of agents
T = 10  # simulation time

# Data preparation
G = lib.network_data.get_graph(N)
x = lib.network_data.get_data(N)

# Simulation
model = DiffusionModel(G, x)
model.animate(T)
