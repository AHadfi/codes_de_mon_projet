"""
Opinion dynamics models.
"""

from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx


class OpinionDynamicsModel(ABC):
    """
    Abstract base class for opinion dynamics models.
    """

    name = ""

    def __init__(self, graph, initial_opinions):
        self.graph = graph
        self.n = len(graph.nodes)
        self.x = initial_opinions

    @abstractmethod
    def step(self):
        """
        Advance the system by one time step.
        """
        pass

    def run(self, T):
        """
        Simulate T time steps.
        """
        history = [self.x.copy()]
        for _ in range(T):
            self.step()
            history.append(self.x.copy())
        return np.array(history)

    def animate(
        self,
        T,
        interval=300,
        cmap='bwr',
        vmin=0,
        vmax=1,
        node_size=300,
        figsize=(6, 6),
    ):
        """
        Animates opinion dynamics.
        """
        history = self.run(T)
        pos = nx.spring_layout(self.graph)
        fig, ax = plt.subplots(figsize=figsize)

        # Initial draw
        nodes = nx.draw_networkx_nodes(
            self.graph,
            pos,
            node_color=history[0],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            node_size=node_size,
            ax=ax
        )
        nx.draw_networkx_edges(self.graph, pos, alpha=0.4, ax=ax)
        nx.draw_networkx_labels(self.graph, pos, font_size=8, ax=ax)

        # ---- Animation update ----
        def update(frame):
            """
            Animates the opinion dynamics.
            """
            nodes.set_array(history[frame])
            ax.set_title(f"{self.name} (t={frame})")
            return nodes
        # ---- Animate ----
        anim = FuncAnimation(
            fig,
            update,
            frames=len(history),
            interval=interval
        )
        plt.show()
        return anim


class DiffusionModel(OpinionDynamicsModel):
    """
    Diffusion model for opinion dynamics:

        dx/dt = (-a) L x(t)
    where L is the graph Laplacian.
    """

    name = "Diffusion Model"

    def __init__(self, graph, initial_opinions):
        super().__init__(graph, initial_opinions)

    def step(self):
        # TODO
        pass
