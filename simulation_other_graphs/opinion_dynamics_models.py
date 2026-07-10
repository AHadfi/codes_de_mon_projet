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

    Discretized with forward Euler, step size delta:

        x(t + delta) = x(t) - delta * a * L * x(t)
    """

    name = "Diffusion Model"

    def __init__(self, graph, initial_opinions, delta=0.1, a=0.06):
        super().__init__(graph, initial_opinions)
        self.delta = delta
        self.a = a
        self.L = nx.laplacian_matrix(graph).toarray().astype(float)

    def step(self):
        self.x = self.x - self.delta * self.a * (self.L @ self.x)


def animate_three_graphs(
    graphs,
    x0,
    T=100,
    delta=0.1,
    a=0.06,
    interval=100,
    cmap="bwr",
    vmin=0,
    vmax=1,
    node_size=150,
    figsize=(16, 5.5),
    seed=42,
):
    """
    Runs DiffusionModel on each graph in `graphs` (dict: name -> Graph)
    starting from the SAME initial condition x0, and animates all three
    networks side by side, frame-synchronized.
    """
    names = list(graphs.keys())

    # One model per topology, all sharing the same x0, delta, a
    models = {name: DiffusionModel(g, x0, delta=delta, a=a) for name, g in graphs.items()}
    histories = {name: models[name].run(T) for name in names}

    # Fixed layout per graph (seeded for reproducibility)
    positions = {name: nx.spring_layout(g, seed=seed) for name, g in graphs.items()}

    fig, axes = plt.subplots(1, len(names), figsize=figsize)
    if len(names) == 1:
        axes = [axes]

    node_collections = {}
    for ax, name in zip(axes, names):
        g = graphs[name]
        pos = positions[name]
        hist = histories[name]

        nodes = nx.draw_networkx_nodes(
            g, pos,
            node_color=hist[0],
            cmap=cmap, vmin=vmin, vmax=vmax,
            node_size=node_size, ax=ax,
        )
        nx.draw_networkx_edges(g, pos, alpha=0.3, ax=ax)
        ax.set_title(f"{name}\n(t=0)")
        ax.set_axis_off()
        node_collections[name] = nodes

    def update(frame):
        artists = []
        for ax, name in zip(axes, names):
            nodes = node_collections[name]
            nodes.set_array(histories[name][frame])
            ax.set_title(f"{name}\n(t={frame})")
            artists.append(nodes)
        return artists

    anim = FuncAnimation(
        fig, update, frames=len(next(iter(histories.values()))),
        interval=interval, blit=False,
    )
    fig.suptitle("Consensus dynamics under different network topologies (N=20)")
    plt.tight_layout()
    plt.show()
    print("Enregistrement du GIF en cours... Cela peut prendre quelques secondes.")
    anim.save("consensus_3_topologies.gif", writer="pillow", fps=10)
    print("GIF enregistré avec succès sous le nom 'consensus_3_topologies.gif' !")
    
    fig.suptitle("Consensus dynamics under different network topologies (N=20)")
    plt.tight_layout()
    plt.show()
    return anim, histories
    return anim, histories