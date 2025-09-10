import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import imageio


import opinionated
import matplotlib.pyplot as plt
plt.style.use("opinionated_rc")
import colormaps as cmaps 

# Default constants
FIGSIZE = (10, 8)
DPI = 100
WIDTH = 10
HEIGHT = 8
EDGE_WIDTH = 0.6
LONG_EDGE_WIDTH = 1.5
NODE_SIZE = 20
# Slow down animations significantly
GIF_FRAME_SEC = 1.

# Color palette
COLOR_EDGE = "#bfbfbf"          # slightly darker light grey for all lines
COLOR_LONG_EDGE = "#bcbcbc"     # slightly darker subtle highlight for long ties
COLOR_NODE_BASE = "#767676"     # basic nodes (updated)
COLOR_NODE_VISITED = "#ff7f0e"  # information orange
COLOR_NODE_FRONTIER = "#ab0b00" # progress color (was green)
COLOR_TEXT = "#333333"          # UI text (e.g., counter)

# Layout/margins (small, consistent)
MARGIN_FRAC = 0.02
PAD_INCHES = 0.08

# Background color
BG_COLOR = "#f3f3f3"

# Initialize random number generator
rng = np.random.default_rng()

# ---------- Utility functions ----------
def random_geometric_radius(n, area, alpha=1.0):
    """
    Connectivity-ish threshold for a random geometric graph on area A ~ sqrt((log n)/(pi*n))*sqrt(A).
    We scale by alpha (>1) to be safely connected and keep hop distances moderate.
    """
    r_unit = math.sqrt(max(1e-9, (math.log(n)) / (math.pi * n)))
    return alpha * r_unit * math.sqrt(area)

def make_rgg(n, width=WIDTH, height=HEIGHT, alpha=1.5):
    # Sample positions
    xs = rng.random(n) * width
    ys = rng.random(n) * height
    pos = {i: (xs[i], ys[i]) for i in range(n)}

    # Build edges by threshold radius
    r = random_geometric_radius(n, width*height, alpha=alpha)
    r2 = r*r
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i+1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx*dx + dy*dy <= r2:
                G.add_edge(i, j)

    # Keep the largest connected component
    components = list(nx.connected_components(G))
    components.sort(key=len, reverse=True)
    if len(components) > 1:
        G = G.subgraph(components[0]).copy()
        pos = {i: pos[i] for i in G.nodes()}
    return G, pos, r

def corner_nodes(pos, width=WIDTH, height=HEIGHT):
    """Return node near lower-left and node near upper-right corner."""
    ll = min(pos, key=lambda i: (pos[i][0]**2 + pos[i][1]**2))
    ur = min(pos, key=lambda i: ((pos[i][0]-width)**2 + (pos[i][1]-height)**2))
    return ll, ur

def draw_graph_image(G, pos, title, save_path, long_edges=set(), highlight_path=None, 
                     figsize=FIGSIZE, dpi=DPI, width=WIDTH, height=HEIGHT, 
                     edge_width=EDGE_WIDTH, long_edge_width=LONG_EDGE_WIDTH, node_size=NODE_SIZE):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    # Small, consistent outer margins
    fig.patch.set_facecolor(BG_COLOR)
    fig.subplots_adjust(
        left=MARGIN_FRAC,
        right=1 - MARGIN_FRAC,
        top=1 - MARGIN_FRAC,
        bottom=MARGIN_FRAC,
    )
    ax.set_facecolor(BG_COLOR)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis('off')

    # Base edges
    base_edges = [(u, v) for (u, v) in G.edges() if (u, v) not in long_edges and (v, u) not in long_edges]
    if base_edges:
        base_segs = [[pos[u], pos[v]] for (u, v) in base_edges]
        lc = LineCollection(base_segs, linewidths=edge_width, colors=COLOR_EDGE, alpha=0.7, zorder=1)
        ax.add_collection(lc)

    # Long ties drawn thicker and visible in static fig (fig 2)
    if long_edges:
        long_segs = [[pos[u], pos[v]] for (u, v) in long_edges]
        lc2 = LineCollection(long_segs, linewidths=max(long_edge_width * 1.8, 2.5),
                             colors=COLOR_NODE_VISITED, alpha=0.5, zorder=1)
        ax.add_collection(lc2)

    # Optional highlighted path lines (still behind nodes)
    if highlight_path is not None and len(highlight_path) > 1:
        path_segs = []
        for a, b in zip(highlight_path[:-1], highlight_path[1:]):
            path_segs.append([pos[a], pos[b]])
        lc_p = LineCollection(path_segs, linewidths=edge_width * 1.5, colors=COLOR_LONG_EDGE, alpha=0.6, zorder=1)
        ax.add_collection(lc_p)

    # Nodes (always above lines)
    xs = [pos[i][0] for i in G.nodes()]
    ys = [pos[i][1] for i in G.nodes()]
    ax.scatter(xs, ys, s=node_size, alpha=0.95, color=COLOR_NODE_BASE, zorder=3)

    # Optional highlighted path nodes (above base nodes)
    if highlight_path is not None and len(highlight_path) > 1:
        ax.scatter([pos[i][0] for i in highlight_path], [pos[i][1] for i in highlight_path],
                   s=node_size * 2.0, alpha=1.0, color=COLOR_NODE_VISITED, zorder=4)

    # Remove headings/titles per spec
    fig.savefig(save_path, bbox_inches='tight', pad_inches=PAD_INCHES, facecolor=BG_COLOR)
    plt.close(fig)

def bfs_layers(G, source):
    dist = {source: 0}
    layers = [[source]]
    visited = {source}
    frontier = [source]
    d = 0
    while frontier:
        next_frontier = []
        for u in frontier:
            for v in G.neighbors(u):
                if v not in visited:
                    visited.add(v)
                    dist[v] = d + 1
                    next_frontier.append(v)
        if not next_frontier:
            break
        layers.append(next_frontier)
        frontier = next_frontier
        d += 1
    return dist, layers

def animate_signal(G, pos, source, target, save_path, long_edges=set(), 
                   figsize=FIGSIZE, dpi=DPI, width=WIDTH, height=HEIGHT,
                   edge_width=EDGE_WIDTH, long_edge_width=LONG_EDGE_WIDTH, 
                   node_size=NODE_SIZE, gif_frame_sec=GIF_FRAME_SEC):
    # BFS from source to get layers
    dist, layers = bfs_layers(G, source)
    if target not in dist:
        end_layer = len(layers) - 1
    else:
        end_layer = dist[target]

    # If reachable, precompute actual shortest path from source to target
    shortest_path_nodes = None
    if target in dist:
        try:
            shortest_path_nodes = nx.shortest_path(G, source=source, target=target)
        except Exception:
            shortest_path_nodes = None

    # Precompute drawings
    base_edges = [(u, v) for (u, v) in G.edges() if (u, v) not in long_edges and (v, u) not in long_edges]
    base_segs = [[pos[u], pos[v]] for (u, v) in base_edges]
    long_segs = [[pos[u], pos[v]] for (u, v) in long_edges]
    xs_all = np.array([pos[i][0] for i in G.nodes()])
    ys_all = np.array([pos[i][1] for i in G.nodes()])

    frames = []
    for t in range(end_layer + 1):
        visited = set().union(*layers[:t+1]) if t < len(layers) else set().union(*layers)
        frontier = set(layers[t]) if t < len(layers) else set()

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        # Small, consistent outer margins for GIF frames
        fig.patch.set_facecolor(BG_COLOR)
        fig.subplots_adjust(
            left=MARGIN_FRAC,
            right=1 - MARGIN_FRAC,
            top=1 - MARGIN_FRAC,
            bottom=MARGIN_FRAC,
        )
        ax.set_facecolor(BG_COLOR)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.axis('off')

        # Edges (always behind nodes)
        if base_segs:
            lc = LineCollection(base_segs, linewidths=edge_width, colors=COLOR_EDGE, alpha=0.7, zorder=1)
            ax.add_collection(lc)
        if long_segs:
            lc2 = LineCollection(long_segs, linewidths=long_edge_width, colors=COLOR_LONG_EDGE, alpha=0.4, zorder=1)
            ax.add_collection(lc2)

        # Nodes: base (unvisited)
        unvisited_mask = np.ones(len(xs_all), dtype=bool)
        for i, node in enumerate(G.nodes()):
            if node in visited:
                unvisited_mask[i] = False
        ax.scatter(xs_all[unvisited_mask], ys_all[unvisited_mask], s=node_size, alpha=0.95,
                   color=COLOR_NODE_BASE, zorder=2)

        # Visited (but not current frontier)
        visited_nodes = [n for n in visited if n not in frontier]
        if visited_nodes:
            ax.scatter([pos[i][0] for i in visited_nodes], [pos[i][1] for i in visited_nodes],
                       s=node_size*1.2, alpha=0.95, color=COLOR_NODE_VISITED, zorder=3)

        # Frontier
        if frontier:
            ax.scatter([pos[i][0] for i in frontier], [pos[i][1] for i in frontier],
                       s=node_size*3.0, alpha=1.0, color=COLOR_NODE_FRONTIER, zorder=4)

        # Source and target emphasis
        sx, sy = pos[source]
        tx, ty = pos[target]
        ax.scatter([sx], [sy], s=60, marker='s', color=COLOR_NODE_VISITED, zorder=5)
        ax.scatter([tx], [ty], s=60, marker='^', color=COLOR_NODE_FRONTIER, zorder=5)

        # On the final frame, draw the actual path that was taken
        if t == end_layer and shortest_path_nodes is not None and len(shortest_path_nodes) > 1:
            path_segs = []
            for a, b in zip(shortest_path_nodes[:-1], shortest_path_nodes[1:]):
                path_segs.append([pos[a], pos[b]])
            lc_path = LineCollection(path_segs, linewidths=max(edge_width * 2.0, 2.5),
                                     colors=COLOR_NODE_VISITED, alpha=0.95, zorder=2)
            ax.add_collection(lc_path)
            ax.scatter([pos[i][0] for i in shortest_path_nodes], [pos[i][1] for i in shortest_path_nodes],
                       s=node_size * 2.0, alpha=1.0, color=COLOR_NODE_VISITED, zorder=4.5)

        # Bottom-left counter instead of title
        ax.text(0.02, 0.02, f"{t}/{end_layer}", transform=ax.transAxes,
                ha='left', va='bottom', fontsize=28, fontweight='bold', color=COLOR_TEXT, zorder=6)

        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
        frames.append(frame)
        plt.close(fig)

    imageio.mimsave(save_path, frames,fps=0.6)

def add_long_ties(G, pos, k, min_frac_of_diag=0.6, width=WIDTH, height=HEIGHT):
    """Add k long-range edges between far-apart nodes that are not already connected."""
    diag = math.hypot(width, height)
    cand = []
    nodes = list(G.nodes())
    for _ in range(k*20):  # oversample attempts
        u, v = rng.choice(nodes, size=2, replace=False)
        if G.has_edge(u, v):
            continue
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        d = math.hypot(x1-x2, y1-y2)
        if d >= min_frac_of_diag * diag:
            cand.append((u, v, d))
    # Keep unique pairs and select up to k by distance (prefer longer)
    seen = set()
    uniq = []
    for u, v, d in sorted(cand, key=lambda t: -t[2]):
        if (u, v) in seen or (v, u) in seen:
            continue
        seen.add((u, v))
        uniq.append((u, v))
        if len(uniq) >= k:
            break

    for u, v in uniq:
        G.add_edge(u, v)
    return set(uniq)
