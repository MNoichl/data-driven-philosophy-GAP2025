
"""
net_animations.py
Utilities to animate ER and BA graph construction and to plot degree distributions.

Requirements:
- networkx
- numpy
- matplotlib
- imageio (preferred) or Pillow (fallback)

Colors requested by user:
"""
from __future__ import annotations

import math
import io
from typing import Dict, List, Tuple, Optional

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.style.use("opinionated_rc")
import colormaps as cmaps 



# Preferred GIF writer
try:
    import imageio.v2 as imageio
    _HAS_IMAGEIO = True
except Exception:
    _HAS_IMAGEIO = False

try:
    from PIL import Image
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

COLOR_EDGE = "#bfbfbf"     # slightly darker light grey for all lines
COLOR_NODE = "#767676"     # basic nodes (updated)
COLOR_POINT = "#333333"    # dark gray for scatter points
COLOR_ER_THEO = "#ff7f0e"  # requested ER theoretical color
COLOR_BA_THEO = "#ab0b00"  # requested BA theoretical color

# -----------------------------
# Internal helpers
# -----------------------------

def _graph_frame_array(G: nx.Graph,
                       pos: Dict,
                       figsize: Tuple[float,float] = (4.8, 4.8),
                       node_size: float = 40.0,
                       edge_width: float = 1.0,
                       dpi: int = 300,
                       title: Optional[str] = None) -> np.ndarray:
    """Render a graph with given positions into an RGB numpy array."""
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.set_aspect('equal', adjustable='datalim')
    try:
        ax.set_box_aspect(1)
    except Exception:
        pass

    # Draw edges first
    if G.number_of_edges() > 0:
        nx.draw_networkx_edges(
            G, pos,
            ax=ax,
            width=edge_width,
            edge_color=COLOR_EDGE
        )

    # Draw nodes
    if G.number_of_nodes() > 0:
        nx.draw_networkx_nodes(
            G, pos,
            ax=ax,
            node_size=node_size,
            node_color="#000000"
        )

    if title:
        ax.set_title(title, fontsize=11, pad=6)

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((height, width, 4))
    frame = frame[:, :, :3]  # Convert RGBA to RGB by dropping alpha channel
    plt.close(fig)
    return frame


def _save_gif(frames: List[np.ndarray], out_path: str, fps: int = 5) -> None:
    """Save frames to a GIF using imageio if available, else Pillow."""
    if len(frames) == 0:
        raise ValueError("No frames to save.")
    duration = 1.0 / max(1, fps)

    if _HAS_IMAGEIO:
        imageio.mimsave(out_path, frames, duration=duration)
        return

    if not _HAS_PIL:
        raise RuntimeError("Neither imageio nor Pillow is available to save GIFs.")

    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(out_path, save_all=True, append_images=pil_frames[1:], duration=int(duration*1000), loop=0)


def _iterative_layout_update(G: nx.Graph,
                             pos: Dict[int, np.ndarray],
                             iterations: int = 10,
                             k: Optional[float] = None,
                             seed: Optional[int] = None) -> Dict[int, np.ndarray]:
    """Update spring layout positions a bit to create motion across frames."""
    # Use small number of iterations for smooth motion
    pos = nx.spring_layout(G, pos=pos, iterations=iterations, k=k, seed=seed)
    return pos


def _initial_positions(n: int, rng: np.random.Generator) -> Dict[int, np.ndarray]:
    """Random initial positions in [0,1]^2 for stability before edges exist."""
    return {i: rng.random(2) for i in range(n)}

# -----------------------------
# Theoretical distributions
# -----------------------------

def _integer_log_ticks(min_value: float, max_value: float, max_ticks: int = 6) -> List[int]:
    """Generate up to max_ticks integer tick positions between min and max on a log scale."""
    min_value = max(1, int(math.floor(min_value)))
    max_value = max(min_value, int(math.ceil(max_value)))
    if min_value == max_value:
        return [min_value]
    # Use log-spaced candidates then round to integers and deduplicate
    raw = np.logspace(math.log10(min_value), math.log10(max_value), num=max_ticks)
    ticks = sorted(set(int(round(v)) for v in raw))
    # Ensure endpoints
    if ticks[0] != min_value:
        ticks.insert(0, min_value)
    if ticks[-1] != max_value:
        ticks.append(max_value)
    # Remove any zeros or duplicates created by rounding
    ticks = [t for t in ticks if t >= 1]
    dedup = []
    for t in ticks:
        if not dedup or dedup[-1] != t:
            dedup.append(t)
    return dedup

def _binomial_pmf_array(num_trials: int, prob: float, k_max: int) -> np.ndarray:
    """Return pmf values for Binomial(num_trials, prob) for k = 0..k_max."""
    pmf = np.zeros(k_max + 1, dtype=float)
    for k in range(0, k_max + 1):
        if 0 <= k <= num_trials:
            pmf[k] = math.comb(num_trials, k) * (prob ** k) * ((1.0 - prob) ** (num_trials - k))
        else:
            pmf[k] = 0.0
    return pmf

def _ba_degree_pmf(m: int, k_values: np.ndarray) -> np.ndarray:
    """Exact BA stationary pmf: P(k) = 2 m (m+1) / [k (k+1) (k+2)] for k >= m."""
    k = np.asarray(k_values, dtype=float)
    pmf = np.zeros_like(k, dtype=float)
    mask = k >= m
    km = k[mask]
    pmf[mask] = (2.0 * m * (m + 1.0)) / (km * (km + 1.0) * (km + 2.0))
    return pmf

def _ba_ccdf_counts(n_nodes: int, m: int, k_values: np.ndarray) -> np.ndarray:
    """Closed-form BA CCDF counts: n * m (m+1) / [k (k+1)] for k >= m, else ~ n."""
    k = np.asarray(k_values, dtype=float)
    ccdf = np.empty_like(k, dtype=float)
    below = k < m
    ccdf[below] = float(n_nodes)
    abv = ~below
    ka = k[abv]
    ccdf[abv] = n_nodes * (m * (m + 1.0)) / (ka * (ka + 1.0))
    return ccdf

def _ba_ccdf_powerlaw_counts(n_nodes: int, m: int, k_values: np.ndarray) -> np.ndarray:
    """Pure power-law BA CCDF counts: n * C_ccdf * k^{-2}, with C_ccdf = m (m+1)."""
    k = np.asarray(k_values, dtype=float)
    k = np.maximum(k, 1.0)
    c_ccdf = m * (m + 1.0)
    return n_nodes * c_ccdf * (k ** -2.0)

def _er_ccdf_counts(n_nodes: int, n_trials: int, prob: float, k_values: np.ndarray) -> np.ndarray:
    """Exact ER Binomial-tail counts: n * P(K >= k) with K ~ Bin(n_trials, prob)."""
    # Precompute pmf for all degrees up to n_trials
    full_pmf = _binomial_pmf_array(n_trials, prob, n_trials)
    # Tail sums for requested k values
    ccdf_probs = []
    for k in k_values:
        kk = int(max(0, min(n_trials, int(k))))
        ccdf_probs.append(float(full_pmf[kk:].sum()))
    ccdf_probs = np.asarray(ccdf_probs, dtype=float)
    return n_nodes * ccdf_probs

# -----------------------------
# Public API
# -----------------------------

def animate_er_gif(n: int = 60,
                   p: float = 0.05,
                   out_path: str = "er_construct.gif",
                   seed: Optional[int] = 42,
                   fps: int = 5,
                   layout_iter: int = 8,
                   node_size: float = 40.0,
                   edge_width: float = 1.0,
                   dpi: int = 300,
                   show_titles: bool = False) -> nx.Graph:
    """
    Animate construction of an Erdős–Rényi G(n,p) graph.
    Returns the final NetworkX graph.
    """
    rng = np.random.default_rng(seed)

    # Decide final edge set
    edges = []
    for i in range(n-1):
        for j in range(i+1, n):
            if rng.random() < p:
                edges.append((i, j))

    rng.shuffle(edges)

    # Build incrementally
    G = nx.Graph()
    G.add_nodes_from(range(n))
    pos = _initial_positions(n, rng)

    frames = []
    k = 1.0 / math.sqrt(n)  # spring length parameter

    # Initial empty graph frame
    frames.append(_graph_frame_array(
        G,
        pos,
        node_size=node_size,
        edge_width=edge_width,
        dpi=dpi,
        title="Erdős–Rényi (ER) construction: 0 edges" if show_titles else None,
    ))

    for t, (u, v) in enumerate(edges, start=1):
        G.add_edge(u, v)
        pos = _iterative_layout_update(G, pos, iterations=layout_iter, k=k, seed=seed)
        title = (
            f"Erdős–Rényi (ER) construction: {t}/{len(edges)} edges" if show_titles else None
        )
        frames.append(_graph_frame_array(G, pos, node_size=node_size, edge_width=edge_width, dpi=dpi, title=title))

    _save_gif(frames, out_path, fps=fps)
    # Tag with model metadata for plotting overlays
    G.graph["model"] = "ER"
    G.graph["n"] = n
    G.graph["p"] = p
    G.graph["seed"] = seed
    return G


def animate_ba_gif(n: int = 60,
                   m: int = 2,
                   out_path: str = "ba_construct.gif",
                   seed: Optional[int] = 42,
                   fps: int = 5,
                   layout_iter: int = 8,
                   node_size: float = 40.0,
                   edge_width: float = 1.0,
                   dpi: int = 300,
                   show_titles: bool = False) -> nx.Graph:
    """
    Animate construction of a Barabási–Albert preferential attachment graph.
    Returns the final NetworkX graph.
    """
    if m < 1:
        raise ValueError("m must be at least 1.")
    if n <= m:
        raise ValueError("n must be greater than m.")

    rng = np.random.default_rng(seed)

    # Start with a clique of size m0 = m
    m0 = max(m, 2)
    G = nx.complete_graph(m0)
    next_node = m0

    # Degree list to compute attachment probabilities
    degrees = np.array([G.degree(i) for i in range(next_node)], dtype=float)  # size m0

    # Positions
    pos = _initial_positions(n, rng)
    k = 1.0 / math.sqrt(n)

    frames = []
    frames.append(
        _graph_frame_array(
            G,
            {i: pos[i] for i in G.nodes},
            node_size=node_size,
            edge_width=edge_width,
            dpi=dpi,
            title="Barabási–Albert (BA) construction: seed" if show_titles else None,
        )
    )

    # Grow the network
    while next_node < n:
        # Compute attachment probabilities for existing nodes
        deg_sum = degrees.sum()
        if deg_sum == 0:
            probs = np.ones_like(degrees) / len(degrees)
        else:
            probs = degrees / deg_sum

        # Sample m distinct targets without replacement (weighted)
        targets = set()
        # Sampling without replacement with changing probs; do sequential draws
        while len(targets) < m:
            pick = rng.choice(len(degrees), p=probs)
            targets.add(int(pick))

        G.add_node(next_node)
        for t in targets:
            G.add_edge(next_node, int(t))

        # Update degrees (expand array)
        degrees = np.append(degrees, 0.0)
        degrees[list(targets)] += 1.0
        degrees[-1] = float(m)

        # Ensure the new node has an initial position
        if next_node not in pos:
            pos[next_node] = np.random.default_rng(seed).random(2)
        # Update positions using the current subgraph's positions
        pos = _iterative_layout_update(G, {i: pos[i] for i in G.nodes}, iterations=layout_iter, k=k, seed=seed)

        title = (
            f"Barabási–Albert (BA) construction: {G.number_of_nodes()}/{n} nodes"
            if show_titles
            else None
        )
        frames.append(_graph_frame_array(G, pos, node_size=node_size, edge_width=edge_width, dpi=dpi, title=title))

        next_node += 1

    # Save
    _save_gif(frames, out_path, fps=fps)
    # Tag with model metadata for plotting overlays
    G.graph["model"] = "BA"
    G.graph["n"] = n
    G.graph["m"] = m
    G.graph["seed"] = seed
    return G


def plot_degree_histogram(G: nx.Graph,
                          out_path: str,
                          title: Optional[str] = None,
                          dpi: int = 300) -> None:
    """Plot degree histogram on linear axes (one figure)."""
    degrees = [d for _, d in G.degree()]
    max_k = max(degrees) if degrees else 0
    bins = np.arange(-0.5, max_k + 1.5, 1.0)  # integer bins centered on ints

    fig = plt.figure(figsize=(4.8, 4.8), dpi=dpi)
    ax = fig.add_subplot(111)
    # If BA, drop degree 0 from histogram
    model = G.graph.get("model", None)
    if model == "BA":
        degrees = [d for d in degrees if d > 0]
        max_k = max(degrees) if degrees else 0
        bins = np.arange(-0.5, max_k + 1.5, 1.0)
    counts, _, _ = ax.hist(degrees, bins=bins, edgecolor="white", color=COLOR_NODE)
    ax.set_xlabel("Degree k", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    if title:
        ax.set_title(title, fontsize=13, pad=6)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    # Fixed integer ticks for ER histogram x-axis, if in range
    if model == "ER":
        fixed_ticks = [1, 5, 10, 50, 100]
        max_tick = int(max_k) if max_k is not None else None
        if max_tick is not None and max_tick >= 1:
            ticks_in_range = [t for t in fixed_ticks if 1 <= t <= max_tick]
            if ticks_in_range:
                ax.set_xticks(ticks_in_range)
                ax.set_xticklabels([str(t) for t in ticks_in_range])
    try:
        ax.set_box_aspect(1)
    except Exception:
        pass

    # Overlay theoretical distributions as lines (scaled to counts)
    n_nodes = G.graph.get("n", G.number_of_nodes())
    centers = 0.5 * (bins[:-1] + bins[1:])
    if model == "ER":
        p = G.graph.get("p", None)
        if p is not None:
            pmf = _binomial_pmf_array(n_nodes - 1, p, int(max_k))
            expected_counts = n_nodes * pmf[: len(centers)]
            print(f"[ER theoretical] n={n_nodes}, p={p}")
            ax.plot(centers, expected_counts, color=COLOR_ER_THEO, linewidth=2, label="ER theoretical")
    elif model == "BA":
        m = G.graph.get("m", None)
        if m is not None:
            k_vals = np.arange(0, int(max_k) + 1)
            pmf = _ba_degree_pmf(m, k_vals)
            expected_counts = n_nodes * pmf[: len(centers)]
            expected_counts = expected_counts[1:] # remove 0 value
            centers = centers[1:]
            print(f"[BA theoretical] n={n_nodes}, m={m}")
            ax.plot(centers, expected_counts, color=COLOR_BA_THEO, linewidth=2, label="BA theoretical")
    if model in {"ER", "BA"}:
        ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_degree_loglog_ccdf(G: nx.Graph,
                            out_path: str,
                            title: Optional[str] = None,
                            dpi: int = 300) -> None:
    """
    Plot the complementary CDF (P(K >= k)) of degrees on log–log axes.
    This is more stable in the tail than a log-binned histogram.
    """
    degrees = np.array([d for _, d in G.degree()], dtype=float)
    if degrees.size == 0:
        raise ValueError("Graph has no nodes.")
    ks = np.unique(degrees)
    ks.sort()
    # Remove zero for log-x
    ks = ks[ks >= 1]
    # CCDF as counts, so y shares tick marks (1,10,100,...)
    ccdf_counts = np.array([(degrees >= k).sum() for k in ks], dtype=float)

    fig = plt.figure(figsize=(4.8, 4.8), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.scatter(ks, ccdf_counts, s=18, color=COLOR_POINT)
    ax.set_xlabel("Degree k", fontsize=12)
    ax.set_ylabel("Nodes with degree ≥ k (count)", fontsize=12)
    if title:
        ax.set_title(title, fontsize=13, pad=6)
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.6)
    # Square axes area
    try:
        ax.set_box_aspect(1)
    except Exception:
        pass
    # Fixed x and y ticks within data range using fixed locators/formatters; disable minor ticks
    fixed_ticks = [1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100]
    if ks.size > 0:
        xmin, xmax = ks.min(), ks.max()
        x_ticks_in_range = [t for t in fixed_ticks if xmin <= t <= xmax]
        if x_ticks_in_range:
            ax.xaxis.set_major_locator(mticker.FixedLocator(x_ticks_in_range))
            ax.xaxis.set_major_formatter(mticker.FixedFormatter([str(x) for x in x_ticks_in_range]))
            ax.xaxis.set_minor_locator(mticker.NullLocator())
            ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    if ccdf_counts.size > 0:
        ymin, ymax = max(1.0, float(ccdf_counts.min())), float(ccdf_counts.max())
        y_ticks_in_range = [t for t in fixed_ticks if ymin <= t <= ymax]
        if y_ticks_in_range:
            ax.yaxis.set_major_locator(mticker.FixedLocator(y_ticks_in_range))
            ax.yaxis.set_major_formatter(mticker.FixedFormatter([str(y) for y in y_ticks_in_range]))
            ax.yaxis.set_minor_locator(mticker.NullLocator())
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    # Overlay theoretical CCDF counts
    n_nodes = G.graph.get("n", G.number_of_nodes())
    model = G.graph.get("model", None)
    if model == "ER":
        p = G.graph.get("p", None)
        if p is not None:
            theo_counts = _er_ccdf_counts(n_nodes, n_nodes - 1, p, ks)
            print(f"[ER theoretical] n={n_nodes}, p={p}")
            ax.plot(ks, theo_counts, color=COLOR_ER_THEO, linewidth=2, label="ER theoretical")
    elif model == "BA":
        m = G.graph.get("m", None)
        if m is not None:
            print(f"[BA theoretical] n={n_nodes}, m={m}")
            print(f"[BA power-law] alpha=3, using CCDF ~ k^-2 with C_ccdf={m*(m+1)}")
            theo_counts = _ba_ccdf_powerlaw_counts(n_nodes, m, ks)
            ax.plot(ks, theo_counts, color=COLOR_BA_THEO, linewidth=2, label="BA theoretical")
    if model in {"ER", "BA"}:
        ax.legend(frameon=False, fontsize=10, loc='upper right')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_ba_er_distributions_side_by_side(
    G_ba: nx.Graph,
    G_er: nx.Graph,
    out_path_hist: str = "ba_er_hist.png",
    out_path_ccdf: str = "ba_er_ccdf.png",
    dpi: int = 300,
    ba_label: str = "BA",
    er_label: str = "ER",
) -> None:
    """Create side-by-side plots for BA vs ER degree distributions.

    Saves two figures:
      - Histogram (linear axes): out_path_hist
      - CCDF (log–log): out_path_ccdf
    """
    # Histogram side-by-side
    degrees_ba = [d for _, d in G_ba.degree()]
    # Remove 0-degree from BA histogram as requested
    degrees_ba = [d for d in degrees_ba if d > 0]
    degrees_er = [d for _, d in G_er.degree()]
    max_k = max(max(degrees_ba) if degrees_ba else 0, max(degrees_er) if degrees_er else 0)
    bins = np.arange(-0.5, max_k + 1.5, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=dpi, constrained_layout=True)
    ax1, ax2 = axes
    counts_ba, _, _ = ax1.hist(degrees_ba, bins=bins, edgecolor="white", color=COLOR_NODE)
    ax1.set_title(f"{ba_label} degree histogram", fontsize=13, pad=6)
    ax1.set_xlabel("Degree k", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    try:
        ax1.set_box_aspect(1)
    except Exception:
        pass

    counts_er, _, _ = ax2.hist(degrees_er, bins=bins, edgecolor="white", color=COLOR_NODE)
    ax2.set_title(f"{er_label} degree histogram", fontsize=13, pad=6)
    ax2.set_xlabel("Degree k", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    # Fixed integer ticks for ER histogram x-axis, if in range
    fixed_ticks = [1, 2,4,6,8, 10, 20, 40,60 , 80, 100]
    max_tick = int(max_k) if max_k is not None else None
    if max_tick is not None and max_tick >= 1:
        ticks_in_range = [t for t in fixed_ticks if 1 <= t <= max_tick]
        if ticks_in_range:
            ax2.set_xticks(ticks_in_range)
            ax2.set_xticklabels([str(t) for t in ticks_in_range])
    try:
        ax2.set_box_aspect(1)
    except Exception:
        pass

    # Overlay theoretical lines on histograms
    n_ba = G_ba.graph.get("n", G_ba.number_of_nodes())
    n_er = G_er.graph.get("n", G_er.number_of_nodes())
    centers = 0.5 * (bins[:-1] + bins[1:])
    m_ba = G_ba.graph.get("m", None)
    if m_ba is not None:
        k_vals = np.arange(0, int(max_k) + 1)
        pmf_ba = _ba_degree_pmf(m_ba, k_vals)
        print(f"[BA theoretical] n={n_ba}, m={m_ba}")
        print(f"[BA power-law] alpha=3, C_pmf={2*m_ba*(m_ba+1)}, C_ccdf={m_ba*(m_ba+1)}")
        expected_counts_ba = n_ba * pmf_ba[: len(centers)]
        # Align with histogram excluding degree 0 by dropping the first bin
        centers_ba = centers[1:]
        expected_counts_ba = expected_counts_ba[1:]
        if centers_ba.size > 0 and expected_counts_ba.size > 0:
            ax1.plot(centers_ba, expected_counts_ba, color=COLOR_BA_THEO, linewidth=2, label=f"{ba_label} theoretical")
            ax1.legend(frameon=False, fontsize=10)
    p_er = G_er.graph.get("p", None)
    if p_er is not None:
        pmf_er = _binomial_pmf_array(n_er - 1, p_er, int(max_k))
        print(f"[ER theoretical] n={n_er}, p={p_er}")
        ax2.plot(centers, n_er * pmf_er[: len(centers)], color=COLOR_ER_THEO, linewidth=2, label=f"{er_label} theoretical")
        ax2.legend(frameon=False, fontsize=10)

    fig.savefig(out_path_hist)
    plt.close(fig)

    # CCDF side-by-side
    def _ccdf_counts_vals(G: nx.Graph) -> Tuple[np.ndarray, np.ndarray]:
        degs = np.array([d for _, d in G.degree()], dtype=float)
        if degs.size == 0:
            return np.array([1.0]), np.array([1.0])
        ks_local = np.unique(degs)
        ks_local.sort()
        ks_local = ks_local[ks_local >= 1]
        ccdf_local = np.array([(degs >= k).sum() for k in ks_local], dtype=float)
        return ks_local, ccdf_local

    ks_ba, ccdf_ba = _ccdf_counts_vals(G_ba)
    ks_er, ccdf_er = _ccdf_counts_vals(G_er)


    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=dpi, constrained_layout=True)
    ax1, ax2 = axes
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.scatter(ks_ba, ccdf_ba, s=18, color=COLOR_POINT)
    ax1.set_title(f"{ba_label} degree CCDF", fontsize=13, pad=6)
    ax1.set_xlabel("Degree k", fontsize=12)
    ax1.set_ylabel("Nodes with degree ≥ k (count)", fontsize=12)
    ax1.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.6)
    try:
        ax1.set_box_aspect(1)
    except Exception:
        pass
    fixed_ticks = [1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100]
    if ks_ba.size > 0:
        xmin, xmax = ks_ba.min(), ks_ba.max()
        x_ticks_in_range = [t for t in fixed_ticks if xmin <= t <= xmax]
        if x_ticks_in_range:
            ax1.xaxis.set_major_locator(mticker.FixedLocator(x_ticks_in_range))
            ax1.xaxis.set_major_formatter(mticker.FixedFormatter([str(x) for x in x_ticks_in_range]))
            ax1.xaxis.set_minor_locator(mticker.NullLocator())
            ax1.xaxis.set_minor_formatter(mticker.NullFormatter())
    if ccdf_ba.size > 0:
        ymin, ymax = max(1.0, float(ccdf_ba.min())), float(ccdf_ba.max())
        y_ticks_in_range = [t for t in fixed_ticks if ymin <= t <= ymax]
        if y_ticks_in_range:
            ax1.yaxis.set_major_locator(mticker.FixedLocator(y_ticks_in_range))
            ax1.yaxis.set_major_formatter(mticker.FixedFormatter([str(y) for y in y_ticks_in_range]))
            ax1.yaxis.set_minor_locator(mticker.NullLocator())
            ax1.yaxis.set_minor_formatter(mticker.NullFormatter())

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.scatter(ks_er, ccdf_er, s=18, color=COLOR_POINT)
    ax2.set_title(f"{er_label} degree CCDF", fontsize=13, pad=6)
    ax2.set_xlabel("Degree k", fontsize=12)
    ax2.set_ylabel("Nodes with degree ≥ k (count)", fontsize=12)
    ax2.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.6)
    try:
        ax2.set_box_aspect(1)
    except Exception:
        pass
    if ks_er.size > 0:
        xmin, xmax = ks_er.min(), ks_er.max()
        x_ticks_in_range = [t for t in fixed_ticks if xmin <= t <= xmax]
        if x_ticks_in_range:
            ax2.xaxis.set_major_locator(mticker.FixedLocator(x_ticks_in_range))
            ax2.xaxis.set_major_formatter(mticker.FixedFormatter([str(x) for x in x_ticks_in_range]))
            ax2.xaxis.set_minor_locator(mticker.NullLocator())
            ax2.xaxis.set_minor_formatter(mticker.NullFormatter())
    if ccdf_er.size > 0:
        ymin, ymax = max(1.0, float(ccdf_er.min())), float(ccdf_er.max())
        y_ticks_in_range = [t for t in fixed_ticks if ymin <= t <= ymax]
        if y_ticks_in_range:
            ax2.yaxis.set_major_locator(mticker.FixedLocator(y_ticks_in_range))
            ax2.yaxis.set_major_formatter(mticker.FixedFormatter([str(y) for y in y_ticks_in_range]))
            ax2.yaxis.set_minor_locator(mticker.NullLocator())
            ax2.yaxis.set_minor_formatter(mticker.NullFormatter())

    # Overlay theoretical CCDFs as lines
    n_ba = G_ba.graph.get("n", G_ba.number_of_nodes())
    n_er = G_er.graph.get("n", G_er.number_of_nodes())
    m_ba = G_ba.graph.get("m", None)
    if m_ba is not None:
        theo_ba = _ba_ccdf_powerlaw_counts(n_ba, m_ba, ks_ba)
        print(f"[BA theoretical] n={n_ba}, m={m_ba}")
        ax1.plot(ks_ba, theo_ba, color=COLOR_BA_THEO, linewidth=2, label=f"{ba_label} theoretical")
        ax1.legend(frameon=False, fontsize=10, loc='upper right')
    p_er = G_er.graph.get("p", None)
    if p_er is not None:
        theo_er = _er_ccdf_counts(n_er, n_er - 1, p_er, ks_er)
        print(f"[ER theoretical] n={n_er}, p={p_er}")
        ax2.plot(ks_er, theo_er, color=COLOR_ER_THEO, linewidth=2, label=f"{er_label} theoretical")
        ax2.legend(frameon=False, fontsize=10, loc='upper right')

    fig.savefig(out_path_ccdf)
    plt.close(fig)
