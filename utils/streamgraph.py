
"""
streamgraph.py
A compact Matplotlib streamgraph helper with options for
- sorted or unsorted ordering
- margins (vertical gaps) between streams
- optional value smoothing (moving-average)
- boundary curve smoothing with Catmull–Rom splines
- label placement at each layer's fattest point
- colormap selection

Author: Rapty 🦖
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

__all__ = [
    "plot_streamgraph",
    "streamgraph_envelopes",
    "catmull_rom_interpolate",
]


# ---------- Smoothing utilities ----------

def moving_average(a: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with reflection at the edges.
    window must be odd and at least 1. When window <= 1 no smoothing is applied.
    """
    if window <= 1:
        return a
    if window % 2 == 0:
        window += 1
    pad = window // 2
    a_pad = np.pad(a, (pad, pad), mode="reflect")
    kernel = np.ones(window) / window
    return np.convolve(a_pad, kernel, mode="valid")


def smooth_series(Y: np.ndarray, window: int = 1) -> np.ndarray:
    """Apply moving average along time for each row of Y. Returns new array."""
    if window <= 1:
        return Y
    return np.vstack([moving_average(y, window) for y in Y])


# ---------- Ordering and stacking ----------

def _order_indices(Yt: np.ndarray, strategy: str = "none", previous: Optional[List[int]] = None) -> List[int]:
    """Ordering of series for a single time step.
    - 'none' keeps the previous order if provided else the original order.
    - 'by_value' sorts by descending magnitude at this time step.
    """
    if strategy == "none":
        return list(range(len(Yt))) if previous is None else previous
    elif strategy == "by_value":
        return list(np.argsort(-Yt))
    else:
        raise ValueError(f"Unknown ordering strategy: {strategy}")


def _baseline_centered(sumY: np.ndarray, margins: float, k: int) -> np.ndarray:
    """Centered baseline so the stack is symmetric around zero.
    margins is a fraction of sumY used as total gap budget at each time.
    """
    total_gap = margins * sumY
    return -0.5 * (sumY + total_gap)


def streamgraph_envelopes(
    Y: np.ndarray,
    margin_frac: float = 0.0,
    order_mode: str = "by_value",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute bottom and top envelopes for each layer.

    Parameters
    ----------
    Y : array, shape (k, n)
        Non-negative series as rows.
    margin_frac : float
        Fraction of the column sum reserved for gaps between layers.
    order_mode : {'by_value','none'}
        Sorting at each time step or not.

    Returns
    -------
    bottoms, tops : arrays of shape (k, n)
    """
    if (Y < 0).any():
        raise ValueError("Y must be non-negative for streamgraphs.")

    k, n = Y.shape
    sumY = Y.sum(axis=0)
    sumY_safe = np.where(sumY == 0, 1.0, sumY)

    baseline = _baseline_centered(sumY, margin_frac, k)
    per_gap = np.where(k > 1, (margin_frac * sumY_safe) / (k - 1), 0.0)

    bottoms = np.zeros_like(Y, dtype=float)
    tops = np.zeros_like(Y, dtype=float)

    prev_order = None
    for t in range(n):
        order_t = _order_indices(Y[:, t], "none" if order_mode == "none" else "by_value", previous=prev_order)
        prev_order = order_t

        b = baseline[t]
        for r, i in enumerate(order_t):
            bottoms[i, t] = b
            tops[i, t] = b + Y[i, t]
            b = tops[i, t] + (per_gap[t] if r < k - 1 else 0.0)

    return bottoms, tops


# ---------- Curve smoothing (Catmull–Rom) ----------

def _catmull_segment(p0, p1, p2, p3, t):
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2 * p1)
        + (-p0 + p2) * t
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
        + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
    )


def catmull_rom_interpolate(x: np.ndarray, y: np.ndarray, samples_per_seg: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """Return dense x_s, y_s that pass through all control points using Catmull–Rom splines.
    samples_per_seg >= 1. When 1, each segment contributes one interior sample and the end
    point of the final segment is appended once.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 2 or samples_per_seg <= 1:
        return x, y

    xs = []
    ys = []
    for i in range(n - 1):
        x0 = x[i - 1] if i - 1 >= 0 else 2 * x[i] - x[i + 1]
        y0 = y[i - 1] if i - 1 >= 0 else 2 * y[i] - y[i + 1]
        x1, y1 = x[i], y[i]
        x2, y2 = x[i + 1], y[i + 1]
        x3 = x[i + 2] if i + 2 < n else 2 * x[i + 1] - x[i]
        y3 = y[i + 2] if i + 2 < n else 2 * y[i + 1] - y[i]

        ts = np.linspace(0, 1, samples_per_seg + 1, endpoint=False)
        xs.append(_catmull_segment(x0, x1, x2, x3, ts))
        ys.append(_catmull_segment(y0, y1, y2, y3, ts))

    xs.append(np.array([x[-1]]))
    ys.append(np.array([y[-1]]))
    return np.concatenate(xs), np.concatenate(ys)


# ---------- Main plotting ----------

def plot_streamgraph(
    X: np.ndarray,
    Y: np.ndarray,
    labels: Optional[List[str]] = None,
    sorted_streams: bool = True,
    margin_frac: float = 0.0,
    smooth_window: int = 1,
    cmap: Optional[str] = None,
    linewidth: float = 0.0,
    alpha: float = 1.0,
    label_placement: bool = True,
    curve_samples: int = 1,
    baseline: str = "center",
    pad_frac: float = 0.05,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot a configurable streamgraph.

    Parameters
    ----------
    X : array (n,)
    Y : array (k, n) with non-negative values
    labels : list of strings
    sorted_streams : bool
        If True, layers are re-ordered by magnitude at each x.
    margin_frac : float
        Fraction of column sum reserved as vertical gaps between layers.
    smooth_window : int
        Moving-average window along the time axis. 1 keeps raw values.
    cmap : str or None
        Matplotlib colormap name. None uses the default property cycle.
    linewidth, alpha : appearance controls.
    label_placement : bool
        Place label at fattest point of each layer.
    curve_samples : int
        When >= 2, densify boundaries using Catmull–Rom curves for smooth transitions.
    baseline : currently only 'center' is supported.
    pad_frac : vertical padding fraction for y-limits.
    ax : optional Matplotlib Axes.
    """
    if baseline != "center":
        raise NotImplementedError("Only baseline='center' is implemented in this version.")

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if (Y < 0).any():
        raise ValueError("Y must be non-negative for streamgraphs.")

    # Value smoothing
    Ys = smooth_series(Y, smooth_window)
    
    # Stacking/interpolation
    order_mode = "by_value" if sorted_streams else "none"
    if curve_samples and curve_samples > 1:
        # Interpolate the non-negative series values first, then restack on the dense grid.
        dense_values = []
        Xp = None
        for i in range(Ys.shape[0]):
            xd, yd = catmull_rom_interpolate(X, Ys[i], samples_per_seg=curve_samples)
            if Xp is None:
                Xp = xd
            dense_values.append(np.clip(yd, 0.0, None))
        Yd = np.vstack(dense_values)
        bottoms, tops = streamgraph_envelopes(Yd, margin_frac=margin_frac, order_mode=order_mode)
    else:
        bottoms, tops = streamgraph_envelopes(Ys, margin_frac=margin_frac, order_mode=order_mode)
        Xp = X

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    colors = None if cmap is None else [get_cmap(cmap, Y.shape[0])(i) for i in range(Y.shape[0])]

    for i in range(Y.shape[0]):
        ax.fill_between(Xp, bottoms[i], tops[i], linewidth=linewidth, alpha=alpha,
                        color=None if colors is None else colors[i])

    # cosmetics
    ax.set_xlim(Xp.min(), Xp.max())
    ymin = float(np.min(bottoms))
    ymax = float(np.max(tops))
    yr = ymax - ymin if ymax > ymin else 1.0
    ax.set_ylim(ymin - pad_frac * yr, ymax + pad_frac * yr)
    ax.spines[["top", "right"]].set_visible(False)

    # labels
    if labels is None:
        labels = [f"S{i+1}" for i in range(Y.shape[0])]

    if label_placement:
        for i, lab in enumerate(labels):
            thickness = tops[i] - bottoms[i]
            if np.all(thickness <= 0):
                continue
            j = int(np.argmax(thickness))
            y = bottoms[i, j] + 0.5 * thickness[j]
            ax.text(Xp[j], y, str(lab), ha="center", va="center", fontsize=10, weight="bold", color="white")

    return ax


# ---------- Minimal demo when executed directly ----------

def _demo():
    rng = np.random.default_rng(7)
    n, k = 40, 5
    X = np.arange(n)
    base = np.linspace(0, 2*np.pi, n)
    Y = []
    for i in range(k):
        phase = rng.uniform(0, 2*np.pi)
        amp = rng.uniform(0.6, 1.3)
        y = amp * (np.sin(base + phase) + 1.2) + rng.normal(0, 0.08, size=n) + 0.15
        y = np.clip(y, 0, None)
        Y.append(y)
    Y = np.vstack(Y)
    labels = list("ABCDE")

    # Figure 1: straight edges
    ax1 = plot_streamgraph(X, Y, labels=labels, sorted_streams=True,
                           margin_frac=0.10, smooth_window=1, cmap=None,
                           curve_samples=1)
    ax1.set_title("Streamgraph with linear boundaries")
    plt.show()

    # Figure 2: spline-smoothed boundaries
    ax2 = plot_streamgraph(X, Y, labels=labels, sorted_streams=True,
                           margin_frac=0.10, smooth_window=1, cmap=None,
                           curve_samples=16)
    ax2.set_title("Streamgraph with Catmull–Rom boundaries")
    plt.show()


if __name__ == "__main__":
    _demo()
