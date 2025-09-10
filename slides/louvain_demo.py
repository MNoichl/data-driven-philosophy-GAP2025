"""
Louvain demo (importable)

Animation logic per active node i (on current stage-graph H):
  1) Intro: i lights up in its FINAL stage color + black ring.
  2) Consider: one NEIGHBOR of i flashes in that color.
  3) Revert: that neighbor goes back (active stays lit).
  Repeat 2–3 for all neighbors of i (deterministic order).
  Then: if i commits this pass (ΔQ>0), keep i colored; else revert i.

Each stage runs on the aggregated graph produced from the previous stage’s
FINAL partition (no separate transition video). A short "settle" easing
aligns positions between stages to avoid jumps.

Outputs (MP4 per stage):
  images/louvain_stage_{level}.mp4

Requires: networkx, matplotlib, numpy, imageio, imageio-ffmpeg, tqdm
"""

import os
import random
from collections import defaultdict, Counter
from statistics import median
import math

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import imageio.v2 as imageio
from tqdm import tqdm
from networkx.algorithms.community.quality import modularity

# -------- Visual defaults --------
GREY = "#c9c9c9"
W, H_PX = 900, 700

plt.style.use("opinionated_rc")
import colormaps as cmaps 


# =====================
# Public helper API
# =====================

def build_graph(which: str = "auto", seed: int = 42):
    """
    Return (G, name) for a built-in example.
    which: "auto" (prefers Les Mis if present), "lesmis", or "karate".
    """
    random.seed(seed)
    np.random.seed(seed)
    if which in ("lesmis", "auto") and hasattr(nx, "les_miserables_graph"):
        G = nx.les_miserables_graph()
        name = "Les Misérables"
    else:
        G = nx.karate_club_graph()
        name = "Zachary's Karate Club (fallback)"
    return nx.Graph(G), name


def compute_pos_orig(G, seed: int = 42):
    """
    Stable original layout (spring). All stages take median positions from these.
    """
    random.seed(seed)
    np.random.seed(seed)
    return nx.spring_layout(G, seed=seed)


def animate_louvain_stages(
    G0,
    pos_orig,
    out_dir: str = "images",
    weight: str = "weight",
    max_levels: int = 10,
    fps: int = 2,
    settle_frames: int = 10,
    seed: int = 42,
):
    """
    End-to-end runner. Returns list of saved stage video paths.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    return _louvain_multilevel_videos(G0, pos_orig, out_dir, weight, max_levels, fps, settle_frames)


# =====================
# Internal utilities
# =====================

def _relabel_partition_dense_with_map(partition):
    """Given node->old_label, return (node->dense_label, old_label->dense_label)."""
    old_to_dense, next_id, dense = {}, 0, {}
    for n, c in partition.items():
        if c not in old_to_dense:
            old_to_dense[c] = next_id
            next_id += 1
        dense[n] = old_to_dense[c]
    return dense, old_to_dense


def _partition_to_communities(partition):
    """node->label  ->  list[set[nodes]]"""
    groups = defaultdict(set)
    for n, c in partition.items():
        groups[c].add(n)
    return list(groups.values())


def _aggregate_graph(H, part, weight="weight"):
    """
    Collapse communities in 'part' (node->label on H) into super-nodes.
    Returns (H2, mapping old_node->supernode).
    """
    groups = defaultdict(list)
    for n, c in part.items():
        groups[c].append(n)
    id_map = {c: idx for idx, c in enumerate(sorted(groups.keys()))}

    H2 = nx.Graph()
    for _, idx in id_map.items():
        H2.add_node(idx)

    for u, v, data in H.edges(data=True):
        w = data.get(weight, 1.0)
        cu = id_map[part[u]]
        cv = id_map[part[v]]
        if H2.has_edge(cu, cv):
            H2[cu][cv]["weight"] += w
        else:
            H2.add_edge(cu, cv, weight=w)

    mapping = {n: id_map[part[n]] for n in H.nodes()}
    return H2, mapping


def _level_layout_and_sizes(H_nodes, pos_orig, orig_to_current):
    """
    For current stage H: place each H-node at the coordinate-wise median of the
    original-node positions that map into it. Size ∝ sqrt(group size).
    """
    groups = defaultdict(list)
    for o, h in orig_to_current.items():
        groups[h].append(o)

    pos_level, sizes = {}, {}
    for h in H_nodes:
        origs = groups.get(h, [])
        if not origs:
            pos_level[h] = (0.0, 0.0)
            sizes[h] = 200.0
        else:
            xs = [pos_orig[o][0] for o in origs]
            ys = [pos_orig[o][1] for o in origs]
            pos_level[h] = (float(median(xs)), float(median(ys)))
            sizes[h] = 200.0 + 60.0 * np.sqrt(len(origs))
    return pos_level, sizes


def _interp_pos(p0, p1, t):
    return (p0[0] * (1 - t) + p1[0] * t, p0[1] * (1 - t) + p1[1] * t)


# ---------------------
# Phase 1 — record event order (neighbors as "considered")
# ---------------------
def _louvain_phase_one_log(H, weight="weight"):
    """
    Record the order of active nodes and whether they commit this pass.
    Also capture the *neighbors* that are considered for each active node.

    Returns:
      final_dense : H-node -> dense final community id (palette for this stage)
      events      : list of dicts with keys:
                    {
                      'active': node,
                      'commit': bool,
                      'considered': [neighbors in sorted order],
                      'dest_comm': community_id_if_moved_else_current (pre-dense ids),
                      'dest_comm_dense': final dense community id of destination
                    }
    """
    nodes_list = list(H.nodes())
    comm_of = {n: idx for idx, n in enumerate(nodes_list)}

    k = dict(H.degree(weight=weight))
    m2 = 2.0 * H.size(weight=weight)
    tot = defaultdict(float)
    for n, deg in k.items():
        tot[comm_of[n]] += deg

    events = []
    pbar = tqdm(desc="Phase 1: building event log", unit="op", leave=False)

    improved = True
    while improved:
        improved = False
        random.shuffle(nodes_list)
        for i in nodes_list:
            ci = comm_of[i]
            ki = k[i]
            tot[ci] -= ki

            neigh_w = defaultdict(float)
            for j, data in H[i].items():
                w = data.get(weight, 1.0)
                cj = comm_of[j]
                neigh_w[cj] += w

            # deterministic set of considered NEIGHBORS
            considered_nodes = sorted(list(H.neighbors(i)), key=lambda x: str(x))

            best_gain = 0.0
            best_comm = ci
            candidate_comms = list(neigh_w.keys()) or [ci]
            for c_neigh in candidate_comms:
                ki_in = neigh_w.get(c_neigh, 0.0)
                gain = ki_in - (ki * tot[c_neigh]) / m2
                if gain > best_gain:
                    best_gain = gain
                    best_comm = c_neigh

            will_commit = (best_comm != ci and best_gain > 1e-15)
            merge_neighbor = None
            if will_commit:
                # Pick a deterministic neighbor exemplar that represents the destination community
                for cand in considered_nodes:
                    if comm_of.get(cand) == best_comm:
                        merge_neighbor = cand
                        break
            events.append({
                "active": i,
                "commit": will_commit,
                "considered": considered_nodes,
                "dest_comm": best_comm,
                "merge_neighbor": merge_neighbor
            })

            if will_commit:
                comm_of[i] = best_comm
                tot[best_comm] += ki
                improved = True
            else:
                tot[ci] += ki

            pbar.update(1)

    pbar.close()
    final_dense, _ = _relabel_partition_dense_with_map({n: comm_of[n] for n in H.nodes()})
    return final_dense, events


# ---------------------
# Rendering
# ---------------------
def _render_frame_level(
    H, pos_level, sizes_level,
    palette_dense, committed_nodes,
    active_node=None, flash_member=None, show_active=True, ring_active=True,
    ring_flash=False,
    title=""
):
    """
    Draw H with stable per-stage palette (palette_dense), persistent colors for committed nodes,
    and optionally overlay the active node and a single flashed neighbor using the active node's final color.
    """
    nodes_list = list(H.nodes())
    cmap = plt.colormaps.get_cmap("tab20")
    labels = sorted(set(palette_dense[n] for n in nodes_list))
    color_map = {c: cmap((i % cmap.N) / cmap.N) for i, c in enumerate(labels)}

    committed_set = set(committed_nodes)
    node_colors = []
    for n in nodes_list:
        if n in committed_set:
            node_colors.append(color_map[palette_dense[n]])
        else:
            node_colors.append(GREY)

    # active node color overlay
    if active_node is not None and show_active:
        dest_color = color_map[palette_dense[active_node]]
        for idx, n in enumerate(nodes_list):
            if n == active_node:
                node_colors[idx] = dest_color

    # flash a single neighbor
    if active_node is not None and flash_member is not None:
        dest_color = color_map[palette_dense[active_node]]
        for idx, n in enumerate(nodes_list):
            if n == flash_member:
                node_colors[idx] = dest_color

    fig = plt.figure(figsize=(W/100, H_PX/100), dpi=100)
    ax = plt.gca()
    ax.axis("off")
    ax.set_title(title, pad=16, fontsize=13)

    nx.draw_networkx_edges(H, pos_level, alpha=0.25, width=1.0, ax=ax)
    node_sizes = [sizes_level.get(n, 200.0) for n in nodes_list]
    nx.draw_networkx_nodes(H, pos_level, nodelist=nodes_list,
                           node_color=node_colors, node_size=node_sizes,
                           linewidths=0.6, edgecolors="black", ax=ax)

    if ring_active and active_node is not None:
        nx.draw_networkx_nodes(
            H, pos_level, nodelist=[active_node],
            node_size=(sizes_level.get(active_node, 200.0) + 120.0),
            node_color='none', edgecolors='black', linewidths=2.2, ax=ax
        )

    # Small band for the currently flashed neighbor (if any)
    if ring_flash and flash_member is not None:
        nx.draw_networkx_nodes(
            H, pos_level, nodelist=[flash_member],
            node_size=(sizes_level.get(flash_member, 200.0) + 60.0),
            node_color='none', edgecolors='black', linewidths=1.6, ax=ax
        )

    # Do not draw node labels

    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    frame = frame[:, :, :3]
    plt.close(fig)
    return frame


def _render_transition_overlay(H_prev, pos_prev, H_next, pos_next, sizes_next,
                               palette_dense_prev, groups_prev,
                               weight_key="weight", title=""):
    """
    Render a static overlay figure that shows the previous-stage graph in the background
    with 50% alpha, and the next-stage aggregated graph on top with node sizes scaled by
    cluster size and edge widths scaled by weights.
    """
    fig = plt.figure(figsize=(W/100, H_PX/100), dpi=100)
    ax = plt.gca()
    ax.axis("off")
    if title:
        ax.set_title(title, pad=16, fontsize=13)

    # Draw previous stage graph in background (50% alpha)
    nx.draw_networkx_edges(H_prev, pos_prev, alpha=0.5, width=1.0, edge_color=GREY, ax=ax)
    nx.draw_networkx_nodes(
        H_prev, pos_prev, nodelist=list(H_prev.nodes()),
        node_color=GREY, alpha=0.5, node_size=120.0, linewidths=0.0, ax=ax
    )

    # Prepare edge widths for H_next based on weights, excluding self-loops
    edges_no_self = [(u, v) for u, v in H_next.edges() if u != v]
    weights = [float(H_next[u][v].get(weight_key, 1.0)) for u, v in edges_no_self]
    max_w = max(weights) if weights else 1.0
    widths = [0.6 + 3.0 * (w / max_w) for w in weights]

    # Foreground node colors based on previous stage cluster colors
    cmap = plt.colormaps.get_cmap("tab20")
    labels_prev = sorted(set(palette_dense_prev[n] for n in H_prev.nodes()))
    color_map_prev = {c: cmap((i % cmap.N) / cmap.N) for i, c in enumerate(labels_prev)}
    next_node_colors = []
    next_node_sizes = []
    for h2 in H_next.nodes():
        members = groups_prev.get(h2, [])
        if members:
            labels_members = [palette_dense_prev[m] for m in members]
            majority_label = Counter(labels_members).most_common(1)[0][0]
            next_node_colors.append(color_map_prev[majority_label])
        else:
            next_node_colors.append("white")
        next_node_sizes.append(sizes_next.get(h2, 200.0))

    # Draw next stage edges and nodes on top
    nx.draw_networkx_edges(
        H_next, pos_next, edgelist=edges_no_self,
        width=widths if widths else 1.5, alpha=0.9, edge_color="black", ax=ax
    )
    nx.draw_networkx_nodes(
        H_next, pos_next, nodelist=list(H_next.nodes()),
        node_color=next_node_colors, edgecolors="black", linewidths=1.6,
        node_size=next_node_sizes, ax=ax
    )

    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    frame = frame[:, :, :3]
    plt.close(fig)
    return frame


# ---------------------
# Multilevel runner
# ---------------------
def _louvain_multilevel_videos(
    G0, pos_orig, out_dir="images", weight="weight",
    max_levels=10, fps=2, settle_frames=10
):
    """
    Internal: produce per-stage MP4s. Returns list of file paths.
    """
    saved_paths = []

    orig_nodes = list(G0.nodes())
    orig_to_current = {o: o for o in orig_nodes}
    H = G0.copy()
    level = 0
    prev_mod = None
    carry_pos_next = None

    while level < max_levels:
        # Positions/sizes for this stage (medians from original layout)
        pos_level, sizes_level = _level_layout_and_sizes(H.nodes(), pos_orig, orig_to_current)

        # Build event log and final palette (stable per-stage colors)
        palette_dense, events = _louvain_phase_one_log(H, weight=weight)

        # -------------------------------
        # Compute total frames for stage
        # -------------------------------
        total_settle = 0
        if carry_pos_next is not None and settle_frames > 0:
            total_settle = settle_frames

        frames_per_event = []
        for ev in events:
            deg_count = len(ev.get("considered", []))
            frames_per_event.append(2 + 2 * deg_count)  # intro + final + 2 per neighbor
        total_event_frames = sum(frames_per_event)
        total_frames_stage = total_settle + total_event_frames

        # -------------------------------
        # Time-warp config for level 0
        # -------------------------------
        def compute_timewarp_repeats(total_frames: int, fps_in: int,
                                     split_sec: float = 180.0,
                                     seg1_out_sec: float = 30.0,
                                     seg2_out_sec: float = 60.0,
                                     min_fps_out: int = 60):
            """
            Return (fps_out, repeats_list[int]) such that:
            - No source frame is dropped (each repeated >= 1 time)
            - First min(3 min, total) of source fits into 30s, rest into 60s
            - Constant-fps writer (fps_out) is used; per-frame duration is controlled by repetition counts
            """
            if total_frames == 0:
                return max(min_fps_out, fps_in), []

            # frames corresponding to 3 minutes in the original stage timeline
            split_frames = min(total_frames, int(round(split_sec * fps_in)))
            rem_frames = total_frames - split_frames

            if rem_frames <= 0:
                # Entire stage fits in the first segment. Use 30s target.
                # Choose fps_out so that we can show all frames without dropping.
                fps_needed = math.ceil(total_frames / max(1e-9, seg1_out_sec))
                fps_out = max(min_fps_out, fps_needed)
                target_total = int(round(seg1_out_sec * fps_out))
                target_total = max(target_total, total_frames)  # ensure no drop
                # Smooth weights (more repeats earlier), baseline 1 repeat per frame
                if total_frames <= 1:
                    repeats = [target_total]
                    return fps_out, repeats
                p = 2.0
                weights = []
                for j in range(total_frames):
                    u = 0.0 if total_frames <= 1 else j / (total_frames - 1)
                    weights.append(1.0 + 2.0 * (1.0 - u) ** p)
                # allocate repeats: start with 1 per frame, then distribute the remaining using weights
                base = [1] * total_frames
                remaining = target_total - total_frames
                if remaining > 0:
                    s = sum(weights)
                    shares = [remaining * (w / s) for w in weights]
                    adds_floor = [int(math.floor(x)) for x in shares]
                    added = sum(adds_floor)
                    frac = [shares[i] - adds_floor[i] for i in range(total_frames)]
                    order = sorted(range(total_frames), key=lambda i: -frac[i])
                    for i in range(total_frames):
                        base[i] += adds_floor[i]
                    for idx in order[:remaining - added]:
                        base[idx] += 1
                return fps_out, base

            # Two segments
            fps_needed_1 = math.ceil(split_frames / max(1e-9, seg1_out_sec))
            fps_needed_2 = math.ceil(rem_frames / max(1e-9, seg2_out_sec))
            fps_out = max(min_fps_out, fps_needed_1, fps_needed_2)
            target_1 = int(round(seg1_out_sec * fps_out))
            target_2 = int(round(seg2_out_sec * fps_out))
            # Ensure we don't under-allocate vs frames count
            target_1 = max(target_1, split_frames)
            target_2 = max(target_2, rem_frames)
            target_total = target_1 + target_2

            # Smoothly varying weights within each segment (quadratic falloff),
            # baseline 1 repeat per frame, distribute remaining to match targets.
            def allocate_segment(F: int, T: int, start_bias: float = 2.0) -> list:
                if F <= 0:
                    return []
                if F == 1:
                    return [T]
                pwr = 2.0
                weights = []
                for j in range(F):
                    u = j / (F - 1)
                    weights.append(1.0 + start_bias * (1.0 - u) ** pwr)
                base = [1] * F
                remaining = T - F
                if remaining > 0:
                    s = sum(weights)
                    shares = [remaining * (w / s) for w in weights]
                    adds_floor = [int(math.floor(x)) for x in shares]
                    added = sum(adds_floor)
                    frac = [shares[i] - adds_floor[i] for i in range(F)]
                    order = sorted(range(F), key=lambda i: -frac[i])
                    for i in range(F):
                        base[i] += adds_floor[i]
                    for idx in order[:remaining - added]:
                        base[idx] += 1
                return base

            base_1 = allocate_segment(split_frames, target_1, start_bias=2.0)
            # Make second segment slightly faster overall by using a smaller bias so repeats continue to decrease
            base_2 = allocate_segment(rem_frames, target_2, start_bias=1.0)

            repeats = base_1 + base_2
            # As a final guard, ensure total matches target_total
            diff = target_total - sum(repeats)
            if diff > 0:
                # Add to the last frames
                for i in range(diff):
                    repeats[-1 - (i % len(repeats))] += 1
            elif diff < 0:
                # Remove extras from the largest repeats but keep >=1
                i = 0
                while diff < 0 and i < len(repeats):
                    if repeats[i] > 1:
                        repeats[i] -= 1
                        diff += 1
                    i += 1
            return fps_out, repeats

        # Decide writer fps and per-frame repeats
        if level == 0:
            fps_out, repeats = compute_timewarp_repeats(
                total_frames_stage, fps,
                split_sec=180.0, seg1_out_sec=30.0, seg2_out_sec=60.0, min_fps_out=60
            )
        else:
            fps_out = fps
            repeats = [1] * total_frames_stage

        # Prepare evolving community assignment to show dynamic modularity during the pass.
        # Start with each node in its own community, matching the initialization in _louvain_phase_one_log.
        nodes_list_stage = list(H.nodes())
        comm_of_current = {n: idx for idx, n in enumerate(nodes_list_stage)}

        def modularity_from_mapping(mapping: dict) -> float:
            parts = _partition_to_communities(mapping)
            return modularity(H, parts, weight=weight)

        # Writer for this stage (constant fps). Per-frame duration achieved via repeats
        stage_path = os.path.join(out_dir, f"louvain_stage_{level}.mp4")
        writer = imageio.get_writer(stage_path, fps=fps_out, codec="libx264", quality=8)

        # Stage settle (ease positions from previous stage carry-in)
        frame_idx_global = 0
        if carry_pos_next is not None and settle_frames > 0:
            pbar_settle = tqdm(total=settle_frames, desc=f"Stage {level}: settle", unit="frame", leave=False)
            for s in range(settle_frames):
                t = (s + 1) / float(settle_frames)
                interp = {n: _interp_pos(carry_pos_next.get(n, pos_level[n]), pos_level[n], t) for n in H.nodes()}
                frame = _render_frame_level(
                    H, interp, sizes_level, palette_dense, committed_nodes=set(),
                    active_node=None, flash_member=None, show_active=False, ring_active=False,
                    title=f"Level {level} (settle) | Q={modularity_from_mapping(comm_of_current):.3f}"
                )
                # write with repeats (advance index once per source frame)
                rep = repeats[frame_idx_global]
                for _ in range(rep):
                    writer.append_data(frame)
                frame_idx_global += 1
                pbar_settle.update(1)
            pbar_settle.close()

        # Render events in exact 1–2–3 pattern with NEIGHBORS as considered nodes
        committed_set = set()
        pbar = tqdm(total=len(events), desc=f"Stage {level}: rendering nodes", unit="node", leave=False)
        for ev in events:
            i = ev["active"]
            will_commit = ev["commit"]
            considered = ev.get("considered", [])
            merge_neighbor = ev.get("merge_neighbor")
            # Local view of which nodes should already persist color during this node's sequence
            temp_committed = set(committed_set)
            if will_commit and merge_neighbor is not None:
                # Merge target should stay colored immediately during this sequence
                temp_committed.add(merge_neighbor)

            # 1) Intro: active lights in final color + ring
            # Compute modularity for the current trial (active moved if it will commit)
            if will_commit:
                comm_trial = dict(comm_of_current)
                comm_trial[i] = ev["dest_comm"]
                q_current = modularity_from_mapping(comm_trial)
            else:
                q_current = modularity_from_mapping(comm_of_current)

            frame_intro = _render_frame_level(
                H, pos_level, sizes_level, palette_dense, committed_nodes=temp_committed,
                active_node=i, flash_member=None, show_active=True, ring_active=True,
                title=f"Level {level} | Q={q_current:.3f}"
            )
            rep = repeats[frame_idx_global]
            for _ in range(rep):
                writer.append_data(frame_intro)
            frame_idx_global += 1

            # 2 & 3) Flash each considered neighbor, then revert (active stays lit)
            for m in considered:
                frame2 = _render_frame_level(
                    H, pos_level, sizes_level, palette_dense, committed_nodes=temp_committed,
                    active_node=i, flash_member=m, show_active=True, ring_active=True,
                    ring_flash=True,
                    title=f"Level {level} | Q={q_current:.3f}"
                )
                rep = repeats[frame_idx_global]
                for _ in range(rep):
                    writer.append_data(frame2)
                frame_idx_global += 1

                # If this neighbor is the merge target, keep it colored from now on
                if will_commit and merge_neighbor is not None and m == merge_neighbor:
                    temp_committed.add(merge_neighbor)

                frame3 = _render_frame_level(
                    H, pos_level, sizes_level, palette_dense, committed_nodes=temp_committed,
                    active_node=i, flash_member=None, show_active=True, ring_active=True,
                    ring_flash=False,
                    title=f"Level {level} | Q={q_current:.3f}"
                )
                rep = repeats[frame_idx_global]
                for _ in range(rep):
                    writer.append_data(frame3)
                frame_idx_global += 1

            # Finalization: active stays if commit, else revert
            if will_commit:
                committed_set.add(i)
                if merge_neighbor is not None:
                    committed_set.add(merge_neighbor)
                frame_final = _render_frame_level(
                    H, pos_level, sizes_level, palette_dense, committed_nodes=committed_set,
                    active_node=None, flash_member=None, show_active=False, ring_active=False,
                    title=f"Level {level} | Q={q_current:.3f}"
                )
                # Commit the move in the evolving partition
                comm_of_current[i] = ev["dest_comm"]
            else:
                frame_final = _render_frame_level(
                    H, pos_level, sizes_level, palette_dense, committed_nodes=committed_set,
                    active_node=None, flash_member=None, show_active=False, ring_active=False,
                    title=f"Level {level} | Q={modularity_from_mapping(comm_of_current):.3f}"
                )
            rep = repeats[frame_idx_global]
            for _ in range(rep):
                writer.append_data(frame_final)
            frame_idx_global += 1
            pbar.update(1)

        pbar.close()
        writer.close()
        saved_paths.append(stage_path)

        # Decide continuation
        comms_H = _partition_to_communities({n: palette_dense[n] for n in H.nodes()})
        mod_H = modularity(H, comms_H, weight=weight)
        if prev_mod is not None and mod_H <= prev_mod + 1e-12:
            break
        prev_mod = mod_H
        if len(comms_H) == H.number_of_nodes():
            break

        # Aggregate for next stage and compute carry-in positions
        H_next, mapping = _aggregate_graph(H, palette_dense, weight=weight)

        # carry-in positions: median of current positions of grouped nodes
        groups_prev = defaultdict(list)
        for u in H.nodes():
            groups_prev[mapping[u]].append(u)
        carry_pos_next = {}
        for h2, members in groups_prev.items():
            xs = [pos_level[u][0] for u in members]
            ys = [pos_level[u][1] for u in members]
            carry_pos_next[h2] = (float(median(xs)), float(median(ys)))

        # sizes for next stage nodes proportional to sqrt(cluster size)
        sizes_next = {}
        for h2, members in groups_prev.items():
            sizes_next[h2] = 200.0 + 60.0 * np.sqrt(len(members))

        # Render and save intermediate overlay image showing next stage on top
        overlay_frame = _render_transition_overlay(
            H_prev=H, pos_prev=pos_level,
            H_next=H_next, pos_next=carry_pos_next,
            sizes_next=sizes_next, palette_dense_prev=palette_dense, groups_prev=groups_prev,
            weight_key=weight,
            title=f"Level {level} - {level+1}: Aggregation"
        )
        overlay_path = os.path.join(out_dir, f"louvain_transition_{level}_to_{level+1}.png")
        imageio.imwrite(overlay_path, overlay_frame)

        # Update original->current mapping for next stage
        orig_to_current = {o: mapping[orig_to_current[o]] for o in orig_nodes}

        H = H_next
        level += 1

    return saved_paths