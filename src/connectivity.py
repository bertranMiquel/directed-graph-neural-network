#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
from collections import Counter
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from .datasets.data_loading import get_dataset
from .datasets.dataset import FullBatchGraphDataset
from torch.utils.data import DataLoader

try:
    import torch
    from torch_geometric.data import Data as PyGData
except Exception:  # pragma: no cover
    torch = None
    PyGData = Any  # type: ignore[assignment]

try:
    from scipy import sparse
except Exception:  # pragma: no cover
    sparse = None


log = logging.getLogger("connectivity")

# Adapted and extended from the attached connectivity.py. :contentReference[oaicite:0]{index=0}


def get_data_from_loader(dataset, batch_size: int = 1):
    data = dataset._data
    data_loader = DataLoader(
        FullBatchGraphDataset(data),
        batch_size=batch_size,
        collate_fn=lambda batch: batch[0],
    )
    return next(iter(data_loader))


# ----------------------------- Utility helpers ----------------------------- #

def gini_coefficient(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    if np.all(x == 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    idx = np.arange(1, n + 1, dtype=np.float64)
    denom = n * x.sum()
    if denom == 0:
        return 0.0
    return float((2.0 * np.dot(idx, x) / denom) - ((n + 1.0) / n))


def ensure_undirected(G: nx.Graph | nx.DiGraph, make_copy: bool = True) -> nx.Graph:
    if G.is_directed():
        UG = nx.Graph()
        UG.add_nodes_from(G.nodes())
        UG.add_edges_from(G.edges())
        return UG
    return G.copy() if make_copy else G  # type: ignore[return-value]


def get_edge_index_numpy(data: PyGData) -> np.ndarray:
    edge_index = data["edge_index"]
    if edge_index is None:
        return np.empty((2, 0), dtype=np.int64)
    if torch is None:
        raise RuntimeError("torch is required to process PyG edge_index")
    return edge_index.detach().cpu().numpy().astype(np.int64, copy=False)


def get_num_nodes(data: PyGData) -> int:
    if getattr(data, "num_nodes", None) is not None:
        return int(data.num_nodes)
    y = data.get("y", None)
    if y is not None:
        return int(y.size(0))
    edge_index = data.get("edge_index", None)
    if edge_index is None or edge_index.numel() == 0:
        return 0
    return int(edge_index.max().item()) + 1


def pyg_to_nx_minimal(data: PyGData, undirected: bool) -> nx.Graph | nx.DiGraph:
    ei = get_edge_index_numpy(data)
    n = get_num_nodes(data)
    G: nx.Graph | nx.DiGraph = nx.Graph() if undirected else nx.DiGraph()
    G.add_nodes_from(range(n))
    if ei.shape[1] > 0:
        G.add_edges_from(zip(ei[0], ei[1]))
    return G


def largest_connected_component(G: nx.Graph | nx.DiGraph) -> nx.Graph:
    UG = ensure_undirected(G)
    if UG.number_of_nodes() == 0:
        return UG
    try:
        lcc_nodes = max(nx.connected_components(UG), key=len)
    except ValueError:
        return UG
    return UG.subgraph(lcc_nodes).copy()


def safe_density(G: nx.Graph) -> float:
    n = G.number_of_nodes()
    if n <= 1:
        return 0.0
    m = G.number_of_edges()
    return (2.0 * m) / (n * (n - 1))


def cheeger_constant(G: nx.Graph) -> float:
    n = G.number_of_nodes()
    if n <= 1:
        return float("nan")

    nodes = np.fromiter(G.nodes(), dtype=np.int64, count=n)
    rng = np.random.default_rng(27)
    n_samples = min(256, max(32, n))
    best = float("inf")

    for _ in range(n_samples):
        k = int(rng.integers(1, max(2, n // 2 + 1)))
        subset = set(rng.choice(nodes, size=k, replace=False).tolist())
        cut_edges = sum(1 for u, v in G.edges() if (u in subset) != (v in subset))
        denom = min(len(subset), n - len(subset))
        if denom > 0:
            best = min(best, cut_edges / denom)

    return float(best) if np.isfinite(best) else float("nan")


def _extract_labels_numpy(data: PyGData) -> np.ndarray | None:
    y = data.get("y", None)
    if y is None or torch is None:
        return None

    y = y.detach().cpu()
    if y.ndim == 1:
        return y.numpy().astype(np.int64, copy=False)

    # For multi-dimensional labels, compress rows to categorical ids.
    y_np = y.numpy()
    _, inv = np.unique(y_np, axis=0, return_inverse=True)
    return inv.astype(np.int64, copy=False)


def _safe_entropy_from_counts(counts: np.ndarray) -> float:
    s = counts.sum()
    if s <= 0:
        return 0.0
    p = counts[counts > 0] / s
    return float(-(p * np.log(p)).sum())


def _js_divergence_from_count_vectors(c1: np.ndarray, c2: np.ndarray) -> float:
    s1 = c1.sum()
    s2 = c2.sum()
    if s1 <= 0 and s2 <= 0:
        return float("nan")
    if s1 <= 0 or s2 <= 0:
        return float("nan")
    p = c1 / s1
    q = c2 / s2
    m = 0.5 * (p + q)

    mask_p = p > 0
    mask_q = q > 0
    kl_pm = np.sum(p[mask_p] * np.log(p[mask_p] / m[mask_p]))
    kl_qm = np.sum(q[mask_q] * np.log(q[mask_q] / m[mask_q]))
    return float(0.5 * (kl_pm + kl_qm))


# ----------------------------- Directed metrics ----------------------------- #

def reverse_edge_ratio_metrics(data: PyGData) -> dict[str, float]:
    """
    Edge-level asymmetry metrics.
    Computed on unique non-self-loop directed edges only.
    """
    ei = get_edge_index_numpy(data)
    if ei.shape[1] == 0:
        return {
            "reverse_edge_ratio": float("nan"),
            "missing_reverse_edges": float("nan"),
            "unique_directed_edges_no_self_loops": 0.0,
            "reciprocity_ratio": float("nan"),
            "bidirectionality_gap": float("nan"),
        }

    src = ei[0]
    dst = ei[1]
    mask = src != dst
    src = src[mask]
    dst = dst[mask]
    if src.size == 0:
        return {
            "reverse_edge_ratio": float("nan"),
            "missing_reverse_edges": float("nan"),
            "unique_directed_edges_no_self_loops": 0.0,
            "reciprocity_ratio": float("nan"),
            "bidirectionality_gap": float("nan"),
        }

    n_base = int(max(src.max(initial=0), dst.max(initial=0))) + 1
    keys = src * n_base + dst
    rev_keys = dst * n_base + src

    unique_keys = np.unique(keys)
    unique_rev = np.unique(rev_keys)
    key_set = set(unique_keys.tolist())

    missing = sum(1 for rk in unique_rev.tolist() if rk not in key_set)
    total = unique_keys.size
    reverse_ratio = float(missing / total) if total > 0 else float("nan")
    reciprocity = float(1.0 - reverse_ratio) if total > 0 else float("nan")

    return {
        "reverse_edge_ratio": 100.0 * reverse_ratio if total > 0 else float("nan"),
        "missing_reverse_edges": float(missing),
        "unique_directed_edges_no_self_loops": float(total),
        "reciprocity_ratio": 100.0 * reciprocity if total > 0 else float("nan"),
        "bidirectionality_gap": 100.0 * reverse_ratio if total > 0 else float("nan"),
    }


def directional_node_role_metrics(data: PyGData) -> dict[str, float]:
    """
    Directional imbalance per node:
      b_i = |out_i - in_i| / (out_i + in_i + eps)

    User requested mean, min, max.
    Also includes fractions of near-sources and near-sinks.
    """
    ei = get_edge_index_numpy(data)
    n = get_num_nodes(data)

    out = {
        "in_degree_mean": float("nan"),
        "out_degree_mean": float("nan"),
        "directional_imbalance_mean": float("nan"),
        "directional_imbalance_min": float("nan"),
        "directional_imbalance_max": float("nan"),
        "fraction_near_sources": float("nan"),
        "fraction_near_sinks": float("nan"),
        "fraction_pure_sources": float("nan"),
        "fraction_pure_sinks": float("nan"),
    }
    if n == 0:
        return out

    if ei.shape[1] == 0:
        zeros = np.zeros(n, dtype=np.float64)
        out.update({
            "in_degree_mean": 0.0,
            "out_degree_mean": 0.0,
            "directional_imbalance_mean": 0.0,
            "directional_imbalance_min": 0.0,
            "directional_imbalance_max": 0.0,
            "fraction_near_sources": 0.0,
            "fraction_near_sinks": 0.0,
            "fraction_pure_sources": 0.0,
            "fraction_pure_sinks": 0.0,
        })
        return out

    src = ei[0]
    dst = ei[1]

    out_deg = np.bincount(src, minlength=n).astype(np.float64, copy=False)
    in_deg = np.bincount(dst, minlength=n).astype(np.float64, copy=False)

    denom = in_deg + out_deg
    imbalance = np.zeros(n, dtype=np.float64)
    nz = denom > 0
    imbalance[nz] = np.abs(out_deg[nz] - in_deg[nz]) / denom[nz]

    # "Near" thresholds: at least 90% of incident edges concentrated in one direction.
    # Equivalent to imbalance >= 0.9, plus presence of at least one incident edge.
    near_source = nz & (out_deg > in_deg) & (imbalance >= 0.9)
    near_sink = nz & (in_deg > out_deg) & (imbalance >= 0.9)

    pure_source = (out_deg > 0) & (in_deg == 0)
    pure_sink = (in_deg > 0) & (out_deg == 0)

    out.update({
        "in_degree_mean": float(in_deg.mean()),
        "out_degree_mean": float(out_deg.mean()),
        "directional_imbalance_mean": float(imbalance.mean()),
        "directional_imbalance_min": float(imbalance.min()),
        "directional_imbalance_max": float(imbalance.max()),
        "fraction_near_sources": float(near_source.mean() * 100.0),
        "fraction_near_sinks": float(near_sink.mean() * 100.0),
        "fraction_pure_sources": float(pure_source.mean() * 100.0),
        "fraction_pure_sinks": float(pure_sink.mean() * 100.0),
    })
    return out


def scc_fragmentation_metrics(G: nx.Graph | nx.DiGraph) -> dict[str, float]:
    """
    Global one-way organization metrics.
    """
    out = {
        "n_strongly_connected_components": float("nan"),
        "largest_scc_size": float("nan"),
        "largest_scc_fraction": float("nan"),
        "n_weakly_connected_components": float("nan"),
        "largest_wcc_size": float("nan"),
        "largest_wcc_fraction": float("nan"),
        "fraction_edges_between_sccs": float("nan"),
        "condensation_n_nodes": float("nan"),
        "condensation_n_edges": float("nan"),
        "condensation_density": float("nan"),
    }

    if G.number_of_nodes() == 0:
        return out

    if not G.is_directed():
        # For undirected graphs SCC/WCC coincide with CCs.
        comps = list(nx.connected_components(G))
        sizes = np.array([len(c) for c in comps], dtype=np.float64) if comps else np.array([], dtype=np.float64)
        n = G.number_of_nodes()
        largest = float(sizes.max()) if sizes.size else 0.0
        frac = float(100.0 * largest / n) if n > 0 else float("nan")
        return {
            "n_strongly_connected_components": float(len(comps)),
            "largest_scc_size": largest,
            "largest_scc_fraction": frac,
            "n_weakly_connected_components": float(len(comps)),
            "largest_wcc_size": largest,
            "largest_wcc_fraction": frac,
            "fraction_edges_between_sccs": 0.0,
            "condensation_n_nodes": float(len(comps)),
            "condensation_n_edges": 0.0,
            "condensation_density": 0.0,
        }

    n = G.number_of_nodes()

    sccs = list(nx.strongly_connected_components(G))
    scc_sizes = np.array([len(c) for c in sccs], dtype=np.float64) if sccs else np.array([], dtype=np.float64)
    largest_scc = float(scc_sizes.max()) if scc_sizes.size else 0.0

    wccs = list(nx.weakly_connected_components(G))
    wcc_sizes = np.array([len(c) for c in wccs], dtype=np.float64) if wccs else np.array([], dtype=np.float64)
    largest_wcc = float(wcc_sizes.max()) if wcc_sizes.size else 0.0

    scc_index: dict[int, int] = {}
    for idx, comp in enumerate(sccs):
        for u in comp:
            scc_index[int(u)] = idx

    inter_scc_edges = 0
    for u, v in G.edges():
        if scc_index.get(int(u), -1) != scc_index.get(int(v), -1):
            inter_scc_edges += 1

    try:
        C = nx.condensation(G, scc=sccs)
        c_nodes = C.number_of_nodes()
        c_edges = C.number_of_edges()
        c_density = float(nx.density(C)) if c_nodes > 1 else 0.0
    except Exception:
        c_nodes = len(sccs)
        c_edges = float("nan")
        c_density = float("nan")

    return {
        "n_strongly_connected_components": float(len(sccs)),
        "largest_scc_size": largest_scc,
        "largest_scc_fraction": float(100.0 * largest_scc / n) if n > 0 else float("nan"),
        "n_weakly_connected_components": float(len(wccs)),
        "largest_wcc_size": largest_wcc,
        "largest_wcc_fraction": float(100.0 * largest_wcc / n) if n > 0 else float("nan"),
        "fraction_edges_between_sccs": float(100.0 * inter_scc_edges / max(1, G.number_of_edges())),
        "condensation_n_nodes": float(c_nodes),
        "condensation_n_edges": float(c_edges),
        "condensation_density": c_density,
    }


def reachability_asymmetry_metrics(
    G: nx.Graph | nx.DiGraph,
    max_sources: int = 128,
    random_state: int = 27,
) -> dict[str, float]:
    """
    Approximate reachability asymmetry using sampled BFS/DFS sources.
    Memory-wise: no all-pairs matrix, only per-source reachability sets.
    """
    out = {
        "reachability_asymmetry": float("nan"),
        "reachable_pairs_sampled": float("nan"),
        "asymmetric_reachable_pairs_sampled": float("nan"),
    }

    if G.number_of_nodes() == 0:
        return out

    if not G.is_directed():
        return {
            "reachability_asymmetry": 0.0,
            "reachable_pairs_sampled": 0.0,
            "asymmetric_reachable_pairs_sampled": 0.0,
        }

    nodes = np.fromiter(G.nodes(), dtype=np.int64, count=G.number_of_nodes())
    if nodes.size == 0:
        return out

    rng = np.random.default_rng(random_state)
    sources = rng.choice(nodes, size=min(max_sources, nodes.size), replace=False)

    reach_sets: dict[int, set[int]] = {}
    for s in sources.tolist():
        # descendants excludes the source itself
        reach_sets[int(s)] = nx.descendants(G, int(s))

    asym = 0
    total_reachable = 0
    source_list = list(reach_sets.keys())

    for u in source_list:
        Ru = reach_sets[u]
        for v in source_list:
            if u == v:
                continue
            uv = v in Ru
            vu = u in reach_sets[v]
            if uv or vu:
                total_reachable += 1
                if uv != vu:
                    asym += 1

    out["reachable_pairs_sampled"] = float(total_reachable)
    out["asymmetric_reachable_pairs_sampled"] = float(asym)
    out["reachability_asymmetry"] = float(100.0 * asym / total_reachable) if total_reachable > 0 else float("nan")
    return out


def label_flow_asymmetry_metrics(data: PyGData) -> dict[str, float]:
    """
    Memory-efficient directed label-flow asymmetry.

    T[a, b] = number of edges from class a to class b
    Measure asymmetry via ||T - T^T||_1 / ||T||_1

    Implemented without dense N x N structures; only K x K label-flow matrix,
    where K is number of classes/label patterns.
    """
    out = {
        "label_flow_asymmetry": float("nan"),
        "label_flow_n_classes": float("nan"),
    }

    y = _extract_labels_numpy(data)
    if y is None:
        return out

    ei = get_edge_index_numpy(data)
    if ei.shape[1] == 0:
        return out

    src = ei[0]
    dst = ei[1]
    if src.size == 0:
        return out

    classes, y_compact = np.unique(y, return_inverse=True)
    k = classes.size
    out["label_flow_n_classes"] = float(k)
    if k <= 1:
        out["label_flow_asymmetry"] = 0.0
        return out

    cs = y_compact[src]
    cd = y_compact[dst]

    T = np.zeros((k, k), dtype=np.int64)
    np.add.at(T, (cs, cd), 1)

    num = np.abs(T - T.T).sum()
    den = T.sum()
    out["label_flow_asymmetry"] = float(100.0 * num / den) if den > 0 else float("nan")
    return out


def in_out_label_jsd_metrics(data: PyGData) -> dict[str, float]:
    """
    For each node i, compare label distributions of predecessors vs successors
    with Jensen-Shannon divergence.

    Implemented in a sparse streaming style:
    - compress labels to 0..K-1
    - accumulate per-node incoming/outgoing class counts with np.add.at
    - compute JSD node-wise
    """
    out = {
        "in_out_label_jsd_mean": float("nan"),
        "in_out_label_jsd_max": float("nan"),
        "in_out_label_jsd_valid_nodes": float("nan"),
    }

    y = _extract_labels_numpy(data)
    if y is None:
        return out

    ei = get_edge_index_numpy(data)
    n = get_num_nodes(data)
    if n == 0 or ei.shape[1] == 0:
        return out

    classes, y_compact = np.unique(y, return_inverse=True)
    k = classes.size
    if k <= 1:
        return {
            "in_out_label_jsd_mean": 0.0,
            "in_out_label_jsd_max": 0.0,
            "in_out_label_jsd_valid_nodes": 0.0,
        }

    src = ei[0]
    dst = ei[1]

    out_cls = y_compact[dst]  # successor labels for source nodes
    in_cls = y_compact[src]   # predecessor labels for destination nodes

    out_counts = np.zeros((n, k), dtype=np.int32)
    in_counts = np.zeros((n, k), dtype=np.int32)

    np.add.at(out_counts, (src, out_cls), 1)
    np.add.at(in_counts, (dst, in_cls), 1)

    js_vals = []
    out_deg = out_counts.sum(axis=1)
    in_deg = in_counts.sum(axis=1)
    valid = (out_deg > 0) & (in_deg > 0)

    valid_idx = np.flatnonzero(valid)
    for i in valid_idx.tolist():
        js = _js_divergence_from_count_vectors(in_counts[i], out_counts[i])
        if np.isfinite(js):
            js_vals.append(js)

    if len(js_vals) == 0:
        return out

    js_arr = np.asarray(js_vals, dtype=np.float64)
    return {
        "in_out_label_jsd_mean": float(js_arr.mean()),
        "in_out_label_jsd_max": float(js_arr.max()),
        "in_out_label_jsd_valid_nodes": float(js_arr.size),
    }


# ----------------------------- Existing metrics ----------------------------- #

def degree_stats(G: nx.Graph) -> dict[str, float]:
    deg = np.fromiter((d for _, d in G.degree()), dtype=np.float64)
    if deg.size == 0:
        return {k: float("nan") for k in [
            "degree_p5",
            "degree_min",
            "degree_max",
            "degree_mean",
            "degree_median",
            "degree_std",
            "degree_p95",
            "degree_gini",
        ]}
    return {
        "degree_p5": float(np.percentile(deg, 5)),
        "degree_min": float(deg.min()),
        "degree_max": float(deg.max()),
        "degree_mean": float(deg.mean()),
        "degree_median": float(np.median(deg)),
        "degree_std": float(deg.std(ddof=0)),
        "degree_p95": float(np.percentile(deg, 95)),
        "degree_gini": float(gini_coefficient(deg)),
    }


def triangle_and_clustering(G: nx.Graph) -> dict[str, float]:
    if G.number_of_nodes() == 0:
        return {
            "triangles_total": 0.0,
            "triangles_avg_per_node": 0.0,
            "triangles_max_per_node": 0.0,
            "clustering_coefficient_avg": float("nan"),
            "transitivity": float("nan"),
        }
    tri_per_node = nx.triangles(G)
    tri_values = np.fromiter(tri_per_node.values(), dtype=np.float64)
    triangles_total = float(tri_values.sum() / 3.0)
    return {
        "triangles_total": triangles_total,
        "triangles_avg_per_node": float(tri_values.mean()) if tri_values.size else 0.0,
        "triangles_max_per_node": float(tri_values.max()) if tri_values.size else 0.0,
        "clustering_coefficient_avg": float(nx.average_clustering(G)),
        "transitivity": float(nx.transitivity(G)),
    }


def degree_assortativity(G: nx.Graph) -> float:
    if G.number_of_edges() == 0:
        return float("nan")
    try:
        return float(nx.degree_assortativity_coefficient(G))
    except Exception:
        return float("nan")


def distances_and_effective_diameter(
    G: nx.Graph,
    approx: bool,
    sampled_bfs_sources: int = 300,
    random_state: int = 27,
) -> dict[str, float]:
    if G.number_of_nodes() == 0:
        return {
            "diameter_LCC": float("nan"),
            "radius_LCC": float("nan"),
            "eccentricity_mean_LCC": float("nan"),
            "effective_diameter90_LCC": float("nan"),
        }

    out: dict[str, float] = {}
    try:
        ecc = nx.eccentricity(G)
        vals = np.fromiter(ecc.values(), dtype=np.float64)
        out["diameter_LCC"] = float(vals.max())
        out["radius_LCC"] = float(vals.min())
        out["eccentricity_mean_LCC"] = float(vals.mean())
    except Exception:
        out["diameter_LCC"] = float("nan")
        out["radius_LCC"] = float("nan")
        out["eccentricity_mean_LCC"] = float("nan")

    rng = np.random.default_rng(random_state)
    nodes = np.fromiter(G.nodes(), dtype=np.int64, count=G.number_of_nodes())
    if approx:
        sources = rng.choice(nodes, size=min(len(nodes), sampled_bfs_sources), replace=False)
    else:
        sources = nodes

    hist: Counter[int] = Counter()
    total = 0
    for s in sources.tolist() if isinstance(sources, np.ndarray) else sources:
        lengths = nx.single_source_shortest_path_length(G, int(s))
        for d in lengths.values():
            if d > 0:
                hist[int(d)] += 1
                total += 1

    if total == 0:
        out["effective_diameter90_LCC"] = 0.0
        return out

    threshold = 0.9 * total
    cumulative = 0
    eff = 0
    for d in sorted(hist):
        cumulative += hist[d]
        if cumulative >= threshold:
            eff = d
            break
    out["effective_diameter90_LCC"] = float(eff)
    return out


def connectivity_measures(G: nx.Graph, heavy_skip: bool, max_n_for_exact: int) -> dict[str, float]:
    n = G.number_of_nodes()
    if n == 0:
        return {
            "vertex_connectivity_kappa_LCC": float("nan"),
            "edge_connectivity_lambda_LCC": float("nan"),
            "cheeger_constant_LCC": float("nan"),
        }
    out = {
        "vertex_connectivity_kappa_LCC": float("nan"),
        "edge_connectivity_lambda_LCC": float("nan"),
        "cheeger_constant_LCC": float("nan"),
    }
    if heavy_skip and n > max_n_for_exact:
        return out
    try:
        out["vertex_connectivity_kappa_LCC"] = float(nx.node_connectivity(G))
    except Exception:
        pass
    try:
        out["edge_connectivity_lambda_LCC"] = float(nx.edge_connectivity(G))
    except Exception:
        pass
    out["cheeger_constant_LCC"] = cheeger_constant(G)
    return out


def spectral_measures(G: nx.Graph, heavy_skip: bool, max_n_for_exact: int) -> dict[str, float]:
    n = G.number_of_nodes()
    empty = {
        "algebraic_connectivity_lambda2_LCC": float("nan"),
        "spectral_gap": float("nan"),
        "adjacency_spectral_radius_LCC": float("nan"),
    }
    if n == 0:
        return empty
    if heavy_skip and n > max_n_for_exact:
        return empty

    lambda2 = float("nan")
    rho = float("nan")
    spectral_gap = float("nan")

    try:
        import scipy.sparse.linalg as spla

        L = nx.laplacian_matrix(G).astype(float)
        if n >= 2:
            vals, _ = spla.eigsh(L, k=min(2, n - 1), which="SM")
            vals = np.sort(np.real(vals))
            if vals.size >= 2:
                lambda2 = float(vals[1])
            elif vals.size == 1:
                lambda2 = float(vals[0])

        A = nx.adjacency_matrix(G).astype(float)
        if n >= 2:
            vals, _ = spla.eigsh(A, k=2, which="LA")
            vals = np.sort(np.real(vals))
            rho = float(vals[-1])
            spectral_gap = float(vals[-1] - vals[-2])
        elif n == 1:
            rho = 0.0
            spectral_gap = 0.0
    except Exception:
        try:
            import numpy.linalg as LA

            Ld = nx.laplacian_matrix(G).toarray().astype(float)
            evals = np.sort(LA.eigvalsh(Ld))
            if evals.size >= 2:
                lambda2 = float(evals[1])

            Ad = nx.to_numpy_array(G, dtype=float)
            evals = np.sort(LA.eigvalsh(Ad))
            if evals.size >= 2:
                rho = float(evals[-1])
                spectral_gap = float(evals[-1] - evals[-2])
            elif evals.size == 1:
                rho = 0.0
                spectral_gap = 0.0
        except Exception:
            pass

    return {
        "algebraic_connectivity_lambda2_LCC": lambda2,
        "spectral_gap": spectral_gap,
        "adjacency_spectral_radius_LCC": rho,
    }


def homophility(data: PyGData) -> dict[str, float]:
    y = data.get("y", None)
    edge_index = data.get("edge_index", None)
    if y is None or edge_index is None or edge_index.size(1) == 0:
        return {"homophility": float("nan")}

    src = edge_index[0]
    dst = edge_index[1]
    n = int(y.size(0))

    if y.dim() == 1:
        same = (y[src] == y[dst]).to(torch.float32)
    else:
        same = (y[src] == y[dst]).all(dim=1).to(torch.float32)

    deg = torch.bincount(src, minlength=n).to(torch.float32)
    same_count = torch.bincount(src, weights=same, minlength=n)

    per_node = torch.zeros(n, dtype=torch.float32)
    mask = deg > 0
    per_node[mask] = same_count[mask] / deg[mask]
    return {"homophility": float(per_node.mean().item())}


def save_edge_homophily_sparse(data: PyGData, path: str) -> None:
    y = data.get("y", None)
    edge_index = data.get("edge_index", None)
    if y is None or edge_index is None or edge_index.size(1) == 0:
        return

    src = edge_index[0].detach().cpu().numpy().astype(np.int64, copy=False)
    dst = edge_index[1].detach().cpu().numpy().astype(np.int64, copy=False)
    n = int(y.size(0))

    if y.dim() == 1:
        vals = (y[edge_index[0]] == y[edge_index[1]]).detach().cpu().numpy().astype(np.uint8, copy=False)
    else:
        vals = (y[edge_index[0]] == y[edge_index[1]]).all(dim=1).detach().cpu().numpy().astype(np.uint8, copy=False)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    if sparse is not None:
        mat = sparse.csr_matrix((vals, (src, dst)), shape=(n, n), dtype=np.uint8)
        sparse.save_npz(path, mat)
    else:
        np.savez_compressed(path.replace(".npz", ".npz"), src=src, dst=dst, vals=vals, n=n)


# ----------------------------- Main pipeline ----------------------------- #

def compute_metrics_for_graph(
    name: str,
    G: nx.Graph | nx.DiGraph,
    data: PyGData,
    undirected_analysis: bool,
    approx_distance: bool,
    sampled_bfs_sources: int,
    heavy_skip: bool,
    max_n_for_exact: int,
    compute_directed_metrics: bool = False,
    reachability_sources: int = 128,
) -> dict[str, str | float | int]:
    Gu = ensure_undirected(G) if undirected_analysis else G

    metrics: dict[str, str | float | int] = {
        "dataset": name,
        "n_nodes": int(Gu.number_of_nodes()),
        "n_edges": int(Gu.number_of_edges()),
        "density": float(safe_density(ensure_undirected(Gu))),
    }

    metrics.update(degree_stats(ensure_undirected(Gu)))
    metrics["assortativity_degree"] = degree_assortativity(ensure_undirected(Gu))
    metrics.update(triangle_and_clustering(ensure_undirected(Gu)))

    LCC = largest_connected_component(ensure_undirected(Gu))
    metrics.update(
        distances_and_effective_diameter(
            LCC,
            approx=approx_distance,
            sampled_bfs_sources=sampled_bfs_sources,
        )
    )
    metrics.update(connectivity_measures(LCC, heavy_skip=heavy_skip, max_n_for_exact=max_n_for_exact))
    metrics.update(spectral_measures(LCC, heavy_skip=heavy_skip, max_n_for_exact=max_n_for_exact))
    metrics.update(homophility(data))

    # Directed explanatory metrics: Tier 1 + best Tier 2 choices
    # Tier 1:
    # - reverse-edge ratio
    # - directional node-role imbalance
    # - SCC fragmentation
    # - reachability asymmetry
    # Tier 2:
    # - label-flow asymmetry
    # - in/out neighborhood label JSD
    if compute_directed_metrics:
        metrics.update(reverse_edge_ratio_metrics(data))
        metrics.update(directional_node_role_metrics(data))
        metrics.update(scc_fragmentation_metrics(G))
        metrics.update(reachability_asymmetry_metrics(G, max_sources=reachability_sources))
        metrics.update(label_flow_asymmetry_metrics(data))
        metrics.update(in_out_label_jsd_metrics(data))
    else:
        nan_directed = {
            "reverse_edge_ratio": float("nan"),
            "missing_reverse_edges": float("nan"),
            "unique_directed_edges_no_self_loops": float("nan"),
            "reciprocity_ratio": float("nan"),
            "bidirectionality_gap": float("nan"),
            "in_degree_mean": float("nan"),
            "out_degree_mean": float("nan"),
            "directional_imbalance_mean": float("nan"),
            "directional_imbalance_min": float("nan"),
            "directional_imbalance_max": float("nan"),
            "fraction_near_sources": float("nan"),
            "fraction_near_sinks": float("nan"),
            "fraction_pure_sources": float("nan"),
            "fraction_pure_sinks": float("nan"),
            "n_strongly_connected_components": float("nan"),
            "largest_scc_size": float("nan"),
            "largest_scc_fraction": float("nan"),
            "n_weakly_connected_components": float("nan"),
            "largest_wcc_size": float("nan"),
            "largest_wcc_fraction": float("nan"),
            "fraction_edges_between_sccs": float("nan"),
            "condensation_n_nodes": float("nan"),
            "condensation_n_edges": float("nan"),
            "condensation_density": float("nan"),
            "reachability_asymmetry": float("nan"),
            "reachable_pairs_sampled": float("nan"),
            "asymmetric_reachable_pairs_sampled": float("nan"),
            "label_flow_asymmetry": float("nan"),
            "label_flow_n_classes": float("nan"),
            "in_out_label_jsd_mean": float("nan"),
            "in_out_label_jsd_max": float("nan"),
            "in_out_label_jsd_valid_nodes": float("nan"),
        }
        metrics.update(nan_directed)

    return metrics


def save_metrics(metrics: dict[str, str | float | int], out_dir: str, stem: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{stem}.csv")
    json_path = os.path.join(out_dir, f"{stem}.json")
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return csv_path


def main() -> None:
    p = argparse.ArgumentParser(description="Compute graph connectivity and direction-aware metrics")
    p.add_argument("-d", "--dataset", type=str, required=True)
    p.add_argument("-u", "--undirected", action="store_true", help="Treat input as undirected")
    p.add_argument("--sampled-bfs-sources", type=int, default=300)
    p.add_argument("--reachability-sources", type=int, default=128)
    p.add_argument("--dataset-directory", type=str, default="dataset")
    p.add_argument("--self-loops", action="store_true")
    p.add_argument("--transpose", action="store_true")
    p.add_argument("--approx-distance", action="store_true")
    p.add_argument("--heavy-skip", action="store_true")
    p.add_argument("--max-n-for-exact", type=int, default=5000)
    p.add_argument("--outdir", default="results/connectivity_metrics")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    print("Arguments:", args)

    dataset_name = args.dataset
    directionality = "undirected" if args.undirected else "directed"
    outdir = os.path.join(args.outdir, dataset_name)
    os.makedirs(outdir, exist_ok=True)

    dataset, evaluator = get_dataset(
        name=args.dataset,
        root_dir=args.dataset_directory,
        undirected=args.undirected,
        self_loops=args.self_loops,
        transpose=args.transpose,
    )
    data = get_data_from_loader(dataset)
    G = pyg_to_nx_minimal(data, undirected=args.undirected)

    metrics = compute_metrics_for_graph(
        dataset_name,
        G,
        data,
        undirected_analysis=True,
        approx_distance=args.approx_distance,
        sampled_bfs_sources=args.sampled_bfs_sources,
        heavy_skip=args.heavy_skip,
        max_n_for_exact=args.max_n_for_exact,
        compute_directed_metrics=not args.undirected,
        reachability_sources=args.reachability_sources,
    )
    csv_path = save_metrics(metrics, outdir, directionality)
    logging.info("Saved metrics to %s", csv_path)

    # Optional sparse edge-homophily dump
    # edge_h_path = os.path.join(outdir, f"{directionality}.npz")
    # save_edge_homophily_sparse(data, edge_h_path)

    del data
    del G
    gc.collect()


if __name__ == "__main__":
    main()