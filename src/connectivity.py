#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
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

def get_data_from_loader(dataset, batch_size: int = 1):
    """
    Same loader logic as your snippet.
    """
    data = dataset._data
    data_loader = DataLoader(
        FullBatchGraphDataset(data),
        batch_size=batch_size,
        collate_fn=lambda batch: batch[0],
    )
    return next(iter(data_loader))

# ----------------------------- Utility helpers ----------------------------- #

def gini_coefficient(x: np.ndarray) -> float:
    """Linear-memory Gini coefficient for a non-negative array."""
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



def bidirectionality_gap(data: PyGData) -> float:
    """Percentage of non-self-loop edges whose reverse edge is missing."""
    ei = get_edge_index_numpy(data)
    if ei.shape[1] == 0:
        return float("nan")

    src = ei[0]
    dst = ei[1]
    mask = src != dst
    src = src[mask]
    dst = dst[mask]
    if src.size == 0:
        return float("nan")

    if src.size == 0:
        return float("nan")

    n_base = int(max(src.max(initial=0), dst.max(initial=0))) + 1
    keys = src * n_base + dst
    rev_keys = dst * n_base + src
    unique_keys = np.unique(keys)
    unique_rev = np.unique(rev_keys)
    key_set = set(unique_keys.tolist())
    missing = sum(1 for key in unique_rev.tolist() if key not in key_set)
    return 100.0 * missing / unique_keys.size if unique_keys.size > 0 else float("nan")



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
    """Approximate Cheeger constant via random cuts to keep memory bounded."""
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


# ----------------------------- Metric routines ----------------------------- #

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
    """Save only edge-supported homophily information, never a dense N x N matrix."""
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
    compute_missing_reverse_metric: bool = False,
) -> dict[str, str | float | int]:
    Gu = ensure_undirected(G) if undirected_analysis else G

    metrics: dict[str, str | float | int] = {
        "dataset": name,
        "n_nodes": int(Gu.number_of_nodes()),
        "n_edges": int(Gu.number_of_edges()),
        "density": float(safe_density(Gu)),
        "bidirectionality_gap": bidirectionality_gap(data) if compute_missing_reverse_metric else float("nan"),
    }

    metrics.update(degree_stats(Gu))
    metrics["assortativity_degree"] = degree_assortativity(Gu)
    metrics.update(triangle_and_clustering(Gu))

    LCC = largest_connected_component(Gu)
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
    p = argparse.ArgumentParser(description="Compute graph metrics and save compact outputs")
    p.add_argument("-d", "--dataset", type=str, required=True)
    p.add_argument("-u", "--undirected", action="store_true", help="Treat input as undirected")
    p.add_argument("--sampled-bfs-sources", type=int, default=300)
    p.add_argument("--dataset-directory", type=str, default="dataset")
    p.add_argument("--self-loops", action="store_true")
    p.add_argument("--transpose", action="store_true")
    p.add_argument("--approx-distance", action="store_true")
    p.add_argument("--heavy-skip", action="store_true")
    p.add_argument("--max-n-for-exact", type=int, default=5000)
    p.add_argument("--outdir", default="results/connectivity_metrics")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    dataset_name = args.dataset
    directionality = "undirected" if args.undirected else "directed"
    outdir = os.path.join(args.outdir, dataset_name)
    os.makedirs(outdir, exist_ok=True)

    # ---------------- Baseline: same loading logic as training code ----------------
    dataset, evaluator = get_dataset(
        name=args.dataset,
        root_dir=args.dataset_directory,
        undirected=args.undirected,
        self_loops=args.self_loops,
        transpose=args.transpose,
    )
    data = get_data_from_loader(dataset)
    G = pyg_to_nx_minimal(data, undirected=args.undirected)
    baseline_metrics = compute_metrics_for_graph(
        dataset_name,
        G,
        data,
        undirected_analysis=True,
        approx_distance=args.approx_distance,
        sampled_bfs_sources=args.sampled_bfs_sources,
        heavy_skip=args.heavy_skip,
        max_n_for_exact=args.max_n_for_exact,
        compute_missing_reverse_metric=not args.undirected,
    )
    baseline_csv = save_metrics(baseline_metrics, outdir, directionality)
    logging.info("Saved baseline metrics to %s", baseline_csv)

    # edge_h_path = os.path.join(args.outdir, dataset_name, f"{directionality}.npz")
    # save_edge_homophily_sparse(data, edge_h_path)


if __name__ == "__main__":
    main()
