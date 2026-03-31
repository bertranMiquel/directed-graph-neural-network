#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, linregress, pearsonr, spearmanr

try:
    import seaborn as sns
except Exception:
    sns = None

try:
    from sklearn.linear_model import LinearRegression
except Exception:
    LinearRegression = None

log = logging.getLogger("relation_from_table")

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

CONNECTIVITY_METRICS = [
    "bidirectionality_gap",
    "directional_imbalance_mean",
    "fraction_near_sinks",
    "fraction_near_sources",
    "fraction_edges_between_sccs",
    "condensation_density",
    "in_out_label_jsd_mean",
    "density",
    "homophility",
    "degree_mean",
    "degree_gini",
    "assortativity_degree",
    "triangles_avg_per_node",
    "clustering_coefficient_avg",
    "transitivity",
    "radius_LCC",
    "eccentricity_mean_LCC",
    "cheeger_constant_LCC",
    "algebraic_connectivity_lambda2_LCC",
    "spectral_gap",
]

MODEL_VARIANTS = {
    "gcn": [
        "dir-gcn_alpha_0.0",
        "dir-gcn_alpha_1.0",
        "dir-gcn_alpha_0.5",
    ],
    "sage": [
        "dir-sage_alpha_0.0",
        "dir-sage_alpha_1.0",
        "dir-sage_alpha_0.5",
    ],
    "gat": [
        "dir-gat_alpha_0.0",
        "dir-gat_alpha_1.0",
        "dir-gat_alpha_0.5",
    ],
}

BASE_TO_DIR_PREFIX = {
    "gcn": "dir-gcn",
    "sage": "dir-sage",
    "gat": "dir-gat",
}

# ---------------------------------------------------------------------
# Accuracy table from the image
# ---------------------------------------------------------------------

ACCURACY_TABLE = {
    "citeseer_full": {
        "gcn": (93.37, 0.22),
        "dir-gcn_alpha_0.0": (93.21, 0.41),
        "dir-gcn_alpha_1.0": (93.44, 0.59),
        "dir-gcn_alpha_0.5": (92.97, 0.31),
        "sage": (94.15, 0.61),
        "dir-sage_alpha_0.0": (94.05, 0.25),
        "dir-sage_alpha_1.0": (93.97, 0.67),
        "dir-sage_alpha_0.5": (94.14, 0.65),
        "gat": (94.53, 0.48),
        "dir-gat_alpha_0.0": (94.48, 0.52),
        "dir-gat_alpha_1.0": (94.08, 0.69),
        "dir-gat_alpha_0.5": (94.12, 0.49),
    },
    "cora_ml": {
        "gcn": (84.37, 1.52),
        "dir-gcn_alpha_0.0": (84.45, 1.69),
        "dir-gcn_alpha_1.0": (83.81, 1.44),
        "dir-gcn_alpha_0.5": (84.21, 2.48),
        "sage": (86.01, 1.56),
        "dir-sage_alpha_0.0": (85.84, 2.09),
        "dir-sage_alpha_1.0": (85.73, 0.35),
        "dir-sage_alpha_0.5": (85.81, 1.18),
        "gat": (86.44, 1.45),
        "dir-gat_alpha_0.0": (86.13, 1.58),
        "dir-gat_alpha_1.0": (86.21, 1.40),
        "dir-gat_alpha_0.5": (86.05, 1.71),
    },
    "ogbn-arxiv": {
        "gcn": (68.39, 0.01),
        "dir-gcn_alpha_0.0": (23.70, 0.20),
        "dir-gcn_alpha_1.0": (62.93, 0.21),
        "dir-gcn_alpha_0.5": (66.66, 0.02),
        "sage": (67.78, 0.07),
        "dir-sage_alpha_0.0": (52.08, 0.17),
        "dir-sage_alpha_1.0": (65.14, 0.03),
        "dir-sage_alpha_0.5": (65.06, 0.28),
        "gat": (69.60, 0.01),
        "dir-gat_alpha_0.0": (52.57, 0.05),
        "dir-gat_alpha_1.0": (66.50, 0.16),
        "dir-gat_alpha_0.5": (66.44, 0.41),
    },
    "chameleon": {
        "gcn": (71.12, 2.28),
        "dir-gcn_alpha_0.0": (29.78, 1.27),
        "dir-gcn_alpha_1.0": (78.77, 1.72),
        "dir-gcn_alpha_0.5": (72.37, 1.50),
        "sage": (61.14, 2.00),
        "dir-sage_alpha_0.0": (48.33, 2.40),
        "dir-sage_alpha_1.0": (64.47, 2.27),
        "dir-sage_alpha_0.5": (60.22, 1.16),
        "gat": (66.82, 2.56),
        "dir-gat_alpha_0.0": (40.44, 3.11),
        "dir-gat_alpha_1.0": (71.40, 1.63),
        "dir-gat_alpha_0.5": (55.57, 1.02),
    },
    "squirrel": {
        "gcn": (62.71, 2.27),
        "dir-gcn_alpha_0.0": (33.03, 0.78),
        "dir-gcn_alpha_1.0": (74.43, 0.74),
        "dir-gcn_alpha_0.5": (67.82, 1.73),
        "sage": (42.64, 1.72),
        "dir-sage_alpha_0.0": (35.31, 0.52),
        "dir-sage_alpha_1.0": (46.05, 1.16),
        "dir-sage_alpha_0.5": (43.29, 1.04),
        "gat": (56.49, 1.73),
        "dir-gat_alpha_0.0": (28.28, 1.02),
        "dir-gat_alpha_1.0": (67.53, 1.04),
        "dir-gat_alpha_0.5": (37.75, 1.24),
    },
    "arxiv_year": {
        "gcn": (46.28, 0.39),
        "dir-gcn_alpha_0.0": (50.51, 0.45),
        "dir-gcn_alpha_1.0": (50.52, 0.09),
        "dir-gcn_alpha_0.5": (59.56, 0.16),
        "sage": (44.05, 0.02),
        "dir-sage_alpha_0.0": (47.45, 0.32),
        "dir-sage_alpha_1.0": (50.37, 0.09),
        "dir-sage_alpha_0.5": (55.76, 0.10),
        "gat": (45.30, 0.23),
        "dir-gat_alpha_0.0": (46.01, 0.06),
        "dir-gat_alpha_1.0": (51.58, 0.19),
        "dir-gat_alpha_0.5": (54.47, 0.14),
    },
    "snap_patents": {
        "gcn": (51.02, 0.07),
        "dir-gcn_alpha_0.0": (51.71, 0.06),
        "dir-gcn_alpha_1.0": (62.24, 0.04),
        "dir-gcn_alpha_0.5": (71.32, 0.06),
        "sage": (52.55, 0.10),
        "dir-sage_alpha_0.0": (52.53, 0.03),
        "dir-sage_alpha_1.0": (61.59, 0.05),
        "dir-sage_alpha_0.5": (70.26, 0.14),
        "gat": (np.nan, np.nan),
        "dir-gat_alpha_0.0": (np.nan, np.nan),
        "dir-gat_alpha_1.0": (np.nan, np.nan),
        "dir-gat_alpha_0.5": (np.nan, np.nan),
    },
    "directed-roman-empire": {
        "gcn": (56.23, 0.37),
        "dir-gcn_alpha_0.0": (42.69, 0.41),
        "dir-gcn_alpha_1.0": (45.52, 0.14),
        "dir-gcn_alpha_0.5": (74.54, 0.71),
        "sage": (72.05, 0.41),
        "dir-sage_alpha_0.0": (76.47, 0.14),
        "dir-sage_alpha_1.0": (68.81, 0.48),
        "dir-sage_alpha_0.5": (79.10, 0.19),
        "gat": (49.18, 1.35),
        "dir-gat_alpha_0.0": (53.58, 2.51),
        "dir-gat_alpha_1.0": (56.24, 0.41),
        "dir-gat_alpha_0.5": (72.25, 0.04),
    },
}

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def set_publication_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 8,
        "legend.title_fontsize": 9,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.grid": False,
        "lines.linewidth": 1.4,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })
    if sns is not None:
        sns.set_theme(style="white", context="paper")


def safe_mkdir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def clean_metric_name(x: str) -> str:
    return x.replace("_", " ").title()


def pretty_model_pair(model_pair: str) -> str:
    mapping = {
        "gcn_vs_dir-gcn": "GCN vs best DirGCN",
        "sage_vs_dir-sage": "SAGE vs best DirSAGE",
        "gat_vs_dir-gat": "GAT vs best DirGAT",
    }
    return mapping.get(model_pair, model_pair)


def significance_stars(p: float) -> str:
    if pd.isna(p):
        return "ns"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"


def get_dataset_palette(datasets: list[str]) -> dict[str, tuple]:
    datasets = sorted(datasets)
    if sns is not None:
        palette = sns.color_palette("tab10", n_colors=max(len(datasets), 1))
    else:
        cmap = plt.get_cmap("tab10")
        palette = [cmap(i) for i in range(max(len(datasets), 1))]
    return {ds: palette[i] for i, ds in enumerate(datasets)}


def add_panel_stats_box(ax, row: pd.Series | None) -> None:
    if row is None:
        return
    txt = (
        f"n={int(row['n'])}\n"
        f"Spearman={row['spearman_r']:.3f}\n"
        f"p={row['spearman_p']:.2e}\n"
        f"R²={row['r2']:.3f}"
    )
    ax.text(
        0.03,
        0.97,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.75"},
    )


def _corr_pair(x: np.ndarray, y: np.ndarray) -> dict:
    valid = np.isfinite(x) & np.isfinite(y)
    n = int(valid.sum())

    if n < 3:
        return {
            "pearson_r": np.nan, "pearson_p": np.nan,
            "spearman_r": np.nan, "spearman_p": np.nan,
            "kendall_tau": np.nan, "kendall_p": np.nan,
            "slope": np.nan, "intercept": np.nan, "r2": np.nan,
            "n": n,
        }

    xv = x[valid]
    yv = y[valid]

    try:
        pr, pp = pearsonr(xv, yv)
    except Exception:
        pr, pp = np.nan, np.nan
    try:
        sr, sp = spearmanr(xv, yv)
    except Exception:
        sr, sp = np.nan, np.nan
    try:
        kr, kp = kendalltau(xv, yv)
    except Exception:
        kr, kp = np.nan, np.nan
    try:
        lr = linregress(xv, yv)
        slope, intercept, r_val = lr.slope, lr.intercept, lr.rvalue
        r2 = r_val ** 2
    except Exception:
        slope, intercept, r2 = np.nan, np.nan, np.nan

    return {
        "pearson_r": pr, "pearson_p": pp,
        "spearman_r": sr, "spearman_p": sp,
        "kendall_tau": kr, "kendall_p": kp,
        "slope": slope, "intercept": intercept, "r2": r2,
        "n": n,
    }


def _draw_regression(ax, x: np.ndarray, y: np.ndarray) -> None:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return
    xv = x[valid]
    yv = y[valid]
    try:
        lr = linregress(xv, yv)
        xs = np.linspace(xv.min(), xv.max(), 200)
        ys = lr.intercept + lr.slope * xs
        ax.plot(xs, ys, linestyle=(0, (4, 2)), linewidth=1.3, color="0.2", alpha=0.9, zorder=2)
    except Exception:
        return


def _fit_linear_model(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> dict:
    valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    Xv = X[valid]
    yv = y[valid]
    n = len(yv)
    p = Xv.shape[1]

    out = {
        "n": n,
        "p": p,
        "r2": np.nan,
        "adj_r2": np.nan,
        "intercept": np.nan,
    }
    for name in feature_names:
        out[f"coef_{name}"] = np.nan

    if LinearRegression is None or n <= p + 1:
        return out

    model = LinearRegression()
    model.fit(Xv, yv)
    yhat = model.predict(Xv)

    ss_res = np.sum((yv - yhat) ** 2)
    ss_tot = np.sum((yv - np.mean(yv)) ** 2)
    if ss_tot == 0:
        return out

    r2 = 1 - ss_res / ss_tot
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    out["r2"] = float(r2)
    out["adj_r2"] = float(adj_r2)
    out["intercept"] = float(model.intercept_)
    for name, coef in zip(feature_names, model.coef_):
        out[f"coef_{name}"] = float(coef)
    return out


def _figure_legend_handles(datasets: list[str], palette: dict[str, tuple]):
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=6.5,
               markerfacecolor=palette[ds], markeredgecolor=palette[ds], label=ds)
        for ds in datasets
    ]
    return handles, datasets


def _place_title_and_legend(fig, title: str, legend_handles, legend_labels) -> None:
    fig.suptitle(title, x=0.35, y=0.975, ha="center", fontweight="bold")
    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            title="Dataset",
            loc="upper left",
            bbox_to_anchor=(0.58, 0.985),
            ncol=2,
            frameon=False,
            fontsize=8,
            title_fontsize=9,
            columnspacing=1.0,
            handletextpad=0.35,
            borderaxespad=0.0,
        )


def _scatter_points(ax, plot_sub: pd.DataFrame, x_col: str, y_col: str, palette: dict[str, tuple]) -> None:
    for ds, g in plot_sub.groupby("dataset"):
        ax.scatter(
            g[x_col],
            g[y_col],
            label=ds,
            color=palette[ds],
            s=48,
            alpha=0.95,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )

# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_connectivity_metrics(metrics_dir: str | Path) -> pd.DataFrame:
    rows: list[dict] = []
    metrics_dir = Path(metrics_dir)
    if not metrics_dir.exists():
        raise FileNotFoundError(f"Metrics directory not found: {metrics_dir}")

    for dataset_dir in sorted(metrics_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        path = dataset_dir / "directed.json"
        if not path.exists():
            log.warning(f"Missing connectivity file: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        row = {"dataset": dataset_dir.name}
        for m in CONNECTIVITY_METRICS:
            row[m] = data.get(m, np.nan)
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("dataset").reset_index(drop=True)
    return df


def load_accuracy_table() -> pd.DataFrame:
    rows = []
    for dataset, models in ACCURACY_TABLE.items():
        for model, (mean, std) in models.items():
            rows.append({
                "dataset": dataset,
                "model": model,
                "test_acc_mean": float(mean) if pd.notna(mean) else np.nan,
                "test_acc_std": float(std) if pd.notna(std) else np.nan,
                "source": "image_table",
            })
    return pd.DataFrame(rows).sort_values(["dataset", "model"]).reset_index(drop=True)

# ---------------------------------------------------------------------
# Delta builders
# ---------------------------------------------------------------------

def build_delta_accuracy_df_all_alphas(acc_df: pd.DataFrame) -> pd.DataFrame:
    if acc_df.empty:
        return pd.DataFrame()

    pivot_mean = acc_df.pivot(index="dataset", columns="model", values="test_acc_mean")
    pivot_std = acc_df.pivot(index="dataset", columns="model", values="test_acc_std")

    rows: list[dict] = []
    for dataset in pivot_mean.index:
        for base_model, directed_variants in MODEL_VARIANTS.items():
            if base_model not in pivot_mean.columns:
                continue
            base_acc = pivot_mean.loc[dataset, base_model]
            base_std = pivot_std.loc[dataset, base_model] if base_model in pivot_std.columns else np.nan
            if pd.isna(base_acc):
                continue

            for dir_model in directed_variants:
                if dir_model not in pivot_mean.columns:
                    continue
                dir_acc = pivot_mean.loc[dataset, dir_model]
                dir_std = pivot_std.loc[dataset, dir_model] if dir_model in pivot_std.columns else np.nan
                if pd.isna(dir_acc):
                    continue

                rows.append({
                    "dataset": dataset,
                    "base_model": base_model,
                    "directed_model": dir_model,
                    "model_pair": f"{base_model}_vs_{dir_model}",
                    "base_acc": float(base_acc),
                    "directed_acc": float(dir_acc),
                    "delta_acc": float(dir_acc - base_acc),
                    "base_acc_std": float(base_std) if pd.notna(base_std) else np.nan,
                    "directed_acc_std": float(dir_std) if pd.notna(dir_std) else np.nan,
                    "alpha": float(dir_model.split("_")[-1]),
                    "analysis_mode": "all_alphas",
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["base_model", "directed_model", "dataset"]).reset_index(drop=True)
    return df


def build_delta_accuracy_df_best_alpha_per_base_model(acc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Select the best alpha independently for each base-model family:
      - gcn vs best dir-gcn
      - sage vs best dir-sage
      - gat vs best dir-gat

    The selection is done per dataset and per base model, using the directed
    configuration with the highest directed accuracy.
    """
    if acc_df.empty:
        return pd.DataFrame()

    pivot_mean = acc_df.pivot(index="dataset", columns="model", values="test_acc_mean")
    pivot_std = acc_df.pivot(index="dataset", columns="model", values="test_acc_std")

    best_rows: list[dict] = []

    for dataset in pivot_mean.index:
        for base_model, directed_variants in MODEL_VARIANTS.items():
            if base_model not in pivot_mean.columns:
                continue

            base_acc = pivot_mean.loc[dataset, base_model]
            base_std = pivot_std.loc[dataset, base_model] if base_model in pivot_std.columns else np.nan
            if pd.isna(base_acc):
                continue

            candidate_rows = []
            for dir_model in directed_variants:
                if dir_model not in pivot_mean.columns:
                    continue
                dir_acc = pivot_mean.loc[dataset, dir_model]
                dir_std = pivot_std.loc[dataset, dir_model] if dir_model in pivot_std.columns else np.nan
                if pd.isna(dir_acc):
                    continue

                candidate_rows.append({
                    "dataset": dataset,
                    "base_model": base_model,
                    "directed_model": dir_model,
                    "directed_family": BASE_TO_DIR_PREFIX[base_model],
                    "model_pair": f"{base_model}_vs_{BASE_TO_DIR_PREFIX[base_model]}",
                    "base_acc": float(base_acc),
                    "directed_acc": float(dir_acc),
                    "delta_acc": float(dir_acc - base_acc),
                    "base_acc_std": float(base_std) if pd.notna(base_std) else np.nan,
                    "directed_acc_std": float(dir_std) if pd.notna(dir_std) else np.nan,
                    "alpha": float(dir_model.split("_")[-1]),
                    "analysis_mode": "best_alpha_per_base_model",
                })

            if not candidate_rows:
                continue

            cand_df = pd.DataFrame(candidate_rows)
            best_idx = cand_df["directed_acc"].idxmax()
            best_row = cand_df.loc[best_idx].to_dict()
            best_row["selected_by"] = "max_directed_acc_within_base_model_family"
            best_rows.append(best_row)

    df = pd.DataFrame(best_rows)
    if not df.empty:
        df = df.sort_values(["base_model", "dataset"]).reset_index(drop=True)
    return df

# ---------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------

def build_master_df(metrics_df: pd.DataFrame, delta_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty or delta_df.empty:
        return pd.DataFrame()
    return pd.merge(delta_df, metrics_df, on="dataset", how="inner")


def compute_connectivity_correlations(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    for m1, m2 in combinations(CONNECTIVITY_METRICS, 2):
        sub = df[[m1, m2]].dropna()
        if len(sub) < 3:
            continue
        x = sub[m1].values.astype(float)
        y = sub[m2].values.astype(float)
        stats = _corr_pair(x, y)
        rows.append({"metric_x": m1, "metric_y": m2, **stats})

    corr_df = pd.DataFrame(rows)
    corr_df.to_csv(out_dir / "connectivity_metric_correlations.csv", index=False)
    return corr_df


def compute_pooled_correlations(master_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []

    for model_pair, grp in master_df.groupby("model_pair"):
        y = grp["delta_acc"].values.astype(float)
        for metric in CONNECTIVITY_METRICS:
            x = grp[metric].values.astype(float)
            row = {"scope": "pooled_by_model_pair", "model_pair": model_pair, "metric": metric}
            row.update(_corr_pair(x, y))
            rows.append(row)

    y_all = master_df["delta_acc"].values.astype(float)
    for metric in CONNECTIVITY_METRICS:
        x_all = master_df[metric].values.astype(float)
        row = {"scope": "pooled_all_models", "model_pair": "all", "metric": metric}
        row.update(_corr_pair(x_all, y_all))
        rows.append(row)

    corr_df = pd.DataFrame(rows)
    if not corr_df.empty:
        corr_df = corr_df.sort_values(
            ["scope", "model_pair", "spearman_r"],
            key=lambda s: s.abs() if s.name == "spearman_r" else s,
            ascending=[True, True, False],
        ).reset_index(drop=True)
    corr_df.to_csv(out_dir / "pooled_correlations.csv", index=False)
    return corr_df


def compute_per_model_correlations(master_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for model_pair, grp in master_df.groupby("model_pair"):
        y = grp["delta_acc"].values.astype(float)
        for metric in CONNECTIVITY_METRICS:
            x = grp[metric].values.astype(float)
            row = {"model_pair": model_pair, "metric": metric}
            row.update(_corr_pair(x, y))
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            ["model_pair", "spearman_r"],
            key=lambda s: s.abs() if s.name == "spearman_r" else s,
            ascending=[True, False],
        ).reset_index(drop=True)
    df.to_csv(out_dir / "per_model_correlations.csv", index=False)
    return df


def compute_single_metric_regressions(master_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []

    y_all = master_df["delta_acc"].values.astype(float)
    for metric in CONNECTIVITY_METRICS:
        res = _fit_linear_model(master_df[[metric]].values.astype(float), y_all, [metric])
        rows.append({
            "scope": "pooled_all_models",
            "model_pair": "all",
            "model_type": "single_metric",
            "metrics": metric,
            **res,
        })

    for model_pair, grp in master_df.groupby("model_pair"):
        y = grp["delta_acc"].values.astype(float)
        for metric in CONNECTIVITY_METRICS:
            res = _fit_linear_model(grp[[metric]].values.astype(float), y, [metric])
            rows.append({
                "scope": "per_model_pair",
                "model_pair": model_pair,
                "model_type": "single_metric",
                "metrics": metric,
                **res,
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "single_metric_models.csv", index=False)
    return df


def compute_second_order_models(master_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []

    pooled = master_df.copy()
    y = pooled["delta_acc"].values.astype(float)
    for m1, m2 in combinations(CONNECTIVITY_METRICS, 2):
        X_add = pooled[[m1, m2]].values.astype(float)
        rows.append({
            "scope": "pooled_all_models",
            "model_pair": "all",
            "model_type": "pair_additive",
            "metrics": f"{m1} + {m2}",
            **_fit_linear_model(X_add, y, [m1, m2]),
        })

        X_int = pooled[[m1, m2]].copy()
        X_int["interaction"] = pooled[m1] * pooled[m2]
        rows.append({
            "scope": "pooled_all_models",
            "model_pair": "all",
            "model_type": "pair_interaction",
            "metrics": f"{m1} + {m2} + {m1}*{m2}",
            **_fit_linear_model(X_int.values.astype(float), y, [m1, m2, f"{m1}*{m2}"]),
        })

    for model_pair, grp in master_df.groupby("model_pair"):
        y = grp["delta_acc"].values.astype(float)
        for m1, m2 in combinations(CONNECTIVITY_METRICS, 2):
            X_add = grp[[m1, m2]].values.astype(float)
            rows.append({
                "scope": "per_model_pair",
                "model_pair": model_pair,
                "model_type": "pair_additive",
                "metrics": f"{m1} + {m2}",
                **_fit_linear_model(X_add, y, [m1, m2]),
            })

            X_int = grp[[m1, m2]].copy()
            X_int["interaction"] = grp[m1] * grp[m2]
            rows.append({
                "scope": "per_model_pair",
                "model_pair": model_pair,
                "model_type": "pair_interaction",
                "metrics": f"{m1} + {m2} + {m1}*{m2}",
                **_fit_linear_model(X_int.values.astype(float), y, [m1, m2, f"{m1}*{m2}"]),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "second_order_models.csv", index=False)
    return df

# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
import math


def _make_metric_grid(
    n_metrics: int,
    max_cols: int = 4,
    panel_w: float = 4.6,
    panel_h: float = 4.1,
    sharey: bool = False,
):
    n_cols = min(max_cols, max(1, n_metrics))
    n_rows = math.ceil(n_metrics / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(panel_w * n_cols, panel_h * n_rows),
        sharey=sharey,
        squeeze=False,
    )
    axes_flat = axes.ravel()
    return fig, axes, axes_flat, n_rows, n_cols


def _hide_unused_axes(axes_flat, n_used: int) -> None:
    for ax in axes_flat[n_used:]:
        ax.set_visible(False)


def _place_title_and_legend_grid(fig, title: str, legend_handles, legend_labels, n_cols: int) -> None:
    fig.suptitle(title, x=0.36, y=0.985, ha="center", fontweight="bold")
    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            title="Dataset",
            loc="upper left",
            bbox_to_anchor=(0.73, 0.99),
            ncol=2,
            frameon=False,
            fontsize=8,
            title_fontsize=9,
            columnspacing=1.0,
            handletextpad=0.35,
            borderaxespad=0.0,
        )


def plot_scatter_by_model(master_df: pd.DataFrame, corr_df: pd.DataFrame, out_dir: Path, title_suffix: str) -> None:
    plot_dir = safe_mkdir(out_dir / "scatter_by_model")
    datasets = sorted(master_df["dataset"].dropna().unique().tolist())
    palette = get_dataset_palette(datasets)
    legend_handles, legend_labels = _figure_legend_handles(datasets, palette)

    n_metrics = len(CONNECTIVITY_METRICS)

    for model_pair, grp in master_df.groupby("model_pair"):
        sub_corr = corr_df[corr_df["model_pair"] == model_pair].copy()
        if sub_corr.empty:
            continue

        fig, axes_flat, n_rows, n_cols = _make_metric_grid(
            n_items=n_metrics,
            max_cols=4,
            panel_w=4.8,
            panel_h=4.3,
            sharey=False,
        )

        for i, metric in enumerate(CONNECTIVITY_METRICS):
            ax = axes_flat[i]
            plot_sub = grp.dropna(subset=[metric, "delta_acc", "dataset"]).copy()
            if plot_sub.empty:
                ax.set_visible(False)
                continue

            x = plot_sub[metric].values.astype(float)
            y = plot_sub["delta_acc"].values.astype(float)

            _scatter_points(ax, plot_sub, metric, "delta_acc", palette)
            _draw_regression(ax, x, y)
            ax.axhline(0, color="0.8", linestyle=":", linewidth=1.0, zorder=1)

            row = sub_corr[sub_corr["metric"] == metric]
            row_series = row.iloc[0] if not row.empty else None
            add_panel_stats_box(ax, row_series)

            ax.set_title(clean_metric_name(metric))
            ax.set_xlabel(clean_metric_name(metric))
            ax.set_ylabel("Δ Accuracy (Directed - Base)")

        _hide_unused_axes(axes_flat, n_metrics)
        _place_title_and_legend_grid(
            fig,
            f"Connectivity vs Δ Accuracy — {pretty_model_pair(model_pair)} — {title_suffix}",
            legend_handles,
            legend_labels,
            n_cols=n_cols,
        )

        fig.subplots_adjust(
            left=0.06,
            right=0.98,
            bottom=0.07,
            top=0.88,
            wspace=0.28,
            hspace=0.38,
        )
        fig.savefig(plot_dir / f"scatter_{model_pair}.png", bbox_inches="tight")
        plt.close(fig)


def plot_scatter_all_models(master_df: pd.DataFrame, corr_df: pd.DataFrame, out_dir: Path, title_suffix: str) -> None:
    plot_dir = safe_mkdir(out_dir / "scatter_all_models")
    sub_corr = corr_df[
        (corr_df["scope"] == "pooled_all_models") & (corr_df["model_pair"] == "all")
    ].copy()

    datasets = sorted(master_df["dataset"].dropna().unique().tolist())
    palette = get_dataset_palette(datasets)
    legend_handles, legend_labels = _figure_legend_handles(datasets, palette)

    n_metrics = len(CONNECTIVITY_METRICS)

    # fig, axes, axes_flat, n_rows, n_cols = _make_metric_grid(
    fig, axes_flat, n_rows, n_cols = _make_metric_grid(
        n_items=n_metrics,
        max_cols=4,
        panel_w=4.8,
        panel_h=4.3,
        sharey=False,
    )

    for i, metric in enumerate(CONNECTIVITY_METRICS):
        ax = axes_flat[i]
        plot_sub = master_df.dropna(subset=[metric, "delta_acc", "dataset"]).copy()
        if plot_sub.empty:
            ax.set_visible(False)
            continue

        x = plot_sub[metric].values.astype(float)
        y = plot_sub["delta_acc"].values.astype(float)

        _scatter_points(ax, plot_sub, metric, "delta_acc", palette)
        _draw_regression(ax, x, y)
        ax.axhline(0, color="0.8", linestyle=":", linewidth=1.0, zorder=1)

        row = sub_corr[sub_corr["metric"] == metric]
        row_series = row.iloc[0] if not row.empty else None
        add_panel_stats_box(ax, row_series)

        ax.set_title(clean_metric_name(metric))
        ax.set_xlabel(clean_metric_name(metric))
        ax.set_ylabel("Δ Accuracy (Directed - Base)")

    _hide_unused_axes(axes_flat, n_metrics)
    _place_title_and_legend_grid(
        fig,
        f"Connectivity vs Δ Accuracy — All Model Pairs — {title_suffix}",
        legend_handles,
        legend_labels,
        n_cols=n_cols,
    )

    fig.subplots_adjust(
        left=0.06,
        right=0.98,
        bottom=0.07,
        top=0.88,
        wspace=0.28,
        hspace=0.38,
    )
    fig.savefig(plot_dir / "scatter_all_model_pairs.png", bbox_inches="tight")
    plt.close(fig)

def plot_heatmap_per_model(corr_df: pd.DataFrame, out_dir: Path, title_suffix: str) -> None:
    if sns is None:
        log.info("Seaborn not available; skipping heatmaps.")
        return

    plot_dir = safe_mkdir(out_dir / "heatmaps")
    sub = corr_df[corr_df["scope"] == "pooled_by_model_pair"].copy()
    sub = sub[sub["spearman_r"].notna()].copy()
    if sub.empty:
        return

    pivot = sub.pivot(index="metric", columns="model_pair", values="spearman_r")
    plt.figure(figsize=(max(8.0, 1.2 * pivot.shape[1]), 4.2))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "Spearman r"},
    )
    plt.title(f"Connectivity Metric vs Δ Accuracy — {title_suffix}", fontweight="bold", pad=10)
    plt.ylabel("Connectivity Metric")
    plt.xlabel("Model Pair")
    plt.tight_layout()
    plt.savefig(plot_dir / "heatmap_spearman_per_model.png", bbox_inches="tight")
    plt.close()

import math
import textwrap


def _pretty_metric_label(metric: str, width: int = 18) -> str:
    label = metric.replace("_", " ")
    return "\n".join(textwrap.wrap(label, width=width, break_long_words=False))


def _make_metric_grid(
    n_items: int,
    max_cols: int = 4,
    panel_w: float = 4.8,
    panel_h: float = 4.2,
    sharex: bool = False,
    sharey: bool = False,
):
    n_cols = min(max_cols, max(1, n_items))
    n_rows = math.ceil(n_items / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(panel_w * n_cols, panel_h * n_rows),
        squeeze=False,
        sharex=sharex,
        sharey=sharey,
    )
    return fig, axes.ravel(), n_rows, n_cols


def _hide_unused_axes(axes_flat, n_used: int) -> None:
    for ax in axes_flat[n_used:]:
        ax.set_visible(False)


def _significance_color(p: float) -> str:
    if pd.isna(p):
        return "0.75"
    if p < 0.05:
        return "#2a7fff"
    return "0.75"


def _significance_label(p: float) -> str:
    if pd.isna(p):
        return "n/a"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"


def plot_connectivity_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    if sns is None:
        log.info("Seaborn not available; skipping connectivity heatmap.")
        return

    sub = df[CONNECTIVITY_METRICS].dropna()
    if len(sub) < 2:
        return

    corr = sub.corr(method="spearman")
    n_metrics = corr.shape[0]

    # Larger figure for many metrics
    fig_w = max(10.0, 0.75 * n_metrics + 4.0)
    fig_h = max(8.0, 0.75 * n_metrics + 2.0)

    # Avoid unreadable white-number clutter when too many metrics
    annotate = n_metrics <= 12
    annot_kws = {"size": 8} if annotate else None

    xlabels = [_pretty_metric_label(c, width=16) for c in corr.columns]
    ylabels = [_pretty_metric_label(i, width=20) for i in corr.index]

    plt.figure(figsize=(fig_w, fig_h))
    ax = sns.heatmap(
        corr,
        annot=annotate,
        fmt=".2f" if annotate else "",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Spearman r", "shrink": 0.9},
        annot_kws=annot_kws,
        square=True,
    )

    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_yticklabels(ylabels, rotation=0)
    ax.set_title("Connectivity Metrics Correlation", fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(out_dir / "connectivity_heatmap.png", bbox_inches="tight")
    plt.close()


def plot_significance_bars(corr_df: pd.DataFrame, out_dir: Path, title_suffix: str) -> None:
    """
    One single overview figure with all metrics together.
    One panel per metric, max 4 panels per row.
    Bars are colored by significance.
    """
    plot_dir = safe_mkdir(out_dir / "significance")
    sub = corr_df[corr_df["scope"] == "pooled_by_model_pair"].copy()
    sub = sub[sub["spearman_r"].notna()].copy()
    if sub.empty:
        return

    metrics = [m for m in CONNECTIVITY_METRICS if m in sub["metric"].unique()]
    if not metrics:
        return

    # Sort panels so the strongest metric appears first
    metric_strength = (
        sub.groupby("metric")["spearman_r"]
        .apply(lambda s: np.nanmax(np.abs(s.values)) if len(s) else np.nan)
        .sort_values(ascending=False)
    )
    metrics = [m for m in metric_strength.index.tolist() if m in metrics]

    fig, axes_flat, n_rows, n_cols = _make_metric_grid(
        n_items=len(metrics),
        max_cols=4,
        panel_w=5.2,
        panel_h=4.5,
        sharey=True,
    )

    y_lim = 1.0
    if not sub["spearman_r"].dropna().empty:
        y_lim = max(0.2, min(1.0, 1.10 * np.abs(sub["spearman_r"]).max()))

    for i, metric in enumerate(metrics):
        ax = axes_flat[i]
        mdf = sub[sub["metric"] == metric].copy()
        if mdf.empty:
            ax.set_visible(False)
            continue

        mdf = mdf.sort_values("spearman_r", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
        colors = [_significance_color(p) for p in mdf["spearman_p"]]

        xpos = np.arange(len(mdf))
        ax.bar(
            xpos,
            mdf["spearman_r"].values,
            color=colors,
            edgecolor="black",
            linewidth=0.7,
        )
        ax.axhline(0, color="black", linewidth=0.9)
        ax.set_ylim(-y_lim, y_lim)

        for j, (_, row) in enumerate(mdf.iterrows()):
            label = _significance_label(row["spearman_p"])
            y = row["spearman_r"]
            offset = 0.03 * y_lim
            ax.text(
                j,
                y + offset if y >= 0 else y - offset,
                label,
                ha="center",
                va="bottom" if y >= 0 else "top",
                fontsize=9,
                fontweight="bold",
            )

        ax.set_title(_pretty_metric_label(clean_metric_name(metric), width=22), fontweight="bold")
        ax.set_xticks(xpos)
        ax.set_xticklabels(
            [pretty_model_pair(x) for x in mdf["model_pair"]],
            rotation=28,
            ha="right",
        )
        ax.set_ylabel("Spearman r")
        ax.grid(False)

    _hide_unused_axes(axes_flat, len(metrics))

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#2a7fff", edgecolor="black", label="p < 0.05"),
        Patch(facecolor="0.75", edgecolor="black", label="p ≥ 0.05"),
    ]

    fig.suptitle(
        f"Spearman significance overview — {title_suffix}",
        y=0.992,
        fontweight="bold",
        fontsize=18,
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
        frameon=False,
        fontsize=10,
    )

    fig.subplots_adjust(
        left=0.05,
        right=0.985,
        bottom=0.07,
        top=0.90,
        wspace=0.24,
        hspace=0.42,
    )
    fig.savefig(plot_dir / "significance_overview_spearman.png", bbox_inches="tight")
    plt.close(fig)

def plot_spearman_overview_heatmap(corr_df: pd.DataFrame, out_dir: Path, title_suffix: str) -> None:
    if sns is None:
        return

    plot_dir = safe_mkdir(out_dir / "significance")
    sub = corr_df[corr_df["scope"] == "pooled_by_model_pair"].copy()
    sub = sub[sub["spearman_r"].notna()].copy()
    if sub.empty:
        return

    pivot_r = sub.pivot(index="metric", columns="model_pair", values="spearman_r")
    pivot_p = sub.pivot(index="metric", columns="model_pair", values="spearman_p")

    metric_order = (
        sub.groupby("metric")["spearman_r"]
        .apply(lambda s: np.nanmax(np.abs(s.values)) if len(s) else np.nan)
        .sort_values(ascending=False)
        .index.tolist()
    )
    pivot_r = pivot_r.loc[[m for m in metric_order if m in pivot_r.index]]

    annot = pivot_r.copy().astype(object)
    for r in pivot_r.index:
        for c in pivot_r.columns:
            val = pivot_r.loc[r, c]
            p = pivot_p.loc[r, c] if (r in pivot_p.index and c in pivot_p.columns) else np.nan
            if pd.isna(val):
                annot.loc[r, c] = ""
            else:
                annot.loc[r, c] = f"{val:.2f}\n{_significance_label(p)}"

    fig_w = max(8.5, 1.5 * len(pivot_r.columns) + 2.0)
    fig_h = max(6.5, 0.65 * len(pivot_r.index) + 2.0)

    plt.figure(figsize=(fig_w, fig_h))
    ax = sns.heatmap(
        pivot_r,
        annot=annot,
        fmt="",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Spearman r"},
        annot_kws={"size": 8},
    )

    ax.set_xticklabels([pretty_model_pair(x) for x in pivot_r.columns], rotation=30, ha="right")
    ax.set_yticklabels([_pretty_metric_label(x, width=24) for x in pivot_r.index], rotation=0)
    ax.set_title(f"Spearman overview heatmap — {title_suffix}", fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(plot_dir / "spearman_overview_heatmap.png", bbox_inches="tight")
    plt.close()

def plot_connectivity_scatter(df: pd.DataFrame, out_dir: Path) -> None:
    plot_dir = safe_mkdir(out_dir / "connectivity_scatter")
    datasets = sorted(df["dataset"].dropna().unique().tolist())
    palette = get_dataset_palette(datasets)

    for m1, m2 in combinations(CONNECTIVITY_METRICS, 2):
        plot_sub = df[["dataset", m1, m2]].dropna().copy()
        if len(plot_sub) < 2:
            continue

        x = plot_sub[m1].values.astype(float)
        y = plot_sub[m2].values.astype(float)
        stats = _corr_pair(x, y)

        fig, ax = plt.subplots(figsize=(5.4, 4.3))
        _scatter_points(ax, plot_sub, m1, m2, palette)
        _draw_regression(ax, x, y)
        add_panel_stats_box(ax, pd.Series(stats))

        ax.set_xlabel(clean_metric_name(m1))
        ax.set_ylabel(clean_metric_name(m2))
        ax.set_title(f"{clean_metric_name(m1)} vs {clean_metric_name(m2)}", fontweight="bold")

        handles, labels = _figure_legend_handles(datasets, palette)
        fig.legend(
            handles,
            labels,
            title="Dataset",
            loc="upper left",
            bbox_to_anchor=(0.63, 0.98),
            ncol=2,
            frameon=False,
            fontsize=8,
            title_fontsize=9,
            columnspacing=1.0,
            handletextpad=0.35,
        )
        fig.subplots_adjust(left=0.14, right=0.98, bottom=0.14, top=0.80)
        plt.savefig(plot_dir / f"{m1}_vs_{m2}.png", bbox_inches="tight")
        plt.close(fig)


def plot_second_order_heatmap(second_df: pd.DataFrame, out_dir: Path, title_suffix: str) -> None:
    if sns is None:
        return
    plot_dir = safe_mkdir(out_dir / "second_order")
    sub = second_df[(second_df["scope"] == "pooled_all_models")].copy()
    sub = sub[sub["adj_r2"].notna()].copy()
    if sub.empty:
        return

    pivot = sub.pivot(index="metrics", columns="model_type", values="adj_r2")
    if pivot.empty:
        return

    plt.figure(figsize=(6.4, max(3.2, 0.55 * len(pivot))))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        linewidths=0.5,
        cbar_kws={"label": "Adjusted R²"},
    )
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Single- and Second-order Models vs Δ Accuracy — {title_suffix}", fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(plot_dir / "pooled_second_order_adj_r2.png", bbox_inches="tight")
    plt.close()

# ---------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------

def build_summary_tables(master_df: pd.DataFrame, corr_df: pd.DataFrame, out_dir: Path) -> None:
    table_dir = safe_mkdir(out_dir / "tables")
    master_df.to_csv(table_dir / "master_connectivity_vs_accuracy.csv", index=False)

    sub = corr_df[corr_df["scope"] == "pooled_by_model_pair"].copy()
    if not sub.empty:
        best_rows = []
        for model_pair, grp in sub.groupby("model_pair"):
            valid = grp[grp["spearman_r"].notna()].copy()
            if valid.empty:
                continue
            best_idx = valid["spearman_r"].abs().idxmax()
            best_rows.append(valid.loc[best_idx])
        if best_rows:
            best = pd.DataFrame(best_rows).sort_values("model_pair").reset_index(drop=True)
        else:
            best = pd.DataFrame(columns=sub.columns)
        best.to_csv(table_dir / "best_metric_per_model_pair.csv", index=False)

    all_sub = corr_df[corr_df["scope"] == "pooled_all_models"].copy()
    all_sub = all_sub[all_sub["spearman_r"].notna()].copy()
    if not all_sub.empty:
        all_sub = all_sub.sort_values("spearman_r", key=lambda s: s.abs(), ascending=False)
        all_sub.to_csv(table_dir / "all_models_ranked_metrics.csv", index=False)


def build_model_comparison_tables(single_df: pd.DataFrame, second_df: pd.DataFrame, out_dir: Path) -> None:
    table_dir = safe_mkdir(out_dir / "tables")
    all_df = pd.concat([single_df, second_df], ignore_index=True)

    pooled = all_df[all_df["scope"] == "pooled_all_models"].copy()
    pooled = pooled[pooled["adj_r2"].notna()].sort_values("adj_r2", ascending=False)
    pooled.to_csv(table_dir / "pooled_model_ranking.csv", index=False)

    best_rows = []
    for model_pair, grp in all_df[all_df["scope"] == "per_model_pair"].groupby("model_pair"):
        grp = grp[grp["adj_r2"].notna()].copy()
        if grp.empty:
            continue
        best_rows.append(grp.loc[grp["adj_r2"].idxmax()])

    if best_rows:
        pd.DataFrame(best_rows).sort_values("model_pair").to_csv(
            table_dir / "best_explanatory_model_per_model_pair.csv", index=False
        )

# ---------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------

def run_analysis(
    metrics_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    out_dir: Path,
    title_suffix: str,
) -> None:
    out_dir = safe_mkdir(out_dir)

    master_df = build_master_df(metrics_df, delta_df)
    if master_df.empty:
        log.warning(f"Master dataframe empty for {title_suffix}. Skipping.")
        return

    master_df.to_csv(out_dir / "master_df.csv", index=False)
    metrics_df.to_csv(out_dir / "connectivity_metrics_raw.csv", index=False)
    delta_df.to_csv(out_dir / "delta_accuracy.csv", index=False)

    pooled_corr = compute_pooled_correlations(master_df, out_dir)
    per_model_corr = compute_per_model_correlations(master_df, out_dir)
    single_df = compute_single_metric_regressions(master_df, out_dir)
    second_df = compute_second_order_models(master_df, out_dir)

    plot_scatter_by_model(master_df, per_model_corr, out_dir, title_suffix)
    plot_scatter_all_models(master_df, pooled_corr, out_dir, title_suffix)
    plot_heatmap_per_model(pooled_corr, out_dir, title_suffix)
    plot_significance_bars(pooled_corr, out_dir, title_suffix)
    plot_spearman_overview_heatmap(pooled_corr, out_dir, title_suffix)
    build_summary_tables(master_df, pooled_corr, out_dir)
    build_model_comparison_tables(single_df, second_df, out_dir)
    plot_second_order_heatmap(pd.concat([single_df, second_df], ignore_index=True), out_dir, title_suffix)

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Relate connectivity metrics to delta accuracy using the image table, with all-alpha and best-alpha-per-base-model analyses."
    )
    parser.add_argument(
        "--metrics-dir",
        default="results/connectivity_metrics",
        help="Directory containing {dataset}/directed.json files",
    )
    parser.add_argument(
        "--out-dir",
        default="results/correlation_connectivity_accuracy_paper",
        help="Root output directory",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    set_publication_style()

    root_out_dir = safe_mkdir(args.out_dir)

    log.info("Loading connectivity metrics...")
    metrics_df = load_connectivity_metrics(args.metrics_dir)
    if metrics_df.empty:
        raise RuntimeError("No connectivity metrics found.")

    log.info("Loading accuracies from image table...")
    acc_df = load_accuracy_table()
    if acc_df.empty:
        raise RuntimeError("No accuracies loaded from image table.")

    acc_df.to_csv(root_out_dir / "accuracy_table_paper.csv", index=False)

    all_alpha_delta_df = build_delta_accuracy_df_all_alphas(acc_df)
    best_alpha_delta_df = build_delta_accuracy_df_best_alpha_per_base_model(acc_df)

    if all_alpha_delta_df.empty:
        raise RuntimeError("All-alpha delta dataframe is empty.")
    if best_alpha_delta_df.empty:
        raise RuntimeError("Best-alpha delta dataframe is empty.")

    all_alpha_delta_df.to_csv(root_out_dir / "delta_accuracy_all_alphas.csv", index=False)
    best_alpha_delta_df.to_csv(root_out_dir / "delta_accuracy_best_alpha_per_base_model.csv", index=False)

    compute_connectivity_correlations(metrics_df, root_out_dir)
    plot_connectivity_heatmap(metrics_df, root_out_dir)
    plot_connectivity_scatter(metrics_df, root_out_dir)

    run_analysis(
        metrics_df=metrics_df,
        delta_df=all_alpha_delta_df,
        out_dir=root_out_dir / "all_alphas",
        title_suffix="All α configurations",
    )

    run_analysis(
        metrics_df=metrics_df,
        delta_df=best_alpha_delta_df,
        out_dir=root_out_dir / "best_alpha_per_base_model",
        title_suffix="Best α per base-model family",
    )

    log.info(f"Done. Outputs saved to: {root_out_dir}")


if __name__ == "__main__":
    main()