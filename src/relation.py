#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau, linregress

try:
    import seaborn as sns
except Exception:
    sns = None
from collections import defaultdict

        
log = logging.getLogger("corr_connectivity_accuracy")

DATASET_GROUPS = {
    "homophilic": [
        "citeseer_full",
        "cora_ml",
        "ogbn-arxiv",
    ],
    "heterophilic": [
        "chameleon",
        "squirrel",
        "arxiv-year",
        "snap-patents",
        "roman-empire",
    ],
}

MODEL_FAMILIES = ["gcn", "sage", "gat"]
ALPHAS_ORDER = [0.0, 1.0, 0.5]


def dataset_display_name(name: str) -> str:
    return name.upper().replace("_", "-")


def metric_display_name(name: str) -> str:
    if name == "homophility":
        return "HOM."
    return name.upper()


def normalize_model_name(model: str) -> str:
    model = model.strip().lower()
    mapping = {
        "dir-gcn": "dir-gcn",
        "dir_gcn": "dir-gcn",
        "dirgcn": "dir-gcn",
        "gcn": "gcn",
        "dir-gat": "dir-gat",
        "dir_gat": "dir-gat",
        "dirgat": "dir-gat",
        "gat": "gat",
        # "dir-gcnii": "dir-gcnii",
        # "dir_gcnii": "dir-gcnii",
        # "dirgcnii": "dir-gcnii",
        # "gcnii": "gcnii",
        "dir-sage": "dir-sage",
        "dir_sage": "dir-sage",
        "dirsage": "dir-sage",
        "sage": "sage",
    }
    return mapping.get(model, model.replace("_", "-"))


def extract_alpha_from_text(text: str) -> float | None:
    if text is None:
        return None
    m = re.search(r"(?:alpha|a)\s*=\s*([0-9.]+)", text.lower())
    if m:
        return float(m.group(1))
    return None


def base_family_from_model(model: str) -> str:
    model = normalize_model_name(model)
    if model.startswith("dir-"):
        return model[4:]
    return model


def format_model_for_table(model: str, alpha: float | None) -> str:
    model = normalize_model_name(model)
    fam = base_family_from_model(model).upper()

    if model.startswith("dir-"):
        if alpha is None:
            return f"DIR-{fam}"
        return f"DIR-{fam}($\\alpha$={alpha:.1f})"
    return fam


def family_row_order() -> list[str]:
    rows = []
    for fam in MODEL_FAMILIES:
        rows.append(format_model_for_table(fam, None))
        for a in ALPHAS_ORDER:
            rows.append(format_model_for_table(f"dir-{fam}", a))
    return rows


def format_mean_std(mean, std, scale=100.0):
    if pd.isna(mean):
        return "-"
    if pd.isna(std):
        return f"{mean * scale:.2f}"
    return f"{mean * scale:.2f}$\\pm${std * scale:.2f}"


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

MODEL_PAIRS = {
    "gcn": "dir-gcn",
    "gat": "dir-gat",
    "gcnii": "dir-gcnii",
    "sage": "dir-sage",
}

CONNECTIVITY_METRICS = [
    "bidirectionality_gap",
    "density",
    "homophility",
]

# Explicit format:
# Dataset:citeseer_full Model:gcn Test Acc: 0.9330812931060791 +- 0.006667379664282218
ACC_RE_EXPLICIT = re.compile(
    r"Dataset:(?P<dataset>\S+)\s+Model:(?P<model>\S+)\s+Test Acc:\s+"
    r"(?P<mean>[0-9eE+\-.]+)\s*\+\-\s*(?P<std>[0-9eE+\-.]+)"
)

ACC_RE_EXPLICIT_FALLBACK = re.compile(
    r"Dataset:(?P<dataset>\S+)\s+Model:(?P<model>\S+)\s+Test Acc:\s+"
    r"(?P<mean>[0-9eE+\-.]+)"
)

# Compact format:
# dir-gcn Test Acc: 0.7925438642501831 +- 0.014586206640446808
ACC_RE_COMPACT = re.compile(
    r"(?P<model>[A-Za-z0-9_\-]+)\s+Test Acc:\s+"
    r"(?P<mean>[0-9eE+\-.]+)\s*\+\-\s*(?P<std>[0-9eE+\-.]+)"
)

ACC_RE_COMPACT_FALLBACK = re.compile(
    r"(?P<model>[A-Za-z0-9_\-]+)\s+Test Acc:\s+"
    r"(?P<mean>[0-9eE+\-.]+)"
)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def safe_mkdir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def clean_metric_name(x: str) -> str:
    return x.replace("_", " ").title()


def clean_model_name(x: str) -> str:
    mapping = {
        "gcn": "GCN",
        "dir-gcn": "DirGCN",
        "gat": "GAT",
        "dir-gat": "DirGAT",
        "sage": "SAGE",
        "dir-sage": "DirSAGE",
    }
    return mapping.get(x.lower(), x)


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


def add_dataset_annotations(ax, plot_df: pd.DataFrame, x_col: str, y_col: str, fontsize: int = 8):
    for _, row in plot_df.iterrows():
        ax.annotate(
            row["dataset"],
            (row[x_col], row[y_col]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=fontsize,
            alpha=0.85,
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


# ---------------------------------------------------------------------
# Connectivity metrics
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


# ---------------------------------------------------------------------
# Accuracy parsing
# ---------------------------------------------------------------------

def extract_alpha(model_name: str):
    """
    Extract alpha from strings like:
    dir-gcn(alpha=0.5) or dir-gcn(a=0.5)
    """
    import re

    m = re.search(r"(?:alpha|a)\s*=\s*([0-9.]+)", model_name)
    if m:
        return float(m.group(1))

    return None

def parse_log_file(path: str | Path, dataset_fallback: str | None = None) -> list[dict]:
    rows: list[dict] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if "OOM" in line.upper():
                continue

            m = ACC_RE_EXPLICIT.search(line)
            if m:
                model_raw = m.group("model")
                rows.append({
                    "dataset": m.group("dataset"),
                    "model_raw": model_raw,
                    "model": normalize_model_name(model_raw),
                    "alpha": extract_alpha_from_text(model_raw),
                    "test_acc_mean": float(m.group("mean")),
                    "test_acc_std": float(m.group("std")),
                    "source_file": str(path),
                    "raw_line": line,
                })
                continue

            m = ACC_RE_EXPLICIT_FALLBACK.search(line)
            if m:
                model_raw = m.group("model")
                rows.append({
                    "dataset": m.group("dataset"),
                    "model_raw": model_raw,
                    "model": normalize_model_name(model_raw),
                    "alpha": extract_alpha_from_text(model_raw),
                    "test_acc_mean": float(m.group("mean")),
                    "test_acc_std": np.nan,
                    "source_file": str(path),
                    "raw_line": line,
                })
                continue

            m = ACC_RE_COMPACT.search(line)
            if m and dataset_fallback is not None:
                model_raw = m.group("model")
                rows.append({
                    "dataset": dataset_fallback,
                    "model_raw": model_raw,
                    "model": normalize_model_name(model_raw),
                    "alpha": extract_alpha_from_text(model_raw),
                    "test_acc_mean": float(m.group("mean")),
                    "test_acc_std": float(m.group("std")),
                    "source_file": str(path),
                    "raw_line": line,
                })
                continue

            m = ACC_RE_COMPACT_FALLBACK.search(line)
            if m and dataset_fallback is not None:
                model_raw = m.group("model")
                rows.append({
                    "dataset": dataset_fallback,
                    "model_raw": model_raw,
                    "model": normalize_model_name(model_raw),
                    "alpha": extract_alpha_from_text(model_raw),
                    "test_acc_mean": float(m.group("mean")),
                    "test_acc_std": np.nan,
                    "source_file": str(path),
                    "raw_line": line,
                })
                continue

    if len(rows) == 0:
        log.warning(f"No accuracy lines parsed from log file: {path}")
    return rows


def load_accuracy_logs(logs_dir: str | Path) -> pd.DataFrame:
    rows: list[dict] = []
    logs_dir = Path(logs_dir)

    if not logs_dir.exists():
        raise FileNotFoundError(f"Logs directory not found: {logs_dir}")

    for dataset_dir in sorted(logs_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name
        for path in glob.glob(str(dataset_dir / "*.out")):
            rows.extend(parse_log_file(path, dataset_fallback=dataset_name))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Keep best test accuracy per (dataset, model)
    df = df.sort_values(
        ["dataset", "model", "test_acc_mean", "source_file"],
        ascending=[True, True, False, True],
    )
    df = df.drop_duplicates(subset=["dataset", "model"], keep="first").reset_index(drop=True)

    log.info(f"Parsed accuracies for datasets: {sorted(df['dataset'].unique().tolist())}")
    return df


# ---------------------------------------------------------------------
# Delta accuracy
# ---------------------------------------------------------------------

def build_delta_accuracy_df(acc_df: pd.DataFrame) -> pd.DataFrame:
    if acc_df.empty:
        return pd.DataFrame()

    pivot_mean = acc_df.pivot(index="dataset", columns="model", values="test_acc_mean")
    pivot_std = acc_df.pivot(index="dataset", columns="model", values="test_acc_std")

    rows: list[dict] = []
    for dataset in pivot_mean.index:
        for base_model, dir_model in MODEL_PAIRS.items():
            if base_model not in pivot_mean.columns or dir_model not in pivot_mean.columns:
                continue

            base_acc = pivot_mean.loc[dataset, base_model]
            dir_acc = pivot_mean.loc[dataset, dir_model]

            if pd.isna(base_acc) or pd.isna(dir_acc):
                continue

            base_std = pivot_std.loc[dataset, base_model] if base_model in pivot_std.columns else np.nan
            dir_std = pivot_std.loc[dataset, dir_model] if dir_model in pivot_std.columns else np.nan

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
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["model_pair", "dataset"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------

def build_master_df(metrics_df: pd.DataFrame, delta_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty or delta_df.empty:
        return pd.DataFrame()
    return pd.merge(delta_df, metrics_df, on="dataset", how="inner")


# ---------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------

def compute_connectivity_correlations(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []

    for i, m1 in enumerate(CONNECTIVITY_METRICS):
        for m2 in CONNECTIVITY_METRICS[i + 1:]:
            sub = df[[m1, m2]].dropna()
            if len(sub) < 3:
                continue

            x = sub[m1].values.astype(float)
            y = sub[m2].values.astype(float)

            try:
                pr, pp = pearsonr(x, y)
            except Exception:
                pr, pp = np.nan, np.nan
            try:
                sr, sp = spearmanr(x, y)
            except Exception:
                sr, sp = np.nan, np.nan
            try:
                kr, kp = kendalltau(x, y)
            except Exception:
                kr, kp = np.nan, np.nan

            rows.append({
                "metric_x": m1,
                "metric_y": m2,
                "pearson_r": pr,
                "pearson_p": pp,
                "spearman_r": sr,
                "spearman_p": sp,
                "kendall_tau": kr,
                "kendall_p": kp,
                "n": len(sub),
            })

    corr_df = pd.DataFrame(rows)
    corr_df.to_csv(out_dir / "connectivity_metric_correlations.csv", index=False)
    return corr_df


def compute_pooled_correlations(master_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []

    for model_pair, grp in master_df.groupby("model_pair"):
        y = grp["delta_acc"].values.astype(float)
        for metric in CONNECTIVITY_METRICS:
            x = grp[metric].values.astype(float)
            row = {
                "scope": "pooled_by_model_pair",
                "model_pair": model_pair,
                "metric": metric,
            }
            row.update(_corr_pair(x, y))
            rows.append(row)

    master_df["delta_acc"] = master_df["delta_acc"].astype(float)
    master_df["delta_acc"] *= 100.0
    y_all = master_df["delta_acc"].values.astype(float)
    for metric in CONNECTIVITY_METRICS:
        x_all = master_df[metric].values.astype(float)
        row = {
            "scope": "pooled_all_models",
            "model_pair": "all",
            "metric": metric,
        }
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
            row = {
                "model_pair": model_pair,
                "metric": metric,
            }
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


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------

def _draw_regression(ax, x: np.ndarray, y: np.ndarray):
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return
    xv = x[valid]
    yv = y[valid]
    try:
        lr = linregress(xv, yv)
        xs = np.linspace(xv.min(), xv.max(), 200)
        ys = lr.intercept + lr.slope * xs
        ax.plot(xs, ys, linestyle="--", linewidth=1.5, color="black", alpha=0.8)
    except Exception:
        pass


def plot_scatter_by_model(master_df: pd.DataFrame, corr_df: pd.DataFrame, out_dir: Path, annotate: bool = True):
    plot_dir = safe_mkdir(out_dir / "scatter_by_model")

    for model_pair, grp in master_df.groupby("model_pair"):
        sub_corr = corr_df[corr_df["model_pair"] == model_pair].copy()
        if sub_corr.empty:
            continue

        n = len(CONNECTIVITY_METRICS)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
        axes = np.atleast_1d(axes).ravel()

        for i, metric in enumerate(CONNECTIVITY_METRICS):
            ax = axes[i]
            plot_sub = grp.dropna(subset=[metric, "delta_acc", "dataset"]).copy()
            if plot_sub.empty:
                ax.set_visible(False)
                continue

            x = plot_sub[metric].values.astype(float)
            y = plot_sub["delta_acc"].values.astype(float)

            if sns is not None:
                sns.scatterplot(
                    data=plot_sub,
                    x=metric,
                    y="delta_acc",
                    hue="dataset",
                    # style="dataset",
                    s=90,
                    alpha=0.9,
                    ax=ax,
                    legend=(i == 0),
                )
            else:
                for ds, ds_grp in plot_sub.groupby("dataset"):
                    ax.scatter(
                        ds_grp[metric],
                        ds_grp["delta_acc"],
                        label=ds if i == 0 else None,
                        alpha=0.9,
                        s=70,
                    )

            if annotate:
                add_dataset_annotations(ax, plot_sub, metric, "delta_acc")

            _draw_regression(ax, x, y)
            ax.axhline(0, color="red", linestyle=":", linewidth=0.8, alpha=0.6)

            row = sub_corr[sub_corr["metric"] == metric]
            if not row.empty:
                row = row.iloc[0]
                title = (
                    f"{clean_metric_name(metric)}\n"
                    f"n={int(row['n'])} | Spearman={row['spearman_r']:.3f}, p={row['spearman_p']:.2e}\n"
                    f"Pearson={row['pearson_r']:.3f}, R²={row['r2']:.3f}"
                )
            else:
                title = clean_metric_name(metric)

            ax.set_title(title, fontsize=10)
            ax.set_xlabel(clean_metric_name(metric))
            base_name, dir_name = model_pair.split("_vs_")
            ax.set_ylabel(f"Δ Acc ({clean_model_name(dir_name)} - {clean_model_name(base_name)})")

            # remove axis legends; we will create one figure-level legend
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()

        # build single external legend from first non-empty subplot
        legend_handles, legend_labels = None, None
        for ax in axes:
            handles, labels = ax.get_legend_handles_labels()
            if handles and labels:
                legend_handles, legend_labels = handles, labels
                break

        # If seaborn removed the legend handles from axes, rebuild from datasets
        if not legend_handles or not legend_labels:
            datasets = grp["dataset"].dropna().unique().tolist()
            if sns is not None and len(datasets) > 0:
                palette = sns.color_palette(n_colors=len(datasets))
                from matplotlib.lines import Line2D
                legend_handles = [
                    Line2D([0], [0], marker='o', linestyle='', label=ds, markersize=7, color=palette[j])
                    for j, ds in enumerate(datasets)
                ]
                legend_labels = datasets

        # if legend_handles and legend_labels:
        #     fig.legend(
        #         legend_handles,
        #         legend_labels,
        #         title="Dataset",
        #         loc="upper center",
        #         bbox_to_anchor=(0.5, 1.15),
        #         frameon=True,
        #         fontsize=8,
        #         title_fontsize=9,
        #     )

        # ---- Legend (right of title) ----
        if legend_handles and legend_labels:
            fig.legend(
                legend_handles,
                legend_labels,
                title="Dataset",
                loc="upper left",
                bbox_to_anchor=(0.62, 1.02),   # 👈 tweak this if needed
                ncol=2,
                frameon=True,
                fontsize=8,
                title_fontsize=9,
                columnspacing=1.2,
                handletextpad=0.4,
            )

        # ---- Title ----
        fig.suptitle(
            f"Connectivity vs Δ Accuracy — {model_pair}",
            fontweight="bold",
            y=1.02,
        )

        # ---- Layout (important) ----
        fig.subplots_adjust(top=0.78, wspace=0.25)

        fig.savefig(plot_dir / f"scatter_{model_pair}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # fig.suptitle(f"Connectivity vs Δ Accuracy — {model_pair}", fontweight="bold", y=1.03)
        # fig.tight_layout()  #rect=[0, 0, 0.86, 1])
        # fig.savefig(plot_dir / f"scatter_{model_pair}.png", dpi=300, bbox_inches="tight")
        # plt.close(fig)

        
def plot_scatter_all_models(master_df: pd.DataFrame, corr_df: pd.DataFrame, out_dir: Path, annotate: bool = True):
    plot_dir = safe_mkdir(out_dir / "scatter_all_models")

    sub_corr = corr_df[
        (corr_df["scope"] == "pooled_all_models") &
        (corr_df["model_pair"] == "all")
    ].copy()

    n = len(CONNECTIVITY_METRICS)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    axes = np.atleast_1d(axes).ravel()

    for i, metric in enumerate(CONNECTIVITY_METRICS):
        ax = axes[i]
        plot_sub = master_df.dropna(subset=[metric, "delta_acc", "dataset"]).copy()
        if plot_sub.empty:
            ax.set_visible(False)
            continue

        x = plot_sub[metric].values.astype(float)
        y = plot_sub["delta_acc"].values.astype(float)

        if sns is not None:
            sns.scatterplot(
                data=plot_sub,
                x=metric,
                y="delta_acc",
                hue="dataset",
                style="model_pair",
                s=95,
                alpha=0.9,
                ax=ax,
                legend=(i == 0),
            )
        else:
            for ds, ds_grp in plot_sub.groupby("dataset"):
                ax.scatter(
                    ds_grp[metric],
                    ds_grp["delta_acc"],
                    label=ds if i == 0 else None,
                    alpha=0.9,
                    s=70,
                )

        if annotate:
            add_dataset_annotations(ax, plot_sub, metric, "delta_acc")

        _draw_regression(ax, x, y)
        ax.axhline(0, color="red", linestyle=":", linewidth=0.8, alpha=0.6)

        row = sub_corr[sub_corr["metric"] == metric]
        if not row.empty:
            row = row.iloc[0]
            title = (
                f"{clean_metric_name(metric)}\n"
                f"n={int(row['n'])} | Spearman={row['spearman_r']:.3f}, p={row['spearman_p']:.2e}\n"
                f"Pearson={row['pearson_r']:.3f}, R²={row['r2']:.3f}"
            )
        else:
            title = clean_metric_name(metric)

        ax.set_title(title, fontsize=10)
        ax.set_xlabel(clean_metric_name(metric))
        ax.set_ylabel("Δ Accuracy (Directed - Base) [%]")

        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    legend_handles, legend_labels = None, None
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            legend_handles, legend_labels = handles, labels
            break

    if not legend_handles or not legend_labels:
        datasets = master_df["dataset"].dropna().unique().tolist()
        if sns is not None and len(datasets) > 0:
            palette = sns.color_palette(n_colors=len(datasets))
            from matplotlib.lines import Line2D
            legend_handles = [
                Line2D([0], [0], marker='o', linestyle='', label=ds, markersize=7, color=palette[j])
                for j, ds in enumerate(datasets)
            ]
            legend_labels = datasets

    # if legend_handles and legend_labels:
    #     fig.legend(
    #         legend_handles,
    #         legend_labels,
    #         title="Dataset",
    #         loc="upper center",
    #         bbox_to_anchor=(0.5, 1.15),
    #         frameon=True,
    #         fontsize=8,
    #         title_fontsize=9,
    #     )
    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            title="Dataset",
            loc="upper left",
            bbox_to_anchor=(0.62, 1.02),
            ncol=2,
            frameon=True,
            fontsize=8,
            title_fontsize=9,
            columnspacing=1.2,
            handletextpad=0.4,
        )

    fig.suptitle(
        "Connectivity vs Δ Accuracy — All Model Pairs",
        fontweight="bold",
        y=1.02,
    )

    fig.subplots_adjust(top=0.78, wspace=0.25)

    fig.savefig(plot_dir / "scatter_all_model_pairs.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    # fig.suptitle("Connectivity vs Δ Accuracy — All Model Pairs", fontweight="bold", y=1.03)
    # fig.tight_layout()  #rect=[0, 0, 0.86, 1])
    # fig.savefig(plot_dir / "scatter_all_model_pairs.png", dpi=300, bbox_inches="tight")
    # plt.close(fig)

def plot_heatmap_per_model(corr_df: pd.DataFrame, out_dir: Path):
    if sns is None:
        log.info("Seaborn not available; skipping heatmaps.")
        return

    plot_dir = safe_mkdir(out_dir / "heatmaps")

    sub = corr_df[corr_df["scope"] == "pooled_by_model_pair"].copy()
    sub = sub[sub["spearman_r"].notna()].copy()
    if sub.empty:
        return

    pivot = sub.pivot(index="metric", columns="model_pair", values="spearman_r")
    if pivot.empty:
        return

    plt.figure(figsize=(max(8, 1.8 * pivot.shape[1]), 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "Spearman r"},
    )
    plt.title("Connectivity Metric vs Δ Accuracy — Spearman Correlation", fontweight="bold", pad=12)
    plt.ylabel("Connectivity Metric")
    plt.xlabel("Model Pair")
    plt.tight_layout()
    plt.savefig(plot_dir / "heatmap_spearman_per_model.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_significance_bars(corr_df: pd.DataFrame, out_dir: Path):
    plot_dir = safe_mkdir(out_dir / "significance")

    sub = corr_df[corr_df["scope"] == "pooled_by_model_pair"].copy()
    sub = sub[sub["spearman_r"].notna()].copy()
    if sub.empty:
        return

    for metric in CONNECTIVITY_METRICS:
        mdf = sub[sub["metric"] == metric].copy()
        if mdf.empty:
            continue

        mdf = mdf.sort_values("spearman_r", key=lambda s: s.abs(), ascending=False)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(mdf["model_pair"], mdf["spearman_r"])
        ax.axhline(0, color="black", linewidth=0.8)

        for i, (_, row) in enumerate(mdf.iterrows()):
            txt = significance_stars(row["spearman_p"])
            ax.text(i, row["spearman_r"], f" {txt}", va="bottom" if row["spearman_r"] >= 0 else "top")

        ax.set_title(f"{clean_metric_name(metric)} vs Δ Accuracy", fontweight="bold")
        ax.set_ylabel("Spearman r")
        ax.set_xlabel("Model Pair")
        plt.tight_layout()
        plt.savefig(plot_dir / f"significance_{metric}.png", dpi=300, bbox_inches="tight")
        plt.close()


def plot_connectivity_heatmap(df: pd.DataFrame, out_dir: Path):
    if sns is None:
        log.info("Seaborn not available; skipping connectivity heatmap.")
        return

    sub = df[CONNECTIVITY_METRICS].dropna()
    if len(sub) < 2:
        return

    corr = sub.corr(method="spearman")

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "Spearman r"},
    )
    plt.title("Connectivity Metrics Correlation", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "connectivity_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_connectivity_scatter(df: pd.DataFrame, out_dir: Path, annotate: bool = True):
    if sns is None:
        return

    plot_dir = safe_mkdir(out_dir / "connectivity_scatter")

    for i, m1 in enumerate(CONNECTIVITY_METRICS):
        for m2 in CONNECTIVITY_METRICS[i + 1:]:
            plot_sub = df[["dataset", m1, m2]].dropna().copy()
            if len(plot_sub) < 2:
                continue

            x = plot_sub[m1].values.astype(float)
            y = plot_sub[m2].values.astype(float)

            try:
                sr, sp = spearmanr(x, y)
            except Exception:
                sr, sp = np.nan, np.nan

            fig, ax = plt.subplots(figsize=(6, 5))
            sns.scatterplot(
                data=plot_sub,
                x=m1,
                y=m2,
                hue="dataset",
                style="dataset",
                s=90,
                alpha=0.9,
                ax=ax,
            )

            _draw_regression(ax, x, y)

            if annotate:
                add_dataset_annotations(ax, plot_sub, m1, m2)

            ax.set_xlabel(clean_metric_name(m1))
            ax.set_ylabel(clean_metric_name(m2))
            ax.set_title(f"{clean_metric_name(m1)} vs {clean_metric_name(m2)}\nSpearman={sr:.3f}, p={sp:.2e}")

            leg = ax.get_legend()
            if leg is not None:
                leg.set_title("Dataset")
                for text in leg.get_texts():
                    text.set_fontsize(8)

            plt.tight_layout()
            plt.savefig(plot_dir / f"{m1}_vs_{m2}.png", dpi=300, bbox_inches="tight")
            plt.close()

# ---------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------

def build_summary_tables(master_df: pd.DataFrame, corr_df: pd.DataFrame, out_dir: Path):
    safe_mkdir(out_dir / "tables")

    master_df.to_csv(out_dir / "tables" / "master_connectivity_vs_accuracy.csv", index=False)

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

        best.to_csv(out_dir / "tables" / "best_metric_per_model_pair.csv", index=False)

    all_sub = corr_df[corr_df["scope"] == "pooled_all_models"].copy()
    all_sub = all_sub[all_sub["spearman_r"].notna()].copy()
    if not all_sub.empty:
        all_sub = all_sub.sort_values("spearman_r", key=lambda s: s.abs(), ascending=False)
        all_sub.to_csv(out_dir / "tables" / "all_models_ranked_metrics.csv", index=False)

def prepare_accuracy_table_df(acc_df: pd.DataFrame) -> pd.DataFrame:
    if acc_df.empty:
        return acc_df.copy()

    df = acc_df.copy()

    # Keep best entry for each dataset/model/alpha combination
    df = df.sort_values(
        ["dataset", "model", "alpha", "test_acc_mean", "source_file"],
        ascending=[True, True, True, False, True],
    )
    df = df.drop_duplicates(subset=["dataset", "model", "alpha"], keep="first").reset_index(drop=True)

    df["family"] = df["model"].apply(base_family_from_model)
    df["is_directed"] = df["model"].str.startswith("dir-")
    df["model_display"] = df.apply(
        lambda r: format_model_for_table(r["model"], r["alpha"]),
        axis=1,
    )
    df["value_str"] = df.apply(
        lambda r: format_mean_std(r["test_acc_mean"], r["test_acc_std"]),
        axis=1,
    )

    return df


def best_overall_per_dataset(acc_table_df: pd.DataFrame) -> dict[tuple[str, str], bool]:
    """
    Mark best row per dataset column across all rows.
    Returns keys (model_display, dataset) -> bool
    """
    best_map = {}
    if acc_table_df.empty:
        return best_map

    tmp = acc_table_df.copy()
    for dataset, grp in tmp.groupby("dataset"):
        grp = grp[grp["test_acc_mean"].notna()].copy()
        if grp.empty:
            continue
        best_idx = grp["test_acc_mean"].idxmax()
        row = grp.loc[best_idx]
        best_map[(row["model_display"], dataset)] = True

    return best_map


def best_directed_alpha_per_family_dataset(acc_table_df: pd.DataFrame) -> dict[tuple[str, str], bool]:
    """
    Within each directed family block and dataset, mark best alpha.
    Returns keys (model_display, dataset) -> bool
    """
    best_map = {}
    if acc_table_df.empty:
        return best_map

    tmp = acc_table_df[acc_table_df["is_directed"]].copy()
    if tmp.empty:
        return best_map

    for (family, dataset), grp in tmp.groupby(["family", "dataset"]):
        grp = grp[grp["test_acc_mean"].notna()].copy()
        if grp.empty:
            continue
        best_idx = grp["test_acc_mean"].idxmax()
        row = grp.loc[best_idx]
        best_map[(row["model_display"], dataset)] = True

    return best_map


def build_results_matrix(acc_df: pd.DataFrame, metrics_df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    acc_table_df = prepare_accuracy_table_df(acc_df)

    dataset_order = DATASET_GROUPS["homophilic"] + DATASET_GROUPS["heterophilic"]
    row_order = family_row_order()

    # Accuracy matrix
    value_matrix = acc_table_df.pivot(index="model_display", columns="dataset", values="value_str")
    value_matrix = value_matrix.reindex(index=row_order, columns=dataset_order)

    # Homophily row
    hom_row = pd.Series(index=dataset_order, dtype=object)
    if not metrics_df.empty and "homophility" in metrics_df.columns:
        metric_map = metrics_df.set_index("dataset")["homophility"].to_dict()
        for ds in dataset_order:
            val = metric_map.get(ds, np.nan)
            hom_row[ds] = "-" if pd.isna(val) else f"{val:.3f}"

    full_matrix = pd.concat(
        [pd.DataFrame([hom_row], index=["HOM."]), value_matrix],
        axis=0,
    )

    best_global = best_overall_per_dataset(acc_table_df)
    best_alpha = best_directed_alpha_per_family_dataset(acc_table_df)

    return full_matrix, best_global, best_alpha

# def format_model_for_table(model: str, alpha: float | None):
#     base = model.replace("dir-", "").upper()

#     if "dir" in model:
#         if alpha is None:
#             return f"DIR-{base}"
#         return f"DIR-{base}(α={alpha})"
#     else:
#         return base
        
def build_results_table(acc_df: pd.DataFrame, out_dir: Path):
    safe_mkdir(out_dir / "tables")

    # Keep best per dataset/model/alpha
    acc_df = acc_df.sort_values("test_acc_mean", ascending=False)
    acc_df["test_acc_mean"] = acc_df["test_acc_mean"].astype(float)
    acc_df["test_acc_mean"] *= 100.0
    acc_df["test_acc_std"] *= 100.0
    acc_df = acc_df.drop_duplicates(
        subset=["dataset", "model", "alpha"], keep="first"
    )

    # Format strings
    acc_df["value"] = acc_df.apply(
        lambda r: f"{r['test_acc_mean']:.2f}±{r['test_acc_std']:.2f}"
        if pd.notna(r["test_acc_std"])
        else f"{r['test_acc_mean']:.2f}",
        axis=1,
    )

    acc_df["model_display"] = acc_df.apply(
        lambda r: format_model_for_table(r["model"], r["alpha"]),
        axis=1,
    )

    # Pivot
    table = acc_df.pivot(
        index="model_display",
        columns="dataset",
        values="value"
    )

    # Sort rows manually (important for paper)
    order = [
        "GCN",
        "DIR-GCN(α=0.0)",
        "DIR-GCN(α=1.0)",
        "DIR-GCN(α=0.5)",
        "SAGE",
        "DIR-SAGE(α=0.0)",
        "DIR-SAGE(α=1.0)",
        "DIR-SAGE(α=0.5)",
        "GAT",
        "DIR-GAT(α=0.0)",
        "DIR-GAT(α=1.0)",
        "DIR-GAT(α=0.5)",
    ]

    table = table.reindex(order)

    table.to_csv(out_dir / "tables" / "results_table.csv")

    return table

def latex_escape_dataset_name(name: str) -> str:
    return dataset_display_name(name)


def style_cell_latex(text: str, bold: bool = False, underline: bool = False) -> str:
    if text is None or text == "" or (isinstance(text, float) and pd.isna(text)):
        return "-"
    out = str(text)
    if underline:
        out = f"\\underline{{{out}}}"
    if bold:
        out = f"\\textbf{{{out}}}"
    return out


def export_results_table_latex(
    matrix: pd.DataFrame,
    out_path: Path,
    best_global: dict,
    best_alpha: dict,
):
    dataset_order = DATASET_GROUPS["homophilic"] + DATASET_GROUPS["heterophilic"]
    hom_datasets = DATASET_GROUPS["homophilic"]
    het_datasets = DATASET_GROUPS["heterophilic"]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{l" + "c" * len(dataset_order) + "}\n")
        f.write("\\toprule\n")

        # Group headers
        f.write(
            " & "
            + f"\\multicolumn{{{len(hom_datasets)}}}{{c}}{{Homophilic}}"
            + " & "
            + f"\\multicolumn{{{len(het_datasets)}}}{{c}}{{Heterophilic}}"
            + " \\\\\n"
        )

        # cmidrules
        left_start = 2
        left_end = 1 + len(hom_datasets)
        right_start = left_end + 1
        right_end = left_end + len(het_datasets)
        f.write(f"\\cmidrule(lr){{{left_start}-{left_end}}}\\cmidrule(lr){{{right_start}-{right_end}}}\n")

        # Dataset names
        header = " & ".join(latex_escape_dataset_name(ds) for ds in dataset_order)
        f.write(f" & {header} \\\\\n")
        f.write("\\midrule\n")

        # Homophily row
        hom_vals = [matrix.loc["HOM.", ds] if ds in matrix.columns else "-" for ds in dataset_order]
        f.write("HOM. & " + " & ".join(hom_vals) + " \\\\\n")
        f.write("\\midrule\n")

        # Accuracy blocks
        for fam in MODEL_FAMILIES:
            base_row = format_model_for_table(fam, None)
            dir_rows = [format_model_for_table(f"dir-{fam}", a) for a in ALPHAS_ORDER]

            block_rows = [base_row] + dir_rows

            for ridx, row_name in enumerate(block_rows):
                vals = []
                for ds in dataset_order:
                    val = matrix.loc[row_name, ds] if (row_name in matrix.index and ds in matrix.columns) else "-"

                    is_best_global = (row_name, ds) in best_global
                    is_best_alpha = (row_name, ds) in best_alpha

                    vals.append(
                        style_cell_latex(
                            val,
                            bold=is_best_global or is_best_alpha,
                            underline=is_best_alpha,
                        )
                    )

                label = row_name.replace("$\\alpha$", r"\alpha")
                f.write(f"{label} & " + " & ".join(vals) + " \\\\\n")

            if fam != MODEL_FAMILIES[-1]:
                f.write("\\midrule\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

def export_results_table_csv(matrix: pd.DataFrame, out_path: Path):
    matrix.to_csv(out_path, index=True)

def build_publication_results_table(acc_df: pd.DataFrame, metrics_df: pd.DataFrame, out_dir: Path):
    table_dir = safe_mkdir(out_dir / "tables")

    matrix, best_global, best_alpha = build_results_matrix(acc_df, metrics_df)

    export_results_table_csv(matrix, table_dir / "results_table_full.csv")
    export_results_table_latex(
        matrix,
        table_dir / "results_table_full.tex",
        best_global=best_global,
        best_alpha=best_alpha,
    )

    return matrix
    
def export_latex_table(table: pd.DataFrame, out_path: Path):
    with open(out_path, "w") as f:
        f.write("\\begin{tabular}{l" + "c" * len(table.columns) + "}\n")
        f.write("\\toprule\n")

        # Header
        cols = " & ".join([c.upper().replace("_", "-") for c in table.columns])
        f.write(f" & {cols} \\\\\n")
        f.write("\\midrule\n")

        for idx, row in table.iterrows():
            values = " & ".join(row.fillna("-"))
            f.write(f"{idx} & {values} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Relate connectivity metrics to delta accuracy between base and directed GNNs."
    )
    parser.add_argument(
        "--metrics-dir",
        default="results/connectivity_metrics",
        help="Directory containing {dataset}/directed.json files",
    )
    parser.add_argument(
        "--logs-dir",
        default="logs/run",
        help="Directory containing {dataset}/*.out log files",
    )
    parser.add_argument(
        "--out-dir",
        default="results/correlation_connectivity_accuracy",
        help="Output directory",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Enable dataset text labels on scatter plots",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    out_dir = safe_mkdir(args.out_dir)
    annotate = args.annotate

    log.info("Loading connectivity metrics...")
    metrics_df = load_connectivity_metrics(args.metrics_dir)
    log.info(f"{len(metrics_df)} metric rows")

    log.info("Parsing accuracy logs...")
    acc_df = load_accuracy_logs(args.logs_dir)
    log.info(f"{len(acc_df)} accuracy rows")

    if acc_df.empty:
        log.error("No accuracy rows parsed from logs.")
        return

    delta_df = build_delta_accuracy_df(acc_df)
    log.info(f"{len(delta_df)} delta-accuracy rows")

    if delta_df.empty:
        log.error("No delta accuracy rows could be built.")
        return

    master_df = build_master_df(metrics_df, delta_df)
    log.info(f"Master dataframe shape: {master_df.shape}")

    if master_df.empty:
        log.error("Master dataframe is empty after merging metrics and accuracies.")
        return

    master_df.to_csv(out_dir / "master_df.csv", index=False)
    metrics_df.to_csv(out_dir / "connectivity_metrics_raw.csv", index=False)

    pooled_corr = compute_pooled_correlations(master_df, out_dir)
    per_model_corr = compute_per_model_correlations(master_df, out_dir)

    plot_scatter_by_model(master_df, per_model_corr, out_dir, annotate=annotate)
    plot_scatter_all_models(master_df, pooled_corr, out_dir, annotate=annotate)
    plot_heatmap_per_model(pooled_corr, out_dir)
    plot_significance_bars(pooled_corr, out_dir)
    build_summary_tables(master_df, pooled_corr, out_dir)

    compute_connectivity_correlations(metrics_df, out_dir)
    plot_connectivity_heatmap(metrics_df, out_dir)
    plot_connectivity_scatter(metrics_df, out_dir, annotate=annotate)

    results_table = build_results_table(acc_df, out_dir)
    export_latex_table(results_table, out_dir / "tables" / "results_table.tex")
    build_publication_results_table(acc_df, metrics_df, out_dir)
    log.info(f"Done. Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()