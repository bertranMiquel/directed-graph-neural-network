"""
Paper-ready result extraction and plotting for baseline vs directed GNN comparisons.

Example:
python extract_results_paper_ready.py \
    --folder1 logs/run \
    --folder2 logs/run_homo_dir \
    --label1 baseline \
    --label2 homo_dir \
    --output-dir results
"""
import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

DIRECTED_ALPHA_ORDER = ["0.0", "0.5", "1.0"]
MODEL_FAMILIES = ["gcn", "sage", "gat"]

DATASET_GROUPS = {
    "homophilic": [
        "amazon-computers",
        "amazon-photo",
        "citeseer_full",
        "coauthor-cs",
        "coauthor-phy",
        "cora_ml",
        "pubmed",
    ],
    "heterophilic": [
        "chameleon",
        "cornell",
        "directed-roman-empire",
        "directed_amazon_ratings",
        "squirrel",
        "texas",
        "wisconsin",
    ],
}

LOG_RE = re.compile(
    r"Dataset:\s*(?P<dataset>\S+)\s+"
    r"Model:\s*(?P<model>\S+)\s+"
    r"(?:Alpha:\s*(?P<alpha>[0-9eE+\-.]+)\s+)?"
    r"Test Acc:\s*(?P<mean>[0-9eE+\-.]+)\s*\+\-\s*(?P<std>[0-9eE+\-.]+)"
)

SERIES_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
MODEL_DISPLAY = {"gcn": "GCN", "sage": "SAGE", "gat": "GAT"}


def setup_plot_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "bold",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.18,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def build_model_key(model, alpha):
    model = model.lower()
    if model.startswith("dir-"):
        if alpha is None:
            raise ValueError(f"Directed model without alpha: {model}")
        return f"{model}_alpha_{alpha}"
    for m in MODEL_FAMILIES:
        if model.startswith(m):
            return m
    return model


def parse_log_file(path):
    results = defaultdict(dict)
    directed_occ = defaultdict(int)

    with open(path, "r") as f:
        for line in f:
            m = LOG_RE.search(line)
            if not m:
                continue

            dataset = m.group("dataset").lower()
            model = m.group("model").lower()
            alpha = m.group("alpha")
            mean = float(m.group("mean")) * 100.0
            std = float(m.group("std")) * 100.0

            if model.startswith("dir-") and alpha is None:
                k = (dataset, model)
                idx = directed_occ[k]
                if idx >= len(DIRECTED_ALPHA_ORDER):
                    raise ValueError(f"Too many alpha-less directed entries for {dataset}/{model} in {path}")
                alpha = DIRECTED_ALPHA_ORDER[idx]
                directed_occ[k] += 1

            results[dataset][build_model_key(model, alpha)] = (mean, std)

    return dict(results)


def merge_results(*dicts):
    out = defaultdict(dict)
    for d in dicts:
        for ds, vals in d.items():
            out[ds].update(vals)
    return dict(out)


def parse_results_root(root):
    root = Path(root)
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    return merge_results(*[
        parse_log_file(p)
        for ds_dir in sorted(subdirs)
        for p in sorted(ds_dir.glob("*.out"))
    ])


def initialize_template():
    return {
        k: None
        for k in (
            list(MODEL_FAMILIES)
            + [f"dir-{m}_alpha_{a}" for m in MODEL_FAMILIES for a in ["0.0", "1.0", "0.5"]]
        )
    }


def fill_missing(results):
    return {ds: (initialize_template() | vals) for ds, vals in results.items()}


def is_missing(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def best_directed(dataset_results, model_family):
    cand = [
        (*dataset_results[k], k)
        for a in DIRECTED_ALPHA_ORDER
        if (k := f"dir-{model_family}_alpha_{a}") in dataset_results
        and dataset_results[k] is not None
        and not is_missing(dataset_results[k][0])
    ]
    if not cand:
        return None
    mean, std, key = max(cand, key=lambda x: x[0])
    return {"mean": mean, "std": std, "key": key}


def get_variant_result(dataset_results, model_family, variant):
    if variant == "gnn":
        v = dataset_results.get(model_family)
        if v is None or is_missing(v[0]):
            return None
        return {"mean": v[0], "std": v[1], "key": model_family}
    if variant == "directed":
        return best_directed(dataset_results, model_family)
    raise ValueError(f"Unknown variant: {variant}")


def format_dataset_short(name):
    return (
        name.replace("amazon-", "amz-")
            .replace("coauthor-", "coa-")
            .replace("citeseer_", "citeseer-")
            .replace("cora_", "cora-")
            .replace("_", "-")
            .upper()
    )


def build_categories(results, dataset_names):
    return [(ds, m) for ds in dataset_names if ds in results for m in MODEL_FAMILIES]


def _compute_dataset_spans(categories):
    spans = []
    if not categories:
        return spans

    start = 0
    current_ds = categories[0][0]
    for i, (ds, _) in enumerate(categories[1:], start=1):
        if ds != current_ds:
            spans.append((current_ds, start, i - 1))
            current_ds = ds
            start = i
    spans.append((current_ds, start, len(categories) - 1))
    return spans


def _finite_values(series):
    vals = []
    for s in series:
        vals.extend([v for v in s["means"] if not math.isnan(v)])
    return vals


def plot_grouped_bars(categories, series, title, output_path, ylabel="Test Accuracy (%)", ylim=None):
    finite_vals = _finite_values(series)
    if not categories or not finite_vals:
        return

    n = len(categories)
    k = len(series)
    x = list(range(n))
    width = min(0.78 / max(k, 1), 0.32)

    if ylim is None:
        ymin = max(0, 5 * math.floor((min(finite_vals) - 4) / 5))
        ymax = min(100, 5 * math.ceil((max(finite_vals) + 3) / 5))
        if ymax - ymin < 20:
            ymin = max(0, ymax - 20)
        ylim = (ymin, ymax)

    fig_w = max(12.5, 0.42 * n + 4.2)
    fig, ax = plt.subplots(figsize=(fig_w, 5.9))

    offsets = [((i - (k - 1) / 2) * width) for i in range(k)]

    for band_idx, (_, start, end) in enumerate(_compute_dataset_spans(categories)):
        if band_idx % 2 == 0:
            ax.axvspan(start - 0.5, end + 0.5, alpha=0.04, color="black", zorder=0)
        if end < n - 1:
            ax.axvline(end + 0.5, color="0.75", linewidth=0.7, zorder=1)

    for i, (off, s) in enumerate(zip(offsets, series)):
        means = s["means"]
        stds = [0.0 if math.isnan(v) else v for v in s["stds"]]
        ax.bar(
            [xi + off for xi in x],
            means,
            width=width,
            yerr=stds,
            capsize=2.3,
            color=SERIES_COLORS[i % len(SERIES_COLORS)],
            edgecolor="white",
            linewidth=0.7,
            error_kw={"elinewidth": 0.8, "capthick": 0.8},
            label=s["label"],
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY[m] for _, m in categories])

    y_top = ylim[1]
    for ds, start, end in _compute_dataset_spans(categories):
        center = (start + end) / 2
        ax.text(
            center,
            y_top - 0.6,
            format_dataset_short(ds),
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.set_xlim(-0.7, n - 0.3)
    ax.set_title(title, pad=12)
    ax.grid(axis="y", zorder=0)

    legend_handles = [
        Patch(facecolor=SERIES_COLORS[i % len(SERIES_COLORS)], edgecolor="none", label=s["label"])
        for i, s in enumerate(series)
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        ncol=min(3, len(series)),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        columnspacing=1.5,
        handlelength=1.4,
    )

    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_model_vs_directed(results, dataset_names, group_name, exp_name, output_dir):
    categories = build_categories(results, dataset_names)
    gnn_means, gnn_stds, dir_means, dir_stds = [], [], [], []

    for ds, model in categories:
        r_gnn = get_variant_result(results[ds], model, "gnn")
        r_dir = get_variant_result(results[ds], model, "directed")

        gnn_means.append(float("nan") if r_gnn is None else r_gnn["mean"])
        gnn_stds.append(float("nan") if r_gnn is None else (0.0 if is_missing(r_gnn["std"]) else r_gnn["std"]))
        dir_means.append(float("nan") if r_dir is None else r_dir["mean"])
        dir_stds.append(float("nan") if r_dir is None else (0.0 if is_missing(r_dir["std"]) else r_dir["std"]))

    plot_grouped_bars(
        categories=categories,
        series=[
            {"label": "Baseline GNN", "means": gnn_means, "stds": gnn_stds},
            {"label": "Best Directed GNN", "means": dir_means, "stds": dir_stds},
        ],
        title=f"{group_name.capitalize()} datasets",
        output_path=output_dir / f"{group_name}_{exp_name}_gnn_vs_directed_paper.pdf",
    )


def plot_folder_comparison(results_a, results_b, dataset_names, group_name, label_a, label_b, variant, output_dir):
    categories = sorted(set(build_categories(results_a, dataset_names)) | set(build_categories(results_b, dataset_names)))
    a_means, a_stds, b_means, b_stds = [], [], [], []

    for ds, model in categories:
        ra = get_variant_result(results_a.get(ds, {}), model, variant) if ds in results_a else None
        rb = get_variant_result(results_b.get(ds, {}), model, variant) if ds in results_b else None

        a_means.append(float("nan") if ra is None else ra["mean"])
        a_stds.append(float("nan") if ra is None else (0.0 if is_missing(ra["std"]) else ra["std"]))
        b_means.append(float("nan") if rb is None else rb["mean"])
        b_stds.append(float("nan") if rb is None else (0.0 if is_missing(rb["std"]) else rb["std"]))

    title_variant = "GNN" if variant == "gnn" else "Directed GNN"
    plot_grouped_bars(
        categories=categories,
        series=[
            {"label": label_a, "means": a_means, "stds": a_stds},
            {"label": label_b, "means": b_means, "stds": b_stds},
        ],
        title=f"{group_name.capitalize()} datasets — {title_variant}",
        output_path=output_dir / f"{group_name}_{variant}_{label_a}_vs_{label_b}_paper.pdf",
    )


def write_results_csv(results, path):
    header = ["dataset"] + list(initialize_template().keys())
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for ds in sorted(results):
            writer.writerow([ds] + [results[ds].get(k, "") for k in header[1:]])


def sanitize_name(path_str):
    return Path(path_str).name.replace(" ", "_").replace("/", "_")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--folder1", default="logs/run", help="First root logs folder")
    p.add_argument("--folder2", nargs="?", default="logs/run_bidir", help="Second root logs folder")
    p.add_argument("--label1", default="baseline", help="Legend label for folder1")
    p.add_argument("--label2", default="bidirected", help="Legend label for folder2")
    p.add_argument("--output-dir", default="results")
    return p.parse_args()


def main():
    setup_plot_style()
    args = parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    name1 = args.label1 or sanitize_name(args.folder1)
    res1 = fill_missing(parse_results_root(args.folder1))
    write_results_csv(res1, out / f"extracted_results_{name1}.csv")

    for group_name, datasets in DATASET_GROUPS.items():
        plot_model_vs_directed(res1, datasets, group_name, name1, out)

    if args.folder2:
        name2 = args.label2 or sanitize_name(args.folder2)
        res2 = fill_missing(parse_results_root(args.folder2))
        write_results_csv(res2, out / f"extracted_results_{name2}.csv")

        for group_name, datasets in DATASET_GROUPS.items():
            plot_model_vs_directed(res2, datasets, group_name, name2, out)
            plot_folder_comparison(res1, res2, datasets, group_name, name1, name2, "gnn", out)
            plot_folder_comparison(res1, res2, datasets, group_name, name1, name2, "directed", out)


if __name__ == "__main__":
    main()
