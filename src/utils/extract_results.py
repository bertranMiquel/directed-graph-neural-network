"""
Example: python src/utils/extract_results.py --folder1 logs/run --folder2 logs/run_homo_dir --label1 baseline --label2 homo_dir
"""
import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

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


def setup_plot_style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.25,
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
            mean = float(m.group("mean")) * 100
            std = float(m.group("std")) * 100

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
    return merge_results(*[
        parse_log_file(p)
        for ds_dir in sorted([p for p in root.iterdir() if p.is_dir()])
        for p in sorted(ds_dir.glob("*.out"))
    ])


def initialize_template():
    return {
        k: None
        for k in (
            [m for m in MODEL_FAMILIES] +
            [f"dir-{m}_alpha_{a}" for m in MODEL_FAMILIES for a in ["0.0", "1.0", "0.5"]]
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


def format_dataset_label(name):
    return name.replace("_", "-").upper()


def build_categories(results, dataset_names):
    return [(ds, m.upper()) for ds in dataset_names if ds in results for m in MODEL_FAMILIES]


def plot_grouped_bars(categories, series, title, output_path, ylabel="Test Accuracy (%)", ylim=(0, 100)):
    if not categories or not any(any(not math.isnan(v) for v in s["means"]) for s in series):
        return

    n = len(categories)
    k = len(series)
    x = list(range(n))
    width = min(0.82 / max(k, 1), 0.36)

    fig_w = max(12, 0.5 * n + 2)
    fig, ax = plt.subplots(figsize=(fig_w, 5.8))

    offsets = [((i - (k - 1) / 2) * width) for i in range(k)]
    for off, s in zip(offsets, series):
        ax.bar(
            [xi + off for xi in x],
            s["means"],
            width=width,
            yerr=s["stds"],
            capsize=2.5,
            label=s["label"],
            edgecolor="black",
            linewidth=0.4,
        )

    ax.set_title(title, pad=10)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{format_dataset_label(ds)}\n{model}" for ds, model in categories], rotation=50, ha="right")
    ax.grid(axis="y")
    ax.legend(frameon=False, ncol=min(len(series), 2), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_model_vs_directed(results, dataset_names, group_name, exp_name, output_dir):
    categories = build_categories(results, dataset_names)
    gnn_means, gnn_stds, dir_means, dir_stds = [], [], [], []

    for ds, model_up in categories:
        m = model_up.lower()
        r_gnn = get_variant_result(results[ds], m, "gnn")
        r_dir = get_variant_result(results[ds], m, "directed")

        gnn_means.append(float("nan") if r_gnn is None else r_gnn["mean"])
        gnn_stds.append(float("nan") if r_gnn is None else (0.0 if is_missing(r_gnn["std"]) else r_gnn["std"]))
        dir_means.append(float("nan") if r_dir is None else r_dir["mean"])
        dir_stds.append(float("nan") if r_dir is None else (0.0 if is_missing(r_dir["std"]) else r_dir["std"]))

    plot_grouped_bars(
        categories,
        [
            {"label": "GNN", "means": gnn_means, "stds": gnn_stds},
            {"label": "Directed GNN", "means": dir_means, "stds": dir_stds},
        ],
        title=f"{group_name.capitalize()} datasets — {exp_name}",
        output_path=output_dir / f"{group_name}_{exp_name}_gnn_vs_directed.pdf",
    )


def plot_folder_comparison(results_a, results_b, dataset_names, group_name, label_a, label_b, variant, output_dir):
    categories = sorted(set(build_categories(results_a, dataset_names)) | set(build_categories(results_b, dataset_names)))
    a_means, a_stds, b_means, b_stds = [], [], [], []

    for ds, model_up in categories:
        m = model_up.lower()
        ra = get_variant_result(results_a.get(ds, {}), m, variant) if ds in results_a else None
        rb = get_variant_result(results_b.get(ds, {}), m, variant) if ds in results_b else None

        a_means.append(float("nan") if ra is None else ra["mean"])
        a_stds.append(float("nan") if ra is None else (0.0 if is_missing(ra["std"]) else ra["std"]))
        b_means.append(float("nan") if rb is None else rb["mean"])
        b_stds.append(float("nan") if rb is None else (0.0 if is_missing(rb["std"]) else rb["std"]))

    title_variant = "GNN" if variant == "gnn" else "Directed GNN"
    plot_grouped_bars(
        categories,
        [
            {"label": label_a, "means": a_means, "stds": a_stds},
            {"label": label_b, "means": b_means, "stds": b_stds},
        ],
        title=f"{group_name.capitalize()} datasets — {title_variant}: {label_a} vs {label_b}",
        output_path=output_dir / f"{group_name}_{variant}_{label_a}_vs_{label_b}.pdf",
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
    p.add_argument("--folder1", help="First root logs folder", default="logs/run")
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