from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, LinearLocator


PROJECT_ROOT = Path("/data/user/hesy/projects/SpatialMETA")
METRICS_DIR = PROJECT_ROOT / "SpaDTA_718/runs/atac_downstream/fig4a_metrics"
OUT_DIR = PROJECT_ROOT / "SpaDTA_718/runs/atac_downstream/fig4ijk_atac"

DATASET_GROUPS = {
    "E11": ["Mouse_Brain_E11_S1"],
    "E13": ["Mouse_Brain_E13_S1"],
    "E15": ["Mouse_Brain_E15_S1"],
    "E18": ["Mouse_Brain_E18_S1"],
    "ATAC_mean": [
        "Mouse_Brain_E11_S1", "Mouse_Brain_E13_S1",
        "Mouse_Brain_E15_S1", "Mouse_Brain_E18_S1",
    ],
}

METHOD_COLORS = {
    "SpaDTA": "#4F5D95",
    "PRESENT": "#16A9CA", "SMART": "#B74B7F", "WNN": "#5CBC93",
    "MOFA+": "#9E365C", "SNF": "#D58A3A", "CellCharter": "#8873A4",
    "SpatialGlue": "#BEB1D6", "MEFISTO": "#8B97AB", "MultiVI": "#6A994E",
    "COSMOS": "#BC6C25", "scMM": "#A44A3F", "MISO": "#5A189A",
}

UNIFIED_FONT_SIZE = 25

METRIC_SOURCES = {
    "1-CHAOS": "CHAOS",
    "1-PAS": "PAS",
}

METRICS = list(METRIC_SOURCES.keys())


def load_method_order(metrics_dir: Path) -> list[str]:
    table = pd.read_csv(metrics_dir / "CHAOS.csv")
    return table["method"].astype(str).tolist()


def prepare_metrics_table(metrics_dir: Path) -> pd.DataFrame:
    frames = []
    for source_name in METRIC_SOURCES.values():
        wide = pd.read_csv(metrics_dir / f"{source_name}.csv").drop(columns="mean", errors="ignore")
        frames.append(wide.melt(id_vars="method", var_name="sample", value_name=source_name))
    df = frames[0].merge(frames[1], on=["method", "sample"], validate="one_to_one")
    for metric_name, source_name in METRIC_SOURCES.items():
        df[metric_name] = 1.0 - pd.to_numeric(df[source_name], errors="coerce")
    return df


def save_method_legend(out_dir: Path, methods: list[str]) -> list[Path]:
    legend_handles = [
        Patch(facecolor=METHOD_COLORS.get(method, "#888888"), edgecolor="white", label=method)
        for method in methods
    ]
    fig, ax = plt.subplots(figsize=(18, 2.8), dpi=220)
    ax.set_axis_off()
    ax.legend(
        handles=legend_handles,
        loc="center",
        ncol=4,
        frameon=False,
        fontsize=UNIFIED_FONT_SIZE,
        handlelength=1.6,
        columnspacing=1.4,
    )
    fig.tight_layout()
    out_svg = out_dir / "method_legend.svg"
    out_png = out_dir / "method_legend.png"
    fig.savefig(out_svg, bbox_inches="tight", format="svg")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return [out_png, out_svg]


def plot_group_metric(
    df: pd.DataFrame,
    method_order: list[str],
    group_name: str,
    samples: list[str],
    metric: str,
    out_dir: Path,
) -> list[Path]:
    available_methods = [method for method in method_order if method in set(df["method"])]
    if not available_methods:
        raise ValueError("No expected methods found in metrics table.")

    subset = df[df["sample"].isin(samples)].copy()
    missing_samples = [sample for sample in samples if sample not in set(subset["sample"])]
    if missing_samples:
        raise ValueError(f"{group_name} missing samples for {metric}: {missing_samples}")

    n_methods = len(available_methods)
    mean_values = (
        subset.groupby("method", as_index=True)[metric]
        .mean()
        .reindex(available_methods)
        .astype(float)
    )
    x = np.arange(n_methods, dtype=float)
    bar_width = 0.68

    fig_width = 5.0
    fig, ax = plt.subplots(figsize=(fig_width, 4.6), dpi=220)

    ax.bar(
        x,
        mean_values.to_numpy(),
        width=bar_width,
        color=[METHOD_COLORS.get(method, "#888888") for method in available_methods],
        edgecolor="white",
        linewidth=0.7,
    )

    left_pad = 0.08
    ax.set_xlim(x[0] - bar_width / 2 - left_pad, x[-1] + bar_width / 2)

    max_value = float(np.nanmax(mean_values.to_numpy(dtype=float)))
    ax.set_ylim(0, max_value * 1.22 if max_value > 0 else 1.0)
    ax.yaxis.set_major_locator(LinearLocator(6))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_xticks([])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("")
    ax.set_title(f"{metric}({group_name})", fontsize=UNIFIED_FONT_SIZE, fontweight="bold", pad=12)
    ax.tick_params(axis="y", labelsize=UNIFIED_FONT_SIZE)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")

    fig.tight_layout()
    safe_metric = metric.replace("-", "_minus_")
    out_png = out_dir / f"{group_name}_{safe_metric}_mean_bar.png"
    out_pdf = out_dir / f"{group_name}_{safe_metric}_mean_bar.pdf"
    out_svg = out_dir / f"{group_name}_{safe_metric}_mean_bar.svg"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight", format="svg")
    plt.close(fig)
    return [out_png, out_pdf, out_svg]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Fig. 1 I-K 1-CHAOS/1-PAS bars by dataset group.")
    parser.add_argument("--metrics-dir", type=Path, default=METRICS_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = prepare_metrics_table(args.metrics_dir)
    method_order = load_method_order(args.metrics_dir)
    outputs: list[Path] = []
    available_methods = [method for method in method_order if method in set(df["method"])]
    outputs.extend(save_method_legend(args.out_dir, available_methods))
    for group_name, samples in DATASET_GROUPS.items():
        for metric in METRICS:
            outputs.extend(plot_group_metric(df, method_order, group_name, samples, metric, args.out_dir))

    print("Generated files:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
