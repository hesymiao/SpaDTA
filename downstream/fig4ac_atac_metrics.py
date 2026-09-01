from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


PROJECT_ROOT = Path("/data/user/hesy/projects/SpatialMETA")
DEFAULT_MEAN_INPUT = (
    PROJECT_ROOT
    / "SpaDTA_718/runs/atac_downstream/fig4a_metrics/seven_metrics_mean.csv"
)
DEFAULT_PER_SAMPLE_INPUT = (
    PROJECT_ROOT
    / "SpaDTA_718/runs/atac_downstream/fig4a_metrics/seven_metrics_per_sample.csv"
)
DEFAULT_FIG4A_DIR = PROJECT_ROOT / "SpaDTA_718/runs/atac_downstream/fig4a"
DEFAULT_FIG4C_DIR = PROJECT_ROOT / "SpaDTA_718/runs/atac_downstream/fig4c"
E18_SAMPLE = "Mouse_Brain_E18_S1"

METHODS = [
    "SpaDTA",
    "PRESENT",
    "SMART",
    "WNN",
    "MOFA+",
    "SNF",
    "CellCharter",
    "SpatialGlue",
    "MEFISTO",
    "MultiVI",
    "COSMOS",
    "scMM",
    "MISO",
]
METRICS = ["ARI", "NMI", "AMI", "Homo", "V-Measure", "FMI", "MI"]
METHOD_COLORS = {
    "SpaDTA": "#4F5D95",
    "PRESENT": "#16A9CA",
    "SMART": "#B74B7F",
    "WNN": "#5CBC93",
    "MOFA+": "#9E365C",
    "SNF": "#D58A3A",
    "CellCharter": "#8873A4",
    "SpatialGlue": "#BEB1D6",
    "MEFISTO": "#8B97AB",
    "MultiVI": "#6A994E",
    "COSMOS": "#BC6C25",
    "scMM": "#A44A3F",
    "MISO": "#5A189A",
}


def validate_table(table: pd.DataFrame) -> pd.DataFrame:
    table = table.set_index("method")
    missing_methods = [method for method in METHODS if method not in table.index]
    missing_metrics = [metric for metric in METRICS if metric not in table.columns]
    if missing_methods or missing_metrics:
        raise ValueError(
            f"Missing methods={missing_methods or 'none'}, metrics={missing_metrics or 'none'}"
        )
    return table.loc[METHODS, METRICS]


def load_mean_table(path: Path) -> pd.DataFrame:
    return validate_table(pd.read_csv(path))


def load_sample_table(path: Path, sample: str) -> pd.DataFrame:
    table = pd.read_csv(path).set_index("method")
    if "sample" not in table.columns:
        raise ValueError(f"Per-sample table {path} is missing the sample column")
    sample_table = table.loc[table["sample"].astype(str).eq(sample)].reset_index()
    if sample_table.empty:
        raise ValueError(f"Sample {sample} not found in {path}")
    return validate_table(sample_table)


def plot_metrics(table: pd.DataFrame, output_stem: Path) -> list[Path]:
    n_methods = len(METHODS)
    group_centers = np.arange(len(METRICS), dtype=float)
    group_width = 0.82
    bar_width = group_width / n_methods

    fig, ax = plt.subplots(figsize=(13.0, 5.3), dpi=220)
    for method_index, method in enumerate(METHODS):
        offset = (method_index - (n_methods - 1) / 2) * bar_width
        ax.bar(
            group_centers + offset,
            table.loc[method].to_numpy(dtype=float),
            width=bar_width * 0.94,
            color=METHOD_COLORS[method],
            edgecolor="#333333",
            linewidth=0.25,
            zorder=3,
        )

    ax.set_xticks(group_centers)
    ax.set_xticklabels(METRICS, fontsize=12)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_xlim(-0.55, len(METRICS) - 0.45)
    upper = float(np.nanmax(table.to_numpy(dtype=float)))
    ax.set_ylim(0, upper * 1.08)
    ax.tick_params(axis="y", labelsize=11, width=0.8, length=4)
    ax.tick_params(axis="x", width=0.8, length=4)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.65, alpha=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    handles = [Patch(facecolor=METHOD_COLORS[m], edgecolor="#333333", label=m) for m in METHODS]
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.015, 0.5),
        ncol=2,
        frameon=False,
        fontsize=10.5,
        handlelength=1.4,
        handletextpad=0.45,
        columnspacing=1.0,
        borderaxespad=0,
    )

    fig.subplots_adjust(left=0.075, right=0.78, bottom=0.14, top=0.97)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_stem.with_suffix(".svg"),
        output_stem.with_suffix(".png"),
        output_stem.with_suffix(".pdf"),
    ]
    for path in outputs:
        save_kwargs = {"bbox_inches": "tight"}
        if path.suffix == ".png":
            save_kwargs["dpi"] = 300
        fig.savefig(path, **save_kwargs)
    plt.close(fig)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the seven SMART-style supervised metrics for all ATAC methods."
    )
    parser.add_argument("--mean-input", type=Path, default=DEFAULT_MEAN_INPUT)
    parser.add_argument("--per-sample-input", type=Path, default=DEFAULT_PER_SAMPLE_INPUT)
    parser.add_argument("--fig4a-dir", type=Path, default=DEFAULT_FIG4A_DIR)
    parser.add_argument("--fig4c-dir", type=Path, default=DEFAULT_FIG4C_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = []
    outputs.extend(
        plot_metrics(
            load_mean_table(args.mean_input),
            args.fig4a_dir / "fig4a_atac_metrics",
        )
    )
    outputs.extend(
        plot_metrics(
            load_sample_table(args.per_sample_input, E18_SAMPLE),
            args.fig4c_dir / "fig4c_e18_atac_metrics",
        )
    )
    print("Generated files:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
