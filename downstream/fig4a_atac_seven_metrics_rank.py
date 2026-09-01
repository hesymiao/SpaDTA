from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/data/user/hesy/projects/SpatialMETA")
INPUT_CSV = (
    PROJECT_ROOT
    / "SpaDTA_718/runs/atac_downstream/fig4a_metrics/seven_metrics_mean.csv"
)
OUTPUT_STEM = (
    PROJECT_ROOT
    / "SpaDTA_718/runs/atac_downstream/fig4a/fig4a_atac_seven_metrics_rank"
)

METRICS = ["ARI", "NMI", "AMI", "Homo", "V-Measure", "FMI", "MI"]
PALETTE = ["#49006A", "#AE007E", "#F768A1", "#FCC5C0", "#FFE9DE"]
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
FONT_SIZE = 12


def normalize_rank_best(values: pd.Series) -> pd.Series:
    valid = values.astype(float)
    mask = valid.notna()
    result = pd.Series(np.nan, index=values.index, dtype=float)
    if mask.sum() == 1:
        result.loc[mask] = 1.0
    elif mask.sum() > 1:
        ranks = valid.loc[mask].rank(method="average", ascending=False)
        result.loc[mask] = (float(mask.sum()) - ranks) / float(mask.sum() - 1)
    return result


def load_and_order() -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    table = pd.read_csv(INPUT_CSV)
    missing = [column for column in ["method", *METRICS] if column not in table.columns]
    if missing:
        raise ValueError(f"Missing columns in {INPUT_CSV}: {', '.join(missing)}")

    rank_scores = pd.DataFrame(
        {metric: normalize_rank_best(table[metric]) for metric in METRICS},
        index=table.index,
    )
    table["__mean_rank_score"] = rank_scores.mean(axis=1)
    table["__priority"] = (table["method"] != "SpaDTA").astype(int)
    table = table.sort_values(
        ["__priority", "__mean_rank_score"],
        ascending=[True, False],
        kind="stable",
    ).drop(columns="__priority").reset_index(drop=True)
    rank_map = {metric: normalize_rank_best(table[metric]) for metric in METRICS}
    return table, rank_map


def plot_rank_matrix(table: pd.DataFrame, rank_map: dict[str, pd.Series]) -> None:
    methods = table["method"].astype(str).tolist()
    x_positions = np.arange(len(METRICS), dtype=float)
    cmap = LinearSegmentedColormap.from_list("bio_conservation", PALETTE)

    fig_w = max(14.2, 1.4 + len(METRICS) * 0.68)
    fig_h = max(5.5, 1.2 + len(methods) * 0.62)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    for row_index in range(len(methods)):
        ax.axhline(row_index, color="#e6e6e6", lw=0.8, zorder=0)

    banner_width = max((len(METRICS) - 1) + 0.84, 0.18 * len("Clustering agreement"))
    banner = FancyBboxPatch(
        (((len(METRICS) - 1) / 2) - banner_width / 2, len(methods) + 0.28),
        banner_width,
        0.55,
        boxstyle="round,pad=0.02,rounding_size=0.14",
        linewidth=0,
        facecolor=PALETTE[0],
        alpha=0.96,
        clip_on=False,
        zorder=4,
    )
    ax.add_patch(banner)
    ax.text(
        (len(METRICS) - 1) / 2,
        len(methods) + 0.56,
        "Clustering agreement",
        ha="center",
        va="center",
        fontsize=FONT_SIZE,
        color="white",
        fontweight="bold",
        zorder=5,
    )

    for row_index, row in table.iterrows():
        y_position = len(methods) - 1 - row_index
        for metric_index, metric in enumerate(METRICS):
            if pd.isna(row[metric]):
                continue
            score = float(rank_map[metric].iloc[row_index])
            ax.scatter(
                metric_index,
                y_position,
                s=np.interp(score, [0.0, 1.0], [50.0, 420.0]),
                c=[cmap(score)],
                edgecolors="#202020",
                linewidths=0.45,
                zorder=3,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(METRICS, rotation=45, ha="right", fontsize=FONT_SIZE)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods[::-1], fontsize=FONT_SIZE)
    for tick in ax.get_yticklabels():
        tick.set_color(METHOD_COLORS.get(tick.get_text(), "#222222"))
        tick.set_fontweight("bold")
    ax.set_xlim(-0.9, len(METRICS) - 0.25)
    ax.set_ylim(-0.7, len(methods) + 1.0)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ranking_ax = fig.add_axes([0.82, 0.59, 0.16, 0.13])
    ranking_ax.set_axis_off()
    ranking_ax.text(
        0.02,
        1.02,
        "Ranking (1 = best)",
        fontsize=FONT_SIZE - 1,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    n_boxes = max(len(methods) * 2 + 1, 15)
    x0, x1 = 0.10, 0.94
    box_gap = 0.004
    box_w_full = (x1 - x0 - box_gap * (n_boxes - 1)) / n_boxes
    box_w = box_w_full * 0.76
    row_w = n_boxes * box_w + box_gap * (n_boxes - 1)
    row_x0 = (x0 + x1 - row_w) / 2
    for index in range(n_boxes):
        value = index / (n_boxes - 1)
        ranking_ax.add_patch(
            Rectangle(
                (row_x0 + index * (box_w + box_gap), 0.86),
                box_w,
                0.075,
                facecolor=cmap(value),
                edgecolor="#7a7a7a",
                linewidth=0.4,
            )
        )
    ranking_ax.text(
        x0,
        0.715,
        f"{len(methods)} (worst)",
        fontsize=FONT_SIZE - 2,
        color="#555555",
        ha="left",
        va="top",
    )
    ranking_ax.text(
        x1,
        0.715,
        "1 (best)",
        fontsize=FONT_SIZE - 2,
        color="#555555",
        ha="right",
        va="top",
    )

    score_ax = fig.add_axes([0.82, 0.31, 0.16, 0.10])
    score_ax.set_axis_off()
    score_ax.text(
        0.5,
        0.98,
        "Score",
        fontsize=FONT_SIZE,
        fontweight="bold",
        ha="center",
        va="top",
    )
    score_values = np.linspace(0.0, 1.0, 6)
    score_x = np.linspace(0.14, 0.86, len(score_values))
    for x, value in zip(score_x, score_values):
        score_ax.scatter(
            [x],
            [0.52],
            s=np.interp(value, [0.0, 1.0], [35.0, 300.0]),
            facecolors="none",
            edgecolors="#7a7a7a",
            linewidths=0.8,
        )
    score_ax.text(
        score_x[0] - 0.02,
        0.18,
        "0%",
        fontsize=FONT_SIZE,
        color="#555555",
        ha="left",
        va="center",
    )
    score_ax.text(
        score_x[-1] + 0.02,
        0.18,
        "100%",
        fontsize=FONT_SIZE,
        color="#555555",
        ha="right",
        va="center",
    )
    score_ax.set_xlim(0, 1)
    score_ax.set_ylim(0, 1)

    fig.subplots_adjust(left=0.12, right=0.79, bottom=0.2, top=0.9)
    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"), bbox_inches="tight", format="svg")
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    table, rank_map = load_and_order()
    plot_rank_matrix(table, rank_map)


if __name__ == "__main__":
    main()
