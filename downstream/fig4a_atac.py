from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/data/user/hesy/projects/SpatialMETA")
OUT_DIR = PROJECT_ROOT / "SpaDTA_718/runs/atac_result/now_result"
DEFAULT_SUMMARY_CSV = OUT_DIR / "plot_summary_mean.csv"
DEFAULT_OUT_SVG = PROJECT_ROOT / "SpaDTA_718/runs/atac_downstream/fig4a_atac.svg"

PALETTES = {
    "overall": ["#2D203E", "#704774", "#B77D9F", "#E0B3BD", "#F4E2E1"],
    "continuity": ["#8c2104", "#f93f06", "#ff6b35", "#f7c59f", "#ffe8dc"],
    "bio": ["#49006A", "#AE007E", "#F768A1", "#FCC5C0", "#FFE9DE"],
}

METHOD_COLOR_DICT = {
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

UNIFIED_FONT_SIZE = 12
def create_colormap(colors: list[str]) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("custom_map", colors)


def normalize_rank_best(values: pd.Series) -> pd.Series:
    """Map worst to 0 and best to 1 using within-metric descending rank."""
    valid = values.astype(float)
    mask = valid.notna()
    result = pd.Series(np.nan, index=values.index, dtype=float)
    if mask.sum() == 0:
        return result
    if mask.sum() == 1:
        result.loc[mask] = 1.0
        return result
    ranks = valid.loc[mask].rank(method="average", ascending=False)
    result.loc[mask] = (float(mask.sum()) - ranks) / float(mask.sum() - 1)
    return result


def plot_visual_benchmark(df_mean: pd.DataFrame, out_svg: Path) -> None:
    groups = [
        ("Overall", [("overall_score", "Overall")], PALETTES["overall"]),
        ("Continuity", [("CHAOS", "1-CHAOS"), ("PAS", "1-PAS"), ("continuity_mean", "Score")], PALETTES["continuity"]),
        (
            "Bio conservation",
            [
                ("ARI", "ARI"),
                ("NMI", "NMI"),
                ("gt_silhouette", "Cell type ASW"),
                ("isolated_asw", "Isolated label"),
                ("clisi_graph", "Graph cLISI"),
                ("biological_conservation_mean", "Score"),
            ],
            PALETTES["bio"],
        ),
    ]
    groups = [
        (
            group_name,
            [(col, label) for col, label in items if col in df_mean.columns and df_mean[col].notna().any()],
            palette,
        )
        for group_name, items, palette in groups
    ]
    groups = [(group_name, items, palette) for group_name, items, palette in groups if items]

    ordered = df_mean[df_mean["overall_score"].notna()].copy()
    ordered["__priority"] = (ordered["method"] != "SpaDTA").astype(int)
    ordered = ordered.sort_values(["__priority", "overall_score"], ascending=[True, False], kind="stable").drop(columns="__priority")
    methods = ordered["method"].tolist()

    x_positions: dict[str, float] = {}
    x_ticks: list[float] = []
    x_labels: list[str] = []
    group_meta: list[tuple[str, float, float, list[str], list[str]]] = []
    x = 0.0
    for group_name, items, palette in groups:
        start = x
        columns = []
        for col, label in items:
            x_positions[col] = x
            x_ticks.append(x)
            x_labels.append(label)
            columns.append(col)
            x += 1.0
        end = x - 1.0
        group_meta.append((group_name, start, end, columns, palette))
        x += 0.7

    fig_w = max(14.2, 1.4 + len(x_ticks) * 0.68)
    fig_h = max(5.5, 1.2 + len(methods) * 0.62)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    rank_norm_map: dict[str, pd.Series] = {
        col: normalize_rank_best(ordered[col])
        for _, items, _ in groups
        for col, _ in items
    }
    for idx in range(len(methods)):
        ax.axhline(idx, color="#e6e6e6", lw=0.8, zorder=0)

    for group_name, start, end, _, palette in group_meta:
        banner_width = max((end - start) + 0.84, 0.18 * len(group_name))
        rect = FancyBboxPatch(
            (((start + end) / 2) - (banner_width / 2), len(methods) + 0.28),
            banner_width,
            0.55,
            boxstyle="round,pad=0.02,rounding_size=0.14",
            linewidth=0,
            facecolor=palette[0],
            alpha=0.96,
            clip_on=False,
            zorder=4,
        )
        ax.add_patch(rect)
        ax.text(
            (start + end) / 2,
            len(methods) + 0.56,
            group_name,
            ha="center",
            va="center",
            fontsize=UNIFIED_FONT_SIZE,
            color="white",
            fontweight="bold",
            zorder=5,
        )

    for _, _, _, columns, palette in group_meta:
        cmap = create_colormap(palette)
        for row_idx, row in enumerate(ordered.itertuples(index=False)):
            for col in columns:
                value = getattr(row, col)
                if pd.isna(value):
                    continue
                xpos = x_positions[col]
                ypos = len(methods) - 1 - row_idx
                color_value = float(rank_norm_map[col].iloc[row_idx])
                size_value = color_value
                size = np.interp(size_value, [0.0, 1.0], [50.0, 420.0])
                ax.scatter(
                    xpos,
                    ypos,
                    s=size,
                    c=[cmap(color_value)],
                    edgecolors="#202020",
                    linewidths=0.45,
                    zorder=3,
                )

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=UNIFIED_FONT_SIZE)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods[::-1], fontsize=UNIFIED_FONT_SIZE)
    for tick in ax.get_yticklabels():
        tick.set_color(METHOD_COLOR_DICT.get(tick.get_text(), "#222222"))
        tick.set_fontweight("bold")
    ax.set_xlim(min(x_ticks) - 0.9, max(x_ticks) + 0.75)
    ax.set_ylim(-0.7, len(methods) + 1.0)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ranking_groups = [
        ("Overall", PALETTES["overall"]),
        ("Continuity", PALETTES["continuity"]),
        ("Bio conservation", PALETTES["bio"]),
    ]
    ranking_groups = [
        (name, colors)
        for name, colors in ranking_groups
        if any(meta_name == name for meta_name, _, _, _, _ in group_meta)
    ]

    ranking_ax = fig.add_axes([0.82, 0.59, 0.16, 0.13])
    ranking_ax.set_axis_off()
    ranking_ax.text(
        0.02,
        1.02,
        "Ranking (1 = best)",
        fontsize=UNIFIED_FONT_SIZE - 1,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    n_boxes = max(len(methods) * 2 + 1, 15)
    x0 = 0.10
    x1 = 0.94
    box_gap = 0.004
    row_gap = 0.11
    box_w_full = (x1 - x0 - box_gap * (n_boxes - 1)) / n_boxes
    box_w = box_w_full * 0.76
    row_w = n_boxes * box_w + box_gap * (n_boxes - 1)
    row_x0 = (x0 + x1 - row_w) / 2
    box_h = 0.075
    for row_idx, (_, colors) in enumerate(ranking_groups):
        cmap = create_colormap(colors)
        y = 0.86 - row_idx * row_gap
        for i in range(n_boxes):
            color_value = i / (n_boxes - 1)
            ranking_ax.add_patch(
                Rectangle(
                    (row_x0 + i * (box_w + box_gap), y),
                    box_w,
                    box_h,
                    facecolor=cmap(color_value),
                    edgecolor="#7a7a7a",
                    linewidth=0.4,
                )
            )
    legend_y = 0.86 - len(ranking_groups) * row_gap - 0.035
    ranking_ax.text(
        x0,
        legend_y,
        f"{len(methods)} (worst)",
        fontsize=UNIFIED_FONT_SIZE - 2,
        color="#555555",
        ha="left",
        va="top",
    )
    ranking_ax.text(
        x1,
        legend_y,
        "1 (best)",
        fontsize=UNIFIED_FONT_SIZE - 2,
        color="#555555",
        ha="right",
        va="top",
    )

    score_ax = fig.add_axes([0.82, 0.31, 0.16, 0.10])
    score_ax.set_axis_off()
    score_ax.text(0.5, 0.98, "Score", fontsize=UNIFIED_FONT_SIZE, fontweight="bold", ha="center", va="top")
    score_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
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
        fontsize=UNIFIED_FONT_SIZE,
        color="#555555",
        ha="left",
        va="center",
    )
    score_ax.text(
        score_x[-1] + 0.02,
        0.18,
        "100%",
        fontsize=UNIFIED_FONT_SIZE,
        color="#555555",
        ha="right",
        va="center",
    )
    score_ax.set_xlim(0, 1)
    score_ax.set_ylim(0, 1)

    fig.subplots_adjust(left=0.12, right=0.79, bottom=0.2, top=0.9)
    output_format = out_svg.suffix.lower().lstrip(".") or "svg"
    fig.savefig(out_svg, bbox_inches="tight", format=output_format)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot target-count visual benchmark from an existing summary table.")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_SUMMARY_CSV,
        help="Existing summary CSV. Default points to the current all-method mclust table.",
    )
    parser.add_argument(
        "--out-svg",
        type=Path,
        default=DEFAULT_OUT_SVG,
        help="Output SVG path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.summary_csv.exists():
        raise FileNotFoundError(f"Summary CSV not found: {args.summary_csv}")

    df_mean = pd.read_csv(args.summary_csv)
    df_mean = df_mean.rename(
        columns={"1-CHAOS": "CHAOS", "1-PAS": "PAS"}
    )
    required_columns = {
        "method",
        "overall_score",
        "CHAOS",
        "PAS",
        "continuity_mean",
        "ARI",
        "NMI",
        "gt_silhouette",
        "isolated_asw",
        "clisi_graph",
        "biological_conservation_mean",
    }
    missing_columns = sorted(required_columns.difference(df_mean.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns in {args.summary_csv}: {', '.join(missing_columns)}")

    args.out_svg.parent.mkdir(parents=True, exist_ok=True)
    plot_visual_benchmark(df_mean, args.out_svg)


if __name__ == "__main__":
    main()
