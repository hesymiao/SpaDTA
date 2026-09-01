from __future__ import annotations

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


project_root = Path("/data/user/hesy/projects/SpatialMETA")

sample = "X49_T"
run_root = project_root / "SpaDTA_718" / "runs" / "sm_downstream"
input_h5ad = run_root / "inputs" / sample / f"{sample}_output.h5ad"
cluster_label_h5ad = input_h5ad
feature_source_h5ad = input_h5ad
output_dir = run_root / "fig2e"

CLUSTER_KEY = "decalign_linear_clusters"
LAYER = "normalized"
MARGIN_MIXED = 0.25
MARGIN_HIGH = 0.6

cluster_key = CLUSTER_KEY
major_value = "Imm"
margin_mixed = MARGIN_MIXED
margin_high = MARGIN_HIGH

MARKER_SETS = {
    "Imm": ["CD3D", "CD74", "PTPRC", "NKG7"],
    "Endo": ["CD34", "PECAM1", "APLN"],
    "Stro": ["COL3A1", "ACTA2", "COL1A1", "DCN", "LUM"],
    "Mal": ["NDUFA4L2", "CNDP2", "EGFR", "CA9", "EPCAM", "KRT8", "KRT18", "KRT19"],
}

GROUP_COLOR_PALETTES = {
    "Imm": ["#5B2A86", "#4E55A7", "#9B4F96", "#6F58A4", "#9A86BA", "#B496C9"],
    "Endo": ["#5B2A86", "#4E55A7", "#9B4F96", "#6F58A4", "#9A86BA", "#B496C9"],
    "Stro": ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"],
    "Mal": ["#C65D4B", "#D57562", "#DF8D79", "#E7A48F", "#EFBBB0", "#F4D0C7"],
    "Mixed": ["#7B2CBF", "#9253C6", "#A96BCD", "#C08BD7", "#D5AEE3", "#E7D1F0"],
}


def to_dense(x) -> np.ndarray:
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def natural_sort(values: list[str]) -> list[str]:
    def key(value: str) -> tuple[int, object]:
        text = str(value)
        return (0, int(text)) if text.isdigit() else (1, text)

    return sorted(values, key=key)


def load_adata(
    cluster_label_h5ad: Path,
    feature_source_h5ad: Path,
    cluster_key: str,
) -> sc.AnnData:
    label_adata = sc.read_h5ad(cluster_label_h5ad).copy()
    label_adata.obs_names = label_adata.obs_names.astype(str)

    feature_adata = sc.read_h5ad(feature_source_h5ad).copy()
    feature_adata.obs_names = feature_adata.obs_names.astype(str)
    if "name" in feature_adata.var.columns:
        feature_adata.var_names = feature_adata.var["name"].astype(str).values
        feature_adata.var_names_make_unique()

    if not feature_adata.obs_names.equals(label_adata.obs_names):
        raise ValueError("cluster-label h5ad and feature-source h5ad have mismatched obs_names.")
    if cluster_key not in label_adata.obs.columns:
        raise KeyError(f"missing obs[{cluster_key!r}] in {cluster_label_h5ad}")
    feature_adata.obs[cluster_key] = label_adata.obs[cluster_key].astype(str).values
    return feature_adata


def annotate_single_sample_group(
    adata: sc.AnnData,
    major_value: str,
    cluster_key: str,
    margin_mixed: float,
    margin_high: float,
) -> tuple[sc.AnnData, pd.DataFrame]:
    adata.obs[cluster_key] = adata.obs[cluster_key].astype(str)

    marker_sets = {k: [g for g in v if g in adata.var_names] for k, v in MARKER_SETS.items()}
    genes = [g for values in marker_sets.values() for g in values]
    X = adata[:, genes].layers[LAYER]
    X = to_dense(X).astype(np.float32, copy=False)
    df = pd.DataFrame(X, columns=genes)
    df["cluster"] = adata.obs[cluster_key].astype(str).values
    means = df.groupby("cluster", observed=True).mean()
    means = means.loc[natural_sort(means.index.astype(str).tolist())]
    zscore = (means - means.mean(axis=0)) / means.std(axis=0).replace(0, np.nan)
    zscore = zscore.fillna(0.0)

    scores = pd.DataFrame(index=means.index)
    for label, genes_in_set in marker_sets.items():
        scores[label] = zscore[genes_in_set].mean(axis=1)

    cluster_sizes = adata.obs[cluster_key].astype(str).value_counts()
    rows: list[dict[str, object]] = []
    for cluster in scores.index.astype(str):
        score_map = {label: float(scores.loc[cluster, label]) for label in marker_sets}
        ranked = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)
        best, best_score = ranked[0]
        second, second_score = ranked[1]
        margin = float(best_score - second_score)
        broad = "Mixed" if margin < float(margin_mixed) else best
        rows.append(
            {
                "cluster": cluster,
                "n_spots": int(cluster_sizes[cluster]),
                "annotation_broad": broad,
                "margin": margin,
                **score_map,
            }
        )
    annotation_df = pd.DataFrame(rows).sort_values("cluster", key=lambda s: s.astype(int)).reset_index(drop=True)
    group_df = annotation_df.loc[annotation_df["annotation_broad"].eq(major_value)].copy()
    group_df = group_df.sort_values(["n_spots", "cluster"], ascending=[False, True]).reset_index(drop=True)
    group_df["marker_named_cluster"] = [f"{major_value}_{i}" for i in range(1, len(group_df) + 1)]
    cluster_to_label = dict(zip(group_df["cluster"].astype(str), group_df["marker_named_cluster"].astype(str)))
    adata.obs["marker_named_cluster"] = adata.obs[cluster_key].astype(str).map(cluster_to_label).fillna("").astype(str)
    return adata, group_df


def to_hires_coords(adata: sc.AnnData, sample: str) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    spatial_block = adata.uns["spatial"][sample]
    image = np.asarray(spatial_block["images"]["hires"])
    scalefactors = spatial_block["scalefactors"]
    scale = float(scalefactors["tissue_hires_scalef"])
    coords = np.asarray(adata.obsm["spatial"], dtype=float)[:, :2] * scale
    return coords, image, scalefactors


def build_legend(ax: plt.Axes, labels: list[str], color_map: dict[str, str]) -> None:
    handles = [
        Line2D([0], [0], marker="o", color="w", label=label, markerfacecolor=color_map[label], markersize=14)
        for label in labels
    ]
    legend = ax.legend(
        handles=handles,
        frameon=False,
        ncol=len(handles),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        handlelength=0.8,
        handletextpad=0.4,
        columnspacing=2.6,
        borderaxespad=0.0,
        fontsize=19,
    )
    for text, label in zip(legend.get_texts(), labels):
        text.set_color(color_map[label])
        text.set_fontweight("bold")


def gray_fill_value(image: np.ndarray) -> np.ndarray:
    channels = 1 if image.ndim == 2 else image.shape[2]
    if np.issubdtype(image.dtype, np.integer):
        value = np.full((channels,), 225, dtype=image.dtype)
    else:
        value = np.full((channels,), 0.88, dtype=image.dtype)
    return value[0] if channels == 1 else value


def tissue_mask_from_image(image: np.ndarray) -> np.ndarray:
    image_array = np.asarray(image)
    if np.issubdtype(image_array.dtype, np.integer):
        threshold = 245
        if image_array.ndim == 2:
            return image_array < threshold
        return np.any(image_array < threshold, axis=2)
    threshold = 0.96
    if image_array.ndim == 2:
        return image_array < threshold
    return np.any(image_array < threshold, axis=2)


def build_region_preserving_background(
    image: np.ndarray,
    region_coords: np.ndarray,
    point_diameter: float,
) -> np.ndarray:
    background = np.array(image, copy=True)
    tissue_mask = tissue_mask_from_image(background)
    if region_coords.shape[0] == 0:
        fill_value = gray_fill_value(background)
        if background.ndim == 2:
            background[tissue_mask] = fill_value
        else:
            background[tissue_mask, :] = fill_value
        return background

    height, width = background.shape[:2]
    mask = np.zeros((height, width), dtype=bool)
    radius = max(2, int(round(point_diameter * 0.78)))
    radius2 = radius * radius

    for x_coord, y_coord in region_coords:
        x_center = int(round(float(x_coord)))
        y_center = int(round(float(y_coord)))
        x0 = max(0, x_center - radius)
        x1 = min(width, x_center + radius + 1)
        y0 = max(0, y_center - radius)
        y1 = min(height, y_center + radius + 1)
        yy_local, xx_local = np.ogrid[y0:y1, x0:x1]
        local = (xx_local - x_center) ** 2 + (yy_local - y_center) ** 2 <= radius2
        mask[y0:y1, x0:x1] |= local

    fill_value = gray_fill_value(background)
    if background.ndim == 2:
        background[tissue_mask & (~mask)] = fill_value
    else:
        background[tissue_mask & (~mask), :] = fill_value
    return background


def plot_group_regions(
    adata: sc.AnnData,
    sample: str,
    major_value: str,
    group_df: pd.DataFrame,
    output_png: Path,
    output_pdf: Path,
    output_svg: Path,
    background_png: Path,
    summary_json: Path,
) -> None:
    labels = adata.obs["marker_named_cluster"].astype(str)
    group_labels = [label for label in group_df["marker_named_cluster"].astype(str).tolist() if label]
    mask = labels.isin(group_labels).to_numpy()
    coords_hires, image, scalefactors = to_hires_coords(adata, sample)
    group_coords = coords_hires[mask]
    group_labels_per_spot = labels.to_numpy()[mask]

    present_labels = [label for label in group_labels if np.any(group_labels_per_spot == label)]
    palette = GROUP_COLOR_PALETTES.get(major_value, GROUP_COLOR_PALETTES["Mixed"])
    color_map = {label: palette[i % len(palette)] for i, label in enumerate(group_labels)}
    point_diameter = float(scalefactors["spot_diameter_fullres"]) * float(scalefactors["tissue_hires_scalef"])
    point_size = max((point_diameter * 0.38) ** 2, 12.0)
    background_image = build_region_preserving_background(image, group_coords, point_diameter)

    background_fig, background_ax = plt.subplots(figsize=(10.8, 8.0))
    background_ax.imshow(background_image, origin="upper")
    background_ax.set_xticks([])
    background_ax.set_yticks([])
    for spine in background_ax.spines.values():
        spine.set_visible(False)
    background_fig.tight_layout()
    background_fig.savefig(background_png, dpi=260, bbox_inches="tight")
    plt.close(background_fig)

    fig, ax = plt.subplots(figsize=(10.8, 8.0))
    ax.imshow(background_image, origin="upper")

    for label in present_labels:
        label_mask = group_labels_per_spot == label
        ax.scatter(
            group_coords[label_mask, 0],
            group_coords[label_mask, 1],
            s=point_size,
            c=color_map.get(label, "#666666"),
            alpha=0.98,
            linewidths=0.35,
            edgecolors="white",
            rasterized=True,
            zorder=3,
        )

    build_legend(ax, present_labels, color_map)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(output_png, dpi=260, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_svg, bbox_inches="tight", format="svg")
    plt.close(fig)

    summary = {
        "sample": sample,
        "figure_png": str(output_png),
        "figure_pdf": str(output_pdf),
        "figure_svg": str(output_svg),
        "background_png": str(background_png),
        "present_labels": present_labels,
        "major_value": major_value,
        "n_group_spots": int(mask.sum()),
        "spot_diameter_hires": point_diameter,
        "point_size": point_size,
        "single_sample_group_clusters": group_df.to_dict(orient="records"),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_group_regions_on_slice(
    sample: str,
    input_h5ad: Path,
    output_dir: Path,
    cluster_label_h5ad: Path | None = None,
    feature_source_h5ad: Path | None = None,
    cluster_key: str = CLUSTER_KEY,
    major_value: str = "Imm",
    margin_mixed: float = MARGIN_MIXED,
    margin_high: float = MARGIN_HIGH,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{sample}_{str(major_value).lower()}_regions_on_slice"
    output_png = output_dir / f"{stem}.png"
    output_pdf = output_dir / f"{stem}.pdf"
    output_svg = output_dir / f"{stem}.svg"
    background_png = output_dir / f"{stem}_background.png"
    summary_json = output_dir / f"{stem}_summary.json"

    cluster_label_path = Path(cluster_label_h5ad or input_h5ad)
    feature_source_path = Path(feature_source_h5ad or input_h5ad)
    adata = load_adata(cluster_label_path, feature_source_path, cluster_key)
    adata, group_df = annotate_single_sample_group(
        adata,
        major_value,
        cluster_key,
        margin_mixed,
        margin_high,
    )
    plot_group_regions(adata, sample, major_value, group_df, output_png, output_pdf, output_svg, background_png, summary_json)
    result = {"figure": str(output_png), "summary": str(summary_json)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    plot_group_regions_on_slice(
        sample=sample,
        input_h5ad=input_h5ad,
        output_dir=output_dir,
        cluster_label_h5ad=cluster_label_h5ad,
        feature_source_h5ad=feature_source_h5ad,
        cluster_key=cluster_key,
        major_value=major_value,
        margin_mixed=margin_mixed,
        margin_high=margin_high,
    )
