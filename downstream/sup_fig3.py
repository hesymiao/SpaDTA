from __future__ import annotations

import json
import logging
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import scanpy as sc
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from molmass import Formula
from PIL import Image, ImageDraw, ImageFont
from scipy import sparse
from scipy.stats import hypergeom


PROJECT_ROOT = Path("/data/user/hesy/projects/SpatialMETA")
INPUT_H5AD = Path(
    "/bigdat2/user/hesy/spatialmeta/SpatialMETA/"
    "SpaDTA_718_model_input_preselect800_20260719/SM/m3_FMP.h5ad"
)
REFERENCE_CLUSTER_IMAGE = PROJECT_ROOT / "compare_method/common/now_result/fig/SpaDTA/m3_FMP.png"
OUTPUT_ROOT = PROJECT_ROOT / "SpaDTA_718/runs/sm_downstream/sup_fig3"
FIGURE_DIR = OUTPUT_ROOT / "figures"
TABLE_DIR = OUTPUT_ROOT / "tables"
HMDB_PATH = PROJECT_ROOT / "spatialmeta/data/hmdb.csv"

SAMPLE = "m3_FMP"
CLUSTER_KEY = "decalign_linear_clusters"
LAYER = "normalized"
ST_COUNTS_LAYER = "counts"
ST_NORMALIZATION_TARGET = 10_000.0
ROI_CLUSTERS = ("12", "8")
ROI_LABELS = {"12": "Cluster 12", "8": "Cluster 8"}
ROI_COLORS = {"12": "#d62728", "8": "#ffbb78"}
ROI_BOUNDARY_COLORS = {"12": "#D62728", "8": "#F2C94C"}
ST_LOG2FC_THRESHOLD = 0.2
SM_LOG2FC_THRESHOLD = 0.75
LOG2FC_EPS = 1e-3
TOP_LABELS = 3
TOP_MAP_FEATURES = 2
TOP_DEG_PER_CLUSTER = 10
PPM_TOLERANCE = 5.0
ANALYSIS_COLORS = {"12": "#C62828", "8": "#B86700"}
ENRICHMENT_LOG2FC_THRESHOLD = 0.2
PPM_TOLERANCE = 5.0
ADDUCTS = [("add", "H"), ("add", "Na"), ("add", "K"), ("sub", "H"), ("sub", "Cl")]

EXPRESSION_CMAP = LinearSegmentedColormap.from_list(
    "expression_soft_teal",
    ["#f7f4ea", "#dcefe7", "#9fd3c7", "#4aa0b5", "#1f5a89"],
)
PURPLE_CMAP = LinearSegmentedColormap.from_list(
    "grey_purple", ["#d8d5dd", "#c8b7eb", "#8e61c0"]
)

# The reference figure uses the fixed SpaDTA palette, while its displayed
# cluster numbers follow the color-keyed legend supplied with this analysis.
REFERENCE_RGB_TO_CLUSTER = {
    (31, 119, 180): "1",
    (170, 64, 252): "2",
    (140, 86, 75): "3",
    (227, 119, 194): "4",
    (181, 189, 97): "5",
    (23, 190, 207): "6",
    (174, 199, 232): "7",
    (255, 187, 120): "8",
    (152, 223, 138): "9",
    (255, 127, 14): "10",
    (39, 158, 104): "11",
    (214, 39, 40): "12",
}


def to_dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def save_figure(fig: plt.Figure, stem: Path, dpi: int = 600) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fitted_hexagons(adata: sc.AnnData, coords: np.ndarray) -> list[np.ndarray]:
    if not {"array_col", "array_row"}.issubset(adata.obs.columns):
        raise KeyError("Hexagonal plotting requires obs['array_col'] and obs['array_row']")
    design = np.column_stack(
        [
            adata.obs["array_col"].to_numpy(float),
            adata.obs["array_row"].to_numpy(float),
            np.ones(adata.n_obs),
        ]
    )
    transform, _, _, _ = np.linalg.lstsq(design, coords, rcond=None)
    fitted = design @ transform
    neighbor_vectors = []
    for delta_col, delta_row in [(2, 0), (1, 1), (1, -1)]:
        vector = delta_col * transform[0] + delta_row * transform[1]
        neighbor_vectors.extend([vector, -vector])
    vertices = []
    for first_index, first in enumerate(neighbor_vectors):
        for second in neighbor_vectors[first_index + 1 :]:
            matrix = np.stack([first, second])
            if abs(np.linalg.det(matrix)) <= 1e-6:
                continue
            rhs = np.array([np.dot(first, first), np.dot(second, second)]) * 0.5
            candidate = np.linalg.solve(matrix, rhs)
            if all(
                np.dot(candidate, vector) <= np.dot(vector, vector) * 0.5 + 1e-6
                for vector in neighbor_vectors
            ):
                vertices.append(candidate)
    unique = []
    for vertex in vertices:
        if not any(np.allclose(vertex, previous, atol=1e-5) for previous in unique):
            unique.append(vertex)
    template = np.asarray(unique)
    center = template.mean(axis=0)
    angles = np.arctan2(template[:, 1] - center[1], template[:, 0] - center[0])
    template = template[np.argsort(angles)]
    return [template + point for point in fitted]


def exterior_hex_edges(polygons: list[np.ndarray], mask: np.ndarray) -> list[np.ndarray]:
    edges: dict[tuple[tuple[float, float], tuple[float, float]], tuple[int, np.ndarray]] = {}
    for polygon, keep in zip(polygons, mask):
        if not keep:
            continue
        for index in range(len(polygon)):
            segment = np.stack([polygon[index], polygon[(index + 1) % len(polygon)]])
            first = tuple(np.round(segment[0], 3))
            second = tuple(np.round(segment[1], 3))
            key = tuple(sorted([first, second]))
            count, _ = edges.get(key, (0, segment))
            edges[key] = (count + 1, segment)
    return [segment for count, segment in edges.values() if count == 1]


def add_roi_boundaries(
    ax: plt.Axes,
    polygons: list[np.ndarray],
    labels: np.ndarray,
    linewidth: float = 2.0,
) -> None:
    for cluster in ROI_CLUSTERS:
        segments = exterior_hex_edges(polygons, labels == cluster)
        if segments:
            ax.add_collection(
                LineCollection(
                    segments,
                    colors=ROI_BOUNDARY_COLORS[cluster],
                    linewidths=linewidth,
                    capstyle="round",
                    joinstyle="round",
                    zorder=8,
                )
            )


def gray_outside_roi(image: np.ndarray, polygons: list[np.ndarray], roi_mask: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        gray = array.copy()
    else:
        rgb = array[..., :3].astype(float)
        luminance = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        gray = np.repeat(luminance[..., None], array.shape[2], axis=2)
        if np.issubdtype(array.dtype, np.integer):
            gray = np.clip(gray, 0, np.iinfo(array.dtype).max).astype(array.dtype)
        else:
            gray = gray.astype(array.dtype)
    mask_image = Image.new("L", (array.shape[1], array.shape[0]), 0)
    draw = ImageDraw.Draw(mask_image)
    for polygon, keep in zip(polygons, roi_mask):
        if keep:
            draw.polygon([(float(x), float(y)) for x, y in polygon], fill=255)
    keep_mask = np.asarray(mask_image, dtype=bool)
    background = gray.copy()
    if array.ndim == 2:
        background[keep_mask] = array[keep_mask]
    else:
        background[keep_mask, :] = array[keep_mask, :]
    return background


def labels_from_reference_image(adata: sc.AnnData) -> np.ndarray:
    """Recover the exact color-keyed labels shown in the stored SpaDTA figure."""
    coords, hires_image, _ = spatial_data(adata)
    reference = np.asarray(Image.open(REFERENCE_CLUSTER_IMAGE).convert("RGB"))
    scale = np.array(
        [reference.shape[1] / hires_image.shape[1], reference.shape[0] / hires_image.shape[0]],
        dtype=float,
    )
    pixels = np.rint(coords * scale).astype(int)
    pixels[:, 0] = np.clip(pixels[:, 0], 0, reference.shape[1] - 1)
    pixels[:, 1] = np.clip(pixels[:, 1], 0, reference.shape[0] - 1)
    sampled_rgb = reference[pixels[:, 1], pixels[:, 0], :]

    palette = np.asarray(list(REFERENCE_RGB_TO_CLUSTER), dtype=np.int16)
    distances = np.linalg.norm(sampled_rgb[:, None, :].astype(np.int16) - palette[None, :, :], axis=2)
    nearest = distances.argmin(axis=1)
    nearest_distance = distances[np.arange(len(adata)), nearest]
    if np.any(nearest_distance > 0):
        raise ValueError(
            f"Reference-image label recovery was not exact for {(nearest_distance > 0).sum()} spots "
            f"(maximum RGB distance {nearest_distance.max():.2f})"
        )

    palette_clusters = np.asarray(list(REFERENCE_RGB_TO_CLUSTER.values()), dtype=object)
    labels = palette_clusters[nearest]
    label_table = pd.DataFrame(
        {
            "obs_name": adata.obs_names.astype(str),
            "reference_x": pixels[:, 0],
            "reference_y": pixels[:, 1],
            "red": sampled_rgb[:, 0],
            "green": sampled_rgb[:, 1],
            "blue": sampled_rgb[:, 2],
            CLUSTER_KEY: labels,
        }
    )
    label_table.to_csv(TABLE_DIR / "m3_FMP_reference_image_cluster_labels.csv", index=False)
    return labels


def load_data() -> sc.AnnData:
    adata = sc.read_h5ad(INPUT_H5AD)
    if "name" in adata.var.columns:
        adata.var_names = adata.var["name"].astype(str).to_numpy()
        adata.var_names_make_unique()
    adata.obs[CLUSTER_KEY] = pd.Categorical(labels_from_reference_image(adata))
    required_obs = {CLUSTER_KEY, "spot_name"}
    missing_obs = sorted(required_obs.difference(adata.obs.columns))
    if missing_obs:
        raise KeyError(f"Missing obs columns: {missing_obs}")
    if LAYER not in adata.layers:
        raise KeyError(f"Missing layer {LAYER!r}")
    if "type" not in adata.var.columns:
        raise KeyError("Missing var['type'] modality labels")
    present = set(adata.obs[CLUSTER_KEY].astype(str))
    missing_clusters = sorted(set(ROI_CLUSTERS).difference(present))
    if missing_clusters:
        raise ValueError(f"Missing requested SpaDTA clusters: {missing_clusters}")
    adata.obs[CLUSTER_KEY] = adata.obs[CLUSTER_KEY].astype(str)
    return adata


def spatial_data(adata: sc.AnnData) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if "spatial" not in adata.uns or not adata.uns["spatial"]:
        raise KeyError("No Visium image found in adata.uns['spatial']")
    library_id = next(iter(adata.uns["spatial"]))
    block = adata.uns["spatial"][library_id]
    image = np.asarray(block["images"]["hires"])
    scale = float(block["scalefactors"]["tissue_hires_scalef"])
    coords = np.asarray(adata.obsm["spatial"], dtype=float)[:, :2] * scale
    return coords, image, block["scalefactors"]


def write_roi_spots(adata: sc.AnnData) -> pd.DataFrame:
    mask = adata.obs[CLUSTER_KEY].isin(ROI_CLUSTERS)
    columns = [c for c in ["spot_name", "x_coord", "y_coord", "array_row", "array_col", CLUSTER_KEY] if c in adata.obs]
    table = adata.obs.loc[mask, columns].copy()
    table.insert(0, "obs_name", table.index.astype(str))
    table["roi_label"] = table[CLUSTER_KEY].map(ROI_LABELS)
    table.to_csv(TABLE_DIR / "roi_spots.csv", index=False)
    return table


def plot_panel_a(adata: sc.AnnData) -> None:
    coords, image, _ = spatial_data(adata)
    labels = adata.obs[CLUSTER_KEY].astype(str).to_numpy()
    polygons = fitted_hexagons(adata, coords)

    fig, ax = plt.subplots(figsize=(10.8, 8.2))
    ax.imshow(image, origin="upper")
    other = ~np.isin(labels, ROI_CLUSTERS)
    ax.add_collection(
        PolyCollection(
            [polygon for polygon, keep in zip(polygons, other) if keep],
            facecolors="#BFC1C7", edgecolors="#FFFFFF", linewidths=0.18,
            alpha=0.28, rasterized=True, zorder=2,
        )
    )
    for cluster in ROI_CLUSTERS:
        mask = labels == cluster
        ax.add_collection(
            PolyCollection(
                [polygon for polygon, keep in zip(polygons, mask) if keep],
                facecolors=ROI_COLORS[cluster], edgecolors="#FFFFFF", linewidths=0.28,
                alpha=0.96, rasterized=True, zorder=3,
            )
        )
    add_roi_boundaries(ax, polygons, labels, linewidth=2.2)

    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=ROI_COLORS[c], markeredgecolor="white", markersize=13, label=ROI_LABELS[c])
        for c in ROI_CLUSTERS
    ]
    legend = ax.legend(
        handles=handles, frameon=False, ncol=2, loc="upper center",
        bbox_to_anchor=(0.5, -0.025), fontsize=18, handletextpad=0.45, columnspacing=2.6,
    )
    for text, cluster in zip(legend.get_texts(), ROI_CLUSTERS):
        text.set_color(ROI_COLORS[cluster])
        text.set_fontweight("bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    save_figure(fig, FIGURE_DIR / "a_spadta_cluster12_cluster8_on_slice")


def build_de_table(adata: sc.AnnData, modality: str) -> pd.DataFrame:
    roi_mask = adata.obs[CLUSTER_KEY].isin(ROI_CLUSTERS).to_numpy()
    modality_mask = adata.var["type"].astype(str).eq(modality).to_numpy()
    subset = adata[roi_mask, modality_mask].copy()
    subset.X = subset.layers[LAYER].copy()
    normalized = to_dense(subset.X).astype(np.float32, copy=False)
    labels = subset.obs[CLUSTER_KEY].astype(str).to_numpy()
    sc.pp.log1p(subset)
    sc.tl.rank_genes_groups(
        subset,
        groupby=CLUSTER_KEY,
        groups=list(ROI_CLUSTERS),
        reference="rest",
        method="wilcoxon",
        use_raw=False,
    )
    result = subset.uns["rank_genes_groups"]
    feature_index = {str(feature): i for i, feature in enumerate(subset.var_names.astype(str))}
    rows: list[pd.DataFrame] = []
    for cluster in ROI_CLUSTERS:
        features = result["names"][cluster].astype(str)
        idx = np.array([feature_index[x] for x in features], dtype=int)
        inside = labels == cluster
        outside = ~inside
        mean_in = normalized[inside][:, idx].mean(axis=0)
        mean_out = normalized[outside][:, idx].mean(axis=0)
        pct_in = (normalized[inside][:, idx] > 0).mean(axis=0)
        pct_out = (normalized[outside][:, idx] > 0).mean(axis=0)
        rows.append(
            pd.DataFrame(
                {
                    "cluster": cluster,
                    "feature": features,
                    "modality": modality,
                    "score": np.asarray(result["scores"][cluster], dtype=float),
                    "pval": np.asarray(result["pvals"][cluster], dtype=float),
                    "pval_adj": np.asarray(result["pvals_adj"][cluster], dtype=float),
                    "mean_in": mean_in,
                    "mean_out": mean_out,
                    "pct_in": pct_in,
                    "pct_out": pct_out,
                    "log2fc": np.log2((mean_in + LOG2FC_EPS) / (mean_out + LOG2FC_EPS)),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def build_st_de_table(adata: sc.AnnData) -> tuple[pd.DataFrame, sc.AnnData]:
    st_mask = adata.var["type"].astype(str).eq("ST").to_numpy()
    st_adata = adata[:, st_mask].copy()
    if ST_COUNTS_LAYER not in st_adata.layers:
        raise KeyError(f"Missing raw ST count layer {ST_COUNTS_LAYER!r}")

    raw_counts = st_adata.layers[ST_COUNTS_LAYER].copy()
    raw_values = raw_counts.data if sparse.issparse(raw_counts) else np.asarray(raw_counts).ravel()
    if np.any(raw_values < 0) or not np.allclose(raw_values, np.rint(raw_values)):
        raise ValueError("The ST count layer is not a non-negative integer count matrix")

    raw_total = np.asarray(raw_counts.sum(axis=1)).ravel()
    if sparse.issparse(raw_counts):
        detected = np.asarray((raw_counts > 0).sum(axis=1)).ravel()
    else:
        detected = (np.asarray(raw_counts) > 0).sum(axis=1)

    st_adata.X = raw_counts.copy()
    sc.pp.normalize_total(st_adata, target_sum=ST_NORMALIZATION_TARGET)
    normalized_linear = st_adata.X.copy()
    normalized_total = np.asarray(normalized_linear.sum(axis=1)).ravel()
    st_adata.layers["normalized_total_1e4"] = normalized_linear.copy()
    sc.pp.log1p(st_adata)

    library_table = pd.DataFrame(
        {
            "obs_name": st_adata.obs_names.astype(str),
            CLUSTER_KEY: st_adata.obs[CLUSTER_KEY].astype(str).to_numpy(),
            "raw_st_total_counts": raw_total,
            "detected_st_genes": detected,
            "normalization_factor": np.divide(
                ST_NORMALIZATION_TARGET,
                raw_total,
                out=np.zeros_like(raw_total, dtype=float),
                where=raw_total > 0,
            ),
            "normalized_st_total": normalized_total,
        }
    )
    library_table = library_table.loc[library_table[CLUSTER_KEY].isin(ROI_CLUSTERS)].copy()
    library_table.to_csv(TABLE_DIR / "st_library_size_and_normalization.csv", index=False)
    library_summary = (
        library_table.groupby(CLUSTER_KEY, observed=True)
        .agg(
            n_spots=("obs_name", "size"),
            raw_total_mean=("raw_st_total_counts", "mean"),
            raw_total_median=("raw_st_total_counts", "median"),
            raw_total_q25=("raw_st_total_counts", lambda x: x.quantile(0.25)),
            raw_total_q75=("raw_st_total_counts", lambda x: x.quantile(0.75)),
            detected_genes_mean=("detected_st_genes", "mean"),
            detected_genes_median=("detected_st_genes", "median"),
        )
        .reset_index()
    )
    library_summary.to_csv(TABLE_DIR / "st_library_size_summary.csv", index=False)

    roi_mask = st_adata.obs[CLUSTER_KEY].isin(ROI_CLUSTERS).to_numpy()
    subset = st_adata[roi_mask].copy()
    subset.obs[CLUSTER_KEY] = pd.Categorical(
        subset.obs[CLUSTER_KEY].astype(str), categories=list(ROI_CLUSTERS)
    )
    sc.tl.rank_genes_groups(
        subset,
        groupby=CLUSTER_KEY,
        groups=["8"],
        reference="12",
        method="wilcoxon",
        use_raw=False,
    )
    result = subset.uns["rank_genes_groups"]
    ranked = pd.DataFrame(
        {
            "feature": np.asarray(result["names"]["8"], dtype=str),
            "score": np.asarray(result["scores"]["8"], dtype=float),
            "pval": np.asarray(result["pvals"]["8"], dtype=float),
            "pval_adj": np.asarray(result["pvals_adj"]["8"], dtype=float),
        }
    ).set_index("feature")

    labels = subset.obs[CLUSTER_KEY].astype(str).to_numpy()
    raw_roi = to_dense(raw_counts[roi_mask]).astype(np.float64, copy=False)
    normalized_roi = to_dense(normalized_linear[roi_mask]).astype(np.float64, copy=False)
    in_8 = labels == "8"
    in_12 = labels == "12"
    features = subset.var_names.astype(str).to_numpy()
    metrics = pd.DataFrame(
        {
            "mean_8": normalized_roi[in_8].mean(axis=0),
            "mean_12": normalized_roi[in_12].mean(axis=0),
            "pct_8": (raw_roi[in_8] > 0).mean(axis=0),
            "pct_12": (raw_roi[in_12] > 0).mean(axis=0),
        },
        index=features,
    )
    table_8 = ranked.join(metrics, how="left").reset_index()
    table_8["cluster"] = "8"
    table_8["modality"] = "ST"
    table_8["mean_in"] = table_8["mean_8"]
    table_8["mean_out"] = table_8["mean_12"]
    table_8["pct_in"] = table_8["pct_8"]
    table_8["pct_out"] = table_8["pct_12"]
    table_8["log2fc"] = np.log2(
        (table_8["mean_8"] + LOG2FC_EPS) / (table_8["mean_12"] + LOG2FC_EPS)
    )

    table_12 = table_8.copy()
    table_12["cluster"] = "12"
    table_12["score"] = -table_12["score"]
    table_12["mean_in"] = table_12["mean_12"]
    table_12["mean_out"] = table_12["mean_8"]
    table_12["pct_in"] = table_12["pct_12"]
    table_12["pct_out"] = table_12["pct_8"]
    table_12["log2fc"] = -table_12["log2fc"]

    columns = [
        "cluster", "feature", "modality", "score", "pval", "pval_adj",
        "mean_in", "mean_out", "pct_in", "pct_out", "log2fc",
    ]
    de_table = pd.concat([table_12[columns], table_8[columns]], ignore_index=True)
    return de_table, st_adata


def significant_features(table: pd.DataFrame, modality: str) -> pd.DataFrame:
    threshold = ST_LOG2FC_THRESHOLD if modality == "ST" else SM_LOG2FC_THRESHOLD
    return table.loc[
        table["modality"].eq(modality)
        & table["pval_adj"].lt(0.05)
        & table["log2fc"].gt(threshold)
        & table["pct_in"].ge(0.10)
    ].copy()


def pairwise_gene_table(st_de: pd.DataFrame) -> pd.DataFrame:
    table = st_de.loc[st_de["cluster"].eq("8")].copy()
    table["comparison"] = "Cluster 8 vs Cluster 12"
    table["neg_log10_padj"] = -np.log10(np.clip(table["pval_adj"], 1e-300, None))
    table["enriched_cluster"] = np.where(table["log2fc"].ge(0), "Cluster 8", "Cluster 12")
    table["significant"] = (
        table["pval_adj"].lt(0.05)
        & table["log2fc"].abs().gt(ST_LOG2FC_THRESHOLD)
        & table[["pct_in", "pct_out"]].max(axis=1).ge(0.10)
    )
    return table.sort_values(["pval_adj", "log2fc"], ascending=[True, False]).reset_index(drop=True)


def plot_panel_b(pairwise: pd.DataFrame, selected: pd.DataFrame) -> None:
    pairwise.to_csv(TABLE_DIR / "cluster12_vs_cluster8_differential_genes_full.csv", index=False)
    deg = pairwise.loc[pairwise["significant"]].copy()
    deg.to_csv(TABLE_DIR / "cluster12_vs_cluster8_differential_genes.csv", index=False)

    fig, ax = plt.subplots(figsize=(9.4, 7.0), layout="constrained")
    nonsignificant = pairwise.loc[~pairwise["significant"]]
    ax.scatter(
        nonsignificant["log2fc"], nonsignificant["neg_log10_padj"],
        s=13, marker="o", color="#B8BBC2", alpha=0.46,
        edgecolors="none", label=f"Not significant (n={len(nonsignificant)})",
        rasterized=True, zorder=1,
    )
    plot_specs = [
        ("12", deg["log2fc"].lt(0), "D", "Cluster 12 enriched"),
        ("8", deg["log2fc"].gt(0), "o", "Cluster 8 enriched"),
    ]
    for cluster, mask, marker, label in plot_specs:
        values = deg.loc[mask]
        ax.scatter(
            values["log2fc"], values["neg_log10_padj"],
            s=28, marker=marker, color=ANALYSIS_COLORS[cluster],
            alpha=0.72, edgecolors="white", linewidths=0.35,
            label=f"{label} (n={len(values)})", rasterized=True, zorder=3,
        )

    ax.axvline(-ST_LOG2FC_THRESHOLD, color="#8A8A8A", linewidth=0.8, linestyle="--")
    ax.axvline(ST_LOG2FC_THRESHOLD, color="#8A8A8A", linewidth=0.8, linestyle="--")
    ax.axhline(-np.log10(0.05), color="#8A8A8A", linewidth=0.8, linestyle=":")

    selected_features = selected["feature"].astype(str).tolist()
    labelled = pairwise.set_index("feature").loc[selected_features].reset_index()
    for index, row in enumerate(labelled.itertuples(index=False)):
        direction = -1 if row.log2fc < 0 else 1
        text_offset = (direction * 8, 7 + (index % 2) * 4)
        ax.annotate(
            str(row.feature),
            xy=(row.log2fc, row.neg_log10_padj),
            xytext=text_offset,
            textcoords="offset points",
            ha="right" if direction < 0 else "left",
            va="bottom",
            fontsize=9,
            arrowprops=dict(arrowstyle="-", color="#606060", lw=0.55),
        )

    x_limit = max(2.2, float(deg["log2fc"].abs().max()) * 1.10)
    ax.set_xlim(-x_limit, x_limit)
    ax.set_ylim(0, float(deg["neg_log10_padj"].max()) * 1.10)
    ax.set_xlabel("log2 fold change (Cluster 8 / Cluster 12)", fontsize=11)
    ax.set_ylabel("-log10 adjusted P", fontsize=11)
    ax.set_title("Differentially expressed genes", fontsize=14, pad=9)
    ax.legend(frameon=False, loc="upper left", fontsize=9, handletextpad=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    save_figure(fig, FIGURE_DIR / "b_cluster12_vs_cluster8_differential_genes")


def select_spatial_deg_features(pairwise: pd.DataFrame) -> pd.DataFrame:
    significant = pairwise.loc[pairwise["significant"]].copy()
    selected = []
    for cluster, mask in [
        ("12", significant["log2fc"].lt(0)),
        ("8", significant["log2fc"].gt(0)),
    ]:
        candidates = significant.loc[mask].nsmallest(TOP_MAP_FEATURES, "pval_adj").copy()
        if cluster == "12" and "Cst3" in set(candidates["feature"]):
            retained = set(candidates.loc[candidates["feature"].ne("Cst3"), "feature"])
            replacement_pool = significant.loc[
                mask
                & significant["feature"].ne("Cst3")
                & ~significant["feature"].isin(retained)
            ].copy()
            replacement_pool["selection_strength"] = (
                replacement_pool["log2fc"].abs()
                * (replacement_pool["pct_in"] - replacement_pool["pct_out"]).abs().add(0.10)
            )
            replacement = replacement_pool.sort_values(
                ["selection_strength", "pval_adj"], ascending=[False, True]
            ).head(1)
            candidates = pd.concat(
                [candidates.loc[candidates["feature"].ne("Cst3")], replacement],
                ignore_index=True,
            )
        candidates["cluster"] = cluster
        candidates["map_rank"] = np.arange(1, len(candidates) + 1)
        selected.append(candidates)
    result = pd.concat(selected, ignore_index=True)
    result.to_csv(TABLE_DIR / "selected_spatial_deg_features.csv", index=False)
    return result


def plot_panel_c(st_adata: sc.AnnData, selected: pd.DataFrame) -> None:
    coords, image, _ = spatial_data(st_adata)
    polygons = fitted_hexagons(st_adata, coords)
    labels = st_adata.obs[CLUSTER_KEY].astype(str).to_numpy()
    roi_mask = np.isin(labels, ROI_CLUSTERS)
    background = gray_outside_roi(image, polygons, roi_mask)
    names = st_adata.var_names.astype(str).to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.4), layout="constrained")
    cmap = EXPRESSION_CMAP

    for row_idx, cluster in enumerate(ROI_CLUSTERS):
        rows = selected.loc[selected["cluster"].eq(cluster)].sort_values("map_rank")
        for col_idx, row in enumerate(rows.itertuples(index=False)):
            ax = axes[row_idx, col_idx]
            feature_idx = int(np.flatnonzero(names == str(row.feature))[0])
            values = to_dense(st_adata.X[:, feature_idx]).astype(float).ravel()
            roi_values = values[roi_mask]
            vmin = 0.0
            vmax = float(np.quantile(roi_values, 0.98))
            if vmax <= vmin:
                vmax = float(roi_values.max() or 1.0)
            norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
            ax.imshow(background, origin="upper", alpha=1.0)
            ax.add_collection(
                PolyCollection(
                    [polygon for polygon, keep in zip(polygons, roi_mask) if keep],
                    facecolors=cmap(norm(roi_values)),
                    edgecolors="#FFFFFF", linewidths=0.18,
                    rasterized=True, zorder=3,
                )
            )
            add_roi_boundaries(ax, polygons, labels, linewidth=1.8)
            ax.set_title(str(row.feature), fontsize=12, fontweight="bold", pad=5)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            scalar_mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
            colorbar = fig.colorbar(scalar_mappable, ax=ax, fraction=0.038, pad=0.012)
            colorbar.set_label("log1p normalized expression", fontsize=8)
            colorbar.ax.tick_params(labelsize=7, length=2)
        axes[row_idx, 0].text(
            -0.045, 0.5, f"{ROI_LABELS[cluster]} enriched", transform=axes[row_idx, 0].transAxes,
            rotation=90, va="center", ha="right", fontsize=11, fontweight="bold", color=ANALYSIS_COLORS[cluster],
        )
    save_figure(fig, FIGURE_DIR / "c_spatial_deg_expression")


def plot_panel_d(st_adata: sc.AnnData, selected: pd.DataFrame) -> None:
    labels = st_adata.obs[CLUSTER_KEY].astype(str).to_numpy()
    roi_mask = np.isin(labels, ROI_CLUSTERS)
    roi_labels = labels[roi_mask]
    names = st_adata.var_names.astype(str).to_numpy()
    selected = selected.sort_values(["cluster", "map_rank"], key=lambda x: x.map({"12": 0, "8": 1}) if x.name == "cluster" else x)

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.4), layout="constrained")
    rng = np.random.default_rng(42)
    expression_rows = []
    for ax, row in zip(axes.flat, selected.itertuples(index=False)):
        feature_idx = int(np.flatnonzero(names == str(row.feature))[0])
        values = to_dense(st_adata.X[:, feature_idx]).astype(float).ravel()[roi_mask]
        grouped = [values[roi_labels == cluster] for cluster in ROI_CLUSTERS]
        box = ax.boxplot(
            grouped, positions=[1, 2], widths=0.58, patch_artist=True,
            showfliers=False, labels=[ROI_LABELS[c] for c in ROI_CLUSTERS],
            medianprops={"color": "#202020", "linewidth": 1.4},
            whiskerprops={"color": "#555555", "linewidth": 1.0},
            capprops={"color": "#555555", "linewidth": 1.0},
        )
        for patch, cluster in zip(box["boxes"], ROI_CLUSTERS):
            patch.set_facecolor(ROI_BOUNDARY_COLORS[cluster])
            patch.set_edgecolor(ROI_BOUNDARY_COLORS[cluster])
            patch.set_alpha(0.55)
        for position, cluster, cluster_values in zip([1, 2], ROI_CLUSTERS, grouped):
            jitter = rng.uniform(-0.16, 0.16, size=cluster_values.size)
            ax.scatter(
                position + jitter, cluster_values, s=8,
                color=ROI_BOUNDARY_COLORS[cluster], alpha=0.42,
                edgecolors="none", rasterized=True, zorder=2,
            )
            expression_rows.extend(
                {"feature": str(row.feature), "cluster": cluster, "expression": float(value)}
                for value in cluster_values
            )
        ax.set_title(str(row.feature), fontsize=12, fontweight="bold", pad=6)
        ax.set_ylabel("log1p normalized expression", fontsize=9)
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#E2E2E2", linewidth=0.6, zorder=0)

    pd.DataFrame(expression_rows).to_csv(TABLE_DIR / "selected_genes_roi_expression.csv", index=False)
    save_figure(fig, FIGURE_DIR / "d_selected_genes_expression_boxplots")


def pairwise_metabolite_table(sm_de: pd.DataFrame) -> pd.DataFrame:
    table = sm_de.loc[sm_de["cluster"].eq("8")].copy()
    table["comparison"] = "Cluster 8 vs Cluster 12"
    table["neg_log10_padj"] = -np.log10(np.clip(table["pval_adj"], 1e-300, None))
    table["enriched_cluster"] = np.where(table["log2fc"].ge(0), "Cluster 8", "Cluster 12")
    table["significant"] = (
        table["pval_adj"].lt(0.05)
        & table["log2fc"].abs().gt(SM_LOG2FC_THRESHOLD)
        & table[["pct_in", "pct_out"]].max(axis=1).ge(0.10)
    )
    return table.sort_values(["pval_adj", "log2fc"], ascending=[True, False]).reset_index(drop=True)


def select_spatial_metabolites(pairwise: pd.DataFrame) -> pd.DataFrame:
    significant = pairwise.loc[pairwise["significant"]].copy()
    selected = []
    for cluster, mask in [
        ("12", significant["log2fc"].lt(0)),
        ("8", significant["log2fc"].gt(0)),
    ]:
        candidates = significant.loc[mask].nsmallest(TOP_MAP_FEATURES, "pval_adj").copy()
        if len(candidates) < TOP_MAP_FEATURES:
            raise ValueError(f"Fewer than {TOP_MAP_FEATURES} significant metabolites for Cluster {cluster}")
        candidates["cluster"] = cluster
        candidates["map_rank"] = np.arange(1, len(candidates) + 1)
        selected.append(candidates)
    result = pd.concat(selected, ignore_index=True)
    result.to_csv(TABLE_DIR / "selected_spatial_metabolites.csv", index=False)
    return result


def mz_label(feature: object) -> str:
    try:
        return f"m/z {float(feature):.4f}"
    except (TypeError, ValueError):
        return str(feature)


def plot_panel_e(pairwise: pd.DataFrame, selected: pd.DataFrame) -> None:
    pairwise.to_csv(TABLE_DIR / "cluster12_vs_cluster8_differential_metabolites_full.csv", index=False)
    differential = pairwise.loc[pairwise["significant"]].copy()
    differential.to_csv(TABLE_DIR / "cluster12_vs_cluster8_differential_metabolites.csv", index=False)

    fig, ax = plt.subplots(figsize=(9.4, 7.0), layout="constrained")
    nonsignificant = pairwise.loc[~pairwise["significant"]]
    ax.scatter(
        nonsignificant["log2fc"], nonsignificant["neg_log10_padj"],
        s=13, marker="o", color="#B8BBC2", alpha=0.46,
        edgecolors="none", label=f"Not significant (n={len(nonsignificant)})",
        rasterized=True, zorder=1,
    )
    for cluster, mask, marker, label in [
        ("12", differential["log2fc"].lt(0), "D", "Cluster 12 enriched"),
        ("8", differential["log2fc"].gt(0), "o", "Cluster 8 enriched"),
    ]:
        values = differential.loc[mask]
        ax.scatter(
            values["log2fc"], values["neg_log10_padj"],
            s=28, marker=marker, color=ANALYSIS_COLORS[cluster],
            alpha=0.72, edgecolors="white", linewidths=0.35,
            label=f"{label} (n={len(values)})", rasterized=True, zorder=3,
        )

    ax.axvline(-SM_LOG2FC_THRESHOLD, color="#8A8A8A", linewidth=0.8, linestyle="--")
    ax.axvline(SM_LOG2FC_THRESHOLD, color="#8A8A8A", linewidth=0.8, linestyle="--")
    ax.axhline(-np.log10(0.05), color="#8A8A8A", linewidth=0.8, linestyle=":")
    labelled = pairwise.set_index("feature").loc[selected["feature"].astype(str)].reset_index()
    direction_counts = {"negative": 0, "positive": 0}
    label_offsets = {
        "negative": [(14, 20), (14, -24)],
        "positive": [(12, 18), (12, 38)],
    }
    for row in labelled.itertuples(index=False):
        direction = "negative" if row.log2fc < 0 else "positive"
        text_offset = label_offsets[direction][direction_counts[direction]]
        direction_counts[direction] += 1
        ax.annotate(
            mz_label(row.feature), xy=(row.log2fc, row.neg_log10_padj),
            xytext=text_offset, textcoords="offset points",
            ha="left", va="bottom" if text_offset[1] >= 0 else "top", fontsize=9,
            arrowprops=dict(arrowstyle="-", color="#606060", lw=0.55),
        )

    x_limit = max(2.2, float(differential["log2fc"].abs().max()) * 1.10)
    ax.set_xlim(-x_limit, x_limit)
    ax.set_ylim(0, float(differential["neg_log10_padj"].max()) * 1.10)
    ax.set_xlabel("log2 fold change (Cluster 8 / Cluster 12)", fontsize=11)
    ax.set_ylabel("-log10 adjusted P", fontsize=11)
    ax.set_title("Differential metabolites", fontsize=14, pad=9)
    ax.legend(frameon=False, loc="upper right", fontsize=9, handletextpad=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    save_figure(fig, FIGURE_DIR / "e_cluster12_vs_cluster8_differential_metabolites")


def prepare_sm_adata(adata: sc.AnnData) -> sc.AnnData:
    sm_mask = adata.var["type"].astype(str).eq("SM").to_numpy()
    sm_adata = adata[:, sm_mask].copy()
    sm_adata.X = sm_adata.layers[LAYER].copy()
    sc.pp.log1p(sm_adata)
    return sm_adata


def plot_panel_f(sm_adata: sc.AnnData, selected: pd.DataFrame) -> None:
    coords, image, _ = spatial_data(sm_adata)
    polygons = fitted_hexagons(sm_adata, coords)
    labels = sm_adata.obs[CLUSTER_KEY].astype(str).to_numpy()
    roi_mask = np.isin(labels, ROI_CLUSTERS)
    background = gray_outside_roi(image, polygons, roi_mask)
    names = sm_adata.var_names.astype(str).to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.4), layout="constrained")

    for row_idx, cluster in enumerate(ROI_CLUSTERS):
        rows = selected.loc[selected["cluster"].eq(cluster)].sort_values("map_rank")
        for col_idx, row in enumerate(rows.itertuples(index=False)):
            ax = axes[row_idx, col_idx]
            feature_idx = int(np.flatnonzero(names == str(row.feature))[0])
            values = to_dense(sm_adata.X[:, feature_idx]).astype(float).ravel()
            roi_values = values[roi_mask]
            vmin = 0.0
            vmax = float(np.quantile(roi_values, 0.98))
            if vmax <= vmin:
                vmax = float(roi_values.max() or 1.0)
            norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
            ax.imshow(background, origin="upper", alpha=1.0)
            ax.add_collection(
                PolyCollection(
                    [polygon for polygon, keep in zip(polygons, roi_mask) if keep],
                    facecolors=EXPRESSION_CMAP(norm(roi_values)),
                    edgecolors="#FFFFFF", linewidths=0.18,
                    rasterized=True, zorder=3,
                )
            )
            add_roi_boundaries(ax, polygons, labels, linewidth=1.8)
            ax.set_title(mz_label(row.feature), fontsize=12, fontweight="bold", pad=5)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            scalar_mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=EXPRESSION_CMAP)
            colorbar = fig.colorbar(scalar_mappable, ax=ax, fraction=0.038, pad=0.012)
            colorbar.set_label("log1p normalized abundance", fontsize=8)
            colorbar.ax.tick_params(labelsize=7, length=2)
        axes[row_idx, 0].text(
            -0.045, 0.5, f"{ROI_LABELS[cluster]} enriched", transform=axes[row_idx, 0].transAxes,
            rotation=90, va="center", ha="right", fontsize=11,
            fontweight="bold", color=ANALYSIS_COLORS[cluster],
        )
    save_figure(fig, FIGURE_DIR / "f_spatial_metabolite_abundance")


def plot_panel_g(sm_adata: sc.AnnData, selected: pd.DataFrame) -> None:
    labels = sm_adata.obs[CLUSTER_KEY].astype(str).to_numpy()
    roi_mask = np.isin(labels, ROI_CLUSTERS)
    roi_labels = labels[roi_mask]
    names = sm_adata.var_names.astype(str).to_numpy()
    selected = selected.sort_values(
        ["cluster", "map_rank"],
        key=lambda x: x.map({"12": 0, "8": 1}) if x.name == "cluster" else x,
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.4), layout="constrained")
    rng = np.random.default_rng(42)
    abundance_rows = []
    for ax, row in zip(axes.flat, selected.itertuples(index=False)):
        feature_idx = int(np.flatnonzero(names == str(row.feature))[0])
        values = to_dense(sm_adata.X[:, feature_idx]).astype(float).ravel()[roi_mask]
        grouped = [values[roi_labels == cluster] for cluster in ROI_CLUSTERS]
        box = ax.boxplot(
            grouped, positions=[1, 2], widths=0.58, patch_artist=True,
            showfliers=False, labels=[ROI_LABELS[c] for c in ROI_CLUSTERS],
            medianprops={"color": "#202020", "linewidth": 1.4},
            whiskerprops={"color": "#555555", "linewidth": 1.0},
            capprops={"color": "#555555", "linewidth": 1.0},
        )
        for patch, cluster in zip(box["boxes"], ROI_CLUSTERS):
            patch.set_facecolor(ROI_BOUNDARY_COLORS[cluster])
            patch.set_edgecolor(ROI_BOUNDARY_COLORS[cluster])
            patch.set_alpha(0.55)
        for position, cluster, cluster_values in zip([1, 2], ROI_CLUSTERS, grouped):
            jitter = rng.uniform(-0.16, 0.16, size=cluster_values.size)
            ax.scatter(
                position + jitter, cluster_values, s=8,
                color=ROI_BOUNDARY_COLORS[cluster], alpha=0.42,
                edgecolors="none", rasterized=True, zorder=2,
            )
            abundance_rows.extend(
                {"feature": str(row.feature), "cluster": cluster, "abundance": float(value)}
                for value in cluster_values
            )
        ax.set_title(mz_label(row.feature), fontsize=12, fontweight="bold", pad=6)
        ax.set_ylabel("log1p normalized abundance", fontsize=9)
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#E2E2E2", linewidth=0.6, zorder=0)

    pd.DataFrame(abundance_rows).to_csv(TABLE_DIR / "selected_metabolites_roi_abundance.csv", index=False)
    save_figure(fig, FIGURE_DIR / "g_selected_metabolites_abundance_boxplots")


def prepare_hmdb() -> pd.DataFrame:
    hmdb = pd.read_csv(HMDB_PATH).copy()
    hmdb["monisotopic_molecular_weight"] = hmdb["monisotopic_molecular_weight"].astype(float)
    for method, adduct in ADDUCTS:
        mass = Formula(adduct).monoisotopic_mass
        column = f"mz_{method}_{adduct}"
        hmdb[column] = (
            hmdb["monisotopic_molecular_weight"] + mass
            if method == "add"
            else hmdb["monisotopic_molecular_weight"] - mass
        )
    return hmdb


def annotate_mz_best_hit(mz: float, hmdb: pd.DataFrame) -> dict[str, object] | None:
    best = None
    for method, adduct in ADDUCTS:
        column = f"mz_{method}_{adduct}"
        errors = np.abs(hmdb[column].astype(float) - float(mz)) / float(mz) * 1e6
        index = errors.idxmin()
        error = float(errors.loc[index])
        if error > PPM_TOLERANCE:
            continue
        row = hmdb.loc[index]
        candidate = {
            "feature_mz": float(mz),
            "ppm_error": error,
            "adduct_mode": f"{method}_{adduct}",
            "accession": row.get("accession"),
            "metabolite_name": row.get("name"),
            "direct_parent": row.get("direct_parent"),
            "class": row.get("class"),
            "sub_class": row.get("sub_class"),
        }
        if best is None or error < float(best["ppm_error"]):
            best = candidate
    return best


def query_mouse_go_bp(genes: list[str]) -> pd.DataFrame:
    response = requests.post(
        "https://biit.cs.ut.ee/gprofiler/api/gost/profile/",
        json={
            "organism": "mmusculus",
            "query": genes,
            "sources": ["GO:BP"],
            "user_threshold": 1.0,
            "no_iea": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    query_size = max(len(set(genes)), 1)
    rows = []
    for item in response.json().get("result", []):
        if str(item.get("source")) != "GO:BP":
            continue
        adjusted_p = float(item.get("p_value", 1.0))
        overlap = int(item.get("intersection_size", 0))
        query_pct = 100.0 * overlap / query_size
        rows.append(
            {
                "term": str(item.get("name")),
                "go_id": str(item.get("native")),
                "adjusted_p_value": adjusted_p,
                "overlap_count": overlap,
                "query_gene_pct": query_pct,
                "combined_score": query_pct * -np.log10(max(adjusted_p, 1e-300)),
                "log10_1_over_fdr": -np.log10(max(adjusted_p, 1e-300)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["adjusted_p_value", "combined_score"], ascending=[True, False]
    ).reset_index(drop=True)


def metabolite_subclass_enrichment(query: pd.DataFrame, background: pd.DataFrame) -> pd.DataFrame:
    query_unique = query[["accession", "sub_class"]].dropna().drop_duplicates()
    background_unique = background[["accession", "sub_class"]].dropna().drop_duplicates()
    query_ids = set(query_unique["accession"].astype(str))
    background_size = background_unique["accession"].astype(str).nunique()
    rows = []
    for group, frame in background_unique.groupby("sub_class", observed=True):
        group_ids = set(frame["accession"].astype(str))
        overlap_ids = sorted(query_ids.intersection(group_ids))
        if not overlap_ids:
            continue
        p_value = float(
            hypergeom.sf(
                len(overlap_ids) - 1,
                background_size,
                len(group_ids),
                len(query_ids),
            )
        )
        rows.append(
            {
                "group": str(group),
                "p_value": p_value,
                "overlap": len(overlap_ids),
                "background_count": len(group_ids),
                "accessions": ";".join(overlap_ids),
                "log10_1_over_p": -np.log10(max(p_value, 1e-300)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["p_value", "overlap"], ascending=[True, False]
    ).reset_index(drop=True)


def build_panel_h_enrichment(st_de: pd.DataFrame, sm_de: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    hmdb = prepare_hmdb()
    background_rows = []
    for mz in sorted(sm_de["feature"].astype(float).dropna().unique()):
        annotation = annotate_mz_best_hit(float(mz), hmdb)
        if annotation is not None:
            background_rows.append(annotation)
    annotated_background = pd.DataFrame(background_rows)
    results = {}

    for cluster in ROI_CLUSTERS:
        genes = st_de.loc[
            st_de["cluster"].eq(cluster)
            & st_de["log2fc"].gt(ENRICHMENT_LOG2FC_THRESHOLD)
        ].sort_values(["log2fc", "score"], ascending=[False, False]).copy()
        metabolites = sm_de.loc[
            sm_de["cluster"].eq(cluster)
            & sm_de["log2fc"].gt(ENRICHMENT_LOG2FC_THRESHOLD)
        ].sort_values(["log2fc", "score"], ascending=[False, False]).copy()
        genes.to_csv(TABLE_DIR / f"cluster{cluster}_h_upregulated_genes.csv", index=False)
        metabolites.to_csv(TABLE_DIR / f"cluster{cluster}_h_upregulated_metabolites.csv", index=False)

        annotations = []
        for row in metabolites.itertuples(index=False):
            annotation = annotate_mz_best_hit(float(row.feature), hmdb)
            if annotation is None:
                continue
            annotation.update(
                {
                    "cluster": cluster,
                    "feature": float(row.feature),
                    "log2fc": float(row.log2fc),
                    "pval_adj": float(row.pval_adj),
                    "score": float(row.score),
                }
            )
            annotations.append(annotation)
        annotated_query = pd.DataFrame(annotations)
        annotated_query.to_csv(TABLE_DIR / f"cluster{cluster}_h_metabolite_annotations.csv", index=False)

        go = query_mouse_go_bp(genes["feature"].astype(str).drop_duplicates().tolist())
        metabolite_enrichment = metabolite_subclass_enrichment(annotated_query, annotated_background)
        go.to_csv(TABLE_DIR / f"cluster{cluster}_h_go_bp_enrichment.csv", index=False)
        metabolite_enrichment.to_csv(
            TABLE_DIR / f"cluster{cluster}_h_metabolite_subclass_enrichment.csv", index=False
        )
        results[cluster] = (go, metabolite_enrichment)
    return results


def plot_panel_h(results: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> None:
    fig, axes = plt.subplots(
        2, 2, figsize=(11.8, 8.0),
        gridspec_kw={"width_ratios": [1.08, 0.92], "hspace": 0.62, "wspace": 0.62},
    )
    for row_index, cluster in enumerate(ROI_CLUSTERS):
        go, metabolites = results[cluster]
        go_plot = go.head(5).sort_values("combined_score").reset_index(drop=True)
        metabolite_plot = metabolites.head(4).sort_values("overlap").reset_index(drop=True)
        ax_go, ax_metabolite = axes[row_index]

        go_norm = Normalize(
            vmin=float(go_plot["log10_1_over_fdr"].min()),
            vmax=float(go_plot["log10_1_over_fdr"].max()),
        )
        scatter = ax_go.scatter(
            go_plot["combined_score"], np.arange(len(go_plot)),
            s=np.clip(go_plot["query_gene_pct"].astype(float) * 30.0, 80, 280),
            c=go_plot["log10_1_over_fdr"], cmap=PURPLE_CMAP, norm=go_norm,
            edgecolors="none", zorder=3,
        )
        left_edge = max(0.0, float(go_plot["combined_score"].min()) - 4.0)
        for y_value, x_value in zip(np.arange(len(go_plot)), go_plot["combined_score"].astype(float)):
            ax_go.hlines(y=y_value, xmin=left_edge, xmax=x_value, color="#dddddd", lw=1.1, zorder=1)
        ax_go.set_yticks(np.arange(len(go_plot)))
        ax_go.set_yticklabels([textwrap.fill(str(value), 22) for value in go_plot["term"]], fontsize=8)
        ax_go.set_xlabel("Combined score", fontsize=9)
        ax_go.set_xlim(left=left_edge)
        ax_go.set_title(f"{ROI_LABELS[cluster]}: GO biological process", fontsize=10, fontweight="bold")
        ax_go.tick_params(axis="x", labelsize=8)
        ax_go.tick_params(axis="y", length=0)
        ax_go.spines[["top", "right"]].set_visible(False)
        colorbar_go = fig.colorbar(scatter, ax=ax_go, fraction=0.052, pad=0.04)
        colorbar_go.ax.set_title("log10(1/FDR)", fontsize=7, pad=5)
        colorbar_go.ax.tick_params(labelsize=7)

        metabolite_norm = Normalize(
            vmin=float(metabolite_plot["log10_1_over_p"].min()),
            vmax=float(metabolite_plot["log10_1_over_p"].max()),
        )
        ax_metabolite.barh(
            np.arange(len(metabolite_plot)), metabolite_plot["overlap"].astype(float),
            color=PURPLE_CMAP(metabolite_norm(metabolite_plot["log10_1_over_p"].astype(float))),
            edgecolor="none", height=0.56,
        )
        ax_metabolite.set_yticks(np.arange(len(metabolite_plot)))
        ax_metabolite.set_yticklabels(
            [textwrap.fill(str(value), 18) for value in metabolite_plot["group"]], fontsize=8
        )
        ax_metabolite.set_xlabel("Metabolites in set", fontsize=9)
        ax_metabolite.set_title(f"{ROI_LABELS[cluster]}: metabolite subclasses", fontsize=10, fontweight="bold")
        ax_metabolite.tick_params(axis="x", labelsize=8)
        ax_metabolite.tick_params(axis="y", length=0)
        ax_metabolite.spines[["top", "right"]].set_visible(False)
        ax_metabolite.set_xlim(0, max(1.0, float(metabolite_plot["overlap"].max()) + 1.5))
        scalar_mappable = matplotlib.cm.ScalarMappable(norm=metabolite_norm, cmap=PURPLE_CMAP)
        colorbar_metabolite = fig.colorbar(scalar_mappable, ax=ax_metabolite, fraction=0.052, pad=0.04)
        colorbar_metabolite.ax.set_title("log10(1/p)", fontsize=7, pad=5)
        colorbar_metabolite.ax.tick_params(labelsize=7)

    save_figure(fig, FIGURE_DIR / "h_cluster12_cluster8_multiomics_enrichment")


def make_composite() -> None:
    paths = [
        FIGURE_DIR / "a_spadta_cluster12_cluster8_on_slice.png",
        FIGURE_DIR / "b_cluster12_vs_cluster8_differential_genes.png",
        FIGURE_DIR / "c_spatial_deg_expression.png",
        FIGURE_DIR / "d_selected_genes_expression_boxplots.png",
        FIGURE_DIR / "e_cluster12_vs_cluster8_differential_metabolites.png",
        FIGURE_DIR / "f_spatial_metabolite_abundance.png",
        FIGURE_DIR / "g_selected_metabolites_abundance_boxplots.png",
        FIGURE_DIR / "h_cluster12_cluster8_multiomics_enrichment.png",
    ]
    images = [Image.open(path).convert("RGB") for path in paths]
    target_width = 2400
    resized = []
    for image in images:
        height = int(round(image.height * target_width / image.width))
        resized.append(image.resize((target_width, height), Image.Resampling.LANCZOS))
    gap = 70
    label_space = 55
    row_heights = [
        max(resized[0].height, resized[1].height),
        max(resized[2].height, resized[3].height),
        max(resized[4].height, resized[5].height),
        max(resized[6].height, resized[7].height),
    ]
    row_y = [label_space]
    for height in row_heights[:-1]:
        row_y.append(row_y[-1] + height + gap + label_space)
    canvas_height = row_y[-1] + row_heights[-1]
    canvas = Image.new("RGB", (target_width * 2 + gap, canvas_height), "white")
    positions = [
        (0, row_y[0]),
        (target_width + gap, row_y[0]),
        (0, row_y[1]),
        (target_width + gap, row_y[1]),
        (0, row_y[2]),
        (target_width + gap, row_y[2]),
        (0, row_y[3]),
        (target_width + gap, row_y[3]),
    ]
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 48)
    except OSError:
        font = ImageFont.load_default()
    for label, image, (x, y) in zip("abcdefgh", resized, positions):
        canvas.paste(image, (x, y))
        draw.text((x + 8, y - label_space + 2), label, fill="black", font=font)
    png = FIGURE_DIR / "sup_fig3.png"
    canvas.save(png, dpi=(600, 600))


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(OUTPUT_ROOT / "run.log", mode="w"), logging.StreamHandler()],
    )
    logging.info("Loading %s", INPUT_H5AD)
    adata = load_data()
    roi_table = write_roi_spots(adata)
    logging.info("ROI counts: %s", roi_table[CLUSTER_KEY].value_counts().to_dict())

    plot_panel_a(adata)
    st_de, st_adata = build_st_de_table(adata)
    st_de.to_csv(TABLE_DIR / "cluster12_vs_cluster8_st_de_full.csv", index=False)
    pairwise = pairwise_gene_table(st_de)
    selected = select_spatial_deg_features(pairwise)
    plot_panel_b(pairwise, selected)
    plot_panel_c(st_adata, selected)
    plot_panel_d(st_adata, selected)
    sm_de = build_de_table(adata, "SM")
    sm_de.to_csv(TABLE_DIR / "cluster12_vs_cluster8_sm_de_full.csv", index=False)
    sm_pairwise = pairwise_metabolite_table(sm_de)
    selected_sm = select_spatial_metabolites(sm_pairwise)
    sm_adata = prepare_sm_adata(adata)
    plot_panel_e(sm_pairwise, selected_sm)
    plot_panel_f(sm_adata, selected_sm)
    plot_panel_g(sm_adata, selected_sm)
    enrichment = build_panel_h_enrichment(st_de, sm_de)
    plot_panel_h(enrichment)
    make_composite()

    summary = {
        "sample": SAMPLE,
        "input_h5ad": str(INPUT_H5AD),
        "cluster_key": CLUSTER_KEY,
        "roi_clusters": list(ROI_CLUSTERS),
        "roi_spot_counts": roi_table[CLUSTER_KEY].value_counts().sort_index().astype(int).to_dict(),
        "st_count_layer": ST_COUNTS_LAYER,
        "st_normalization": {
            "method": "per-spot library-size normalization on all 2000 retained ST genes, followed by log1p",
            "target_sum": ST_NORMALIZATION_TARGET,
            "de_test": "two-sided Wilcoxon rank-sum, Cluster 8 versus Cluster 12, BH adjusted",
            "log2fc_scale": "mean normalized counts before log1p",
            "detection_scale": "raw counts > 0",
        },
        "comparison": "cluster 12 versus cluster 8 in the stored m3_FMP SpaDTA spatial-domain figure",
        "cluster_label_source": str(REFERENCE_CLUSTER_IMAGE),
        "st_log2fc_threshold": ST_LOG2FC_THRESHOLD,
        "sm_log2fc_threshold": SM_LOG2FC_THRESHOLD,
        "figure_format": "high-resolution PNG only",
        "figures": {path.stem: {"png": str(path.with_suffix(".png"))} for path in [
            FIGURE_DIR / "a_spadta_cluster12_cluster8_on_slice",
            FIGURE_DIR / "b_cluster12_vs_cluster8_differential_genes",
            FIGURE_DIR / "c_spatial_deg_expression",
            FIGURE_DIR / "d_selected_genes_expression_boxplots",
            FIGURE_DIR / "e_cluster12_vs_cluster8_differential_metabolites",
            FIGURE_DIR / "f_spatial_metabolite_abundance",
            FIGURE_DIR / "g_selected_metabolites_abundance_boxplots",
            FIGURE_DIR / "h_cluster12_cluster8_multiomics_enrichment",
            FIGURE_DIR / "sup_fig3",
        ]},
    }
    (OUTPUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logging.info("Finished. Composite: %s", FIGURE_DIR / "sup_fig3.png")


if __name__ == "__main__":
    main()
