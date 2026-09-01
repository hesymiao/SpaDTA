from __future__ import annotations

import argparse
from pathlib import Path
import re

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy import sparse


PROJECT_ROOT = Path("/data/user/hesy/projects/SpatialMETA")
DATA_DIR = Path(
    "/bigdat2/user/hesy/spatialmeta/SpatialMETA/smart/SMART_data/Mouse_Brain_E18_S1"
)
SPADTA_DIR = PROJECT_ROOT / "SpaDTA_718/runs/ATAC/Mouse_Brain_E18_S1"
OUTPUT_DIR = PROJECT_ROOT / "SpaDTA_718/runs/atac_downstream/fig4ij"

MIN_REGION_SPOTS = 30
MARKERS_PER_REGION = 2

REGION_COLORS = [
    "#4E79A7", "#E15759", "#59A14F", "#F28E2B", "#B07AA1",
    "#76B7B2", "#EDC948", "#FF9DA7", "#9C755F", "#7F7F7F",
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9",
]


def natural_key(value: str) -> tuple[object, ...]:
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value))


def load_annotation() -> tuple[pd.DataFrame, list[tuple[str, str]], dict[str, str]]:
    annotation = pd.read_csv(DATA_DIR / "anno.csv", dtype=str).set_index("barcode")
    pairs = (
        annotation[["0", "cluster"]]
        .drop_duplicates()
        .sort_values("0", key=lambda values: values.map(natural_key))
    )
    regions = list(pairs.itertuples(index=False, name=None))
    colors = {
        region_name: REGION_COLORS[index]
        for index, (_, region_name) in enumerate(regions)
    }
    return annotation, regions, colors


def load_spadta_labels() -> pd.Series:
    spot_ids = pd.read_csv(
        SPADTA_DIR / "saved_epoch_embeddings/epoch_0300/spot_ids.csv",
        dtype=str,
    )["spot_id"]
    labels = pd.read_csv(
        SPADTA_DIR / "final_protocol/epoch_0300/spot_labels.csv",
        dtype={"mclust_label": str},
    )["mclust_label"]
    if len(spot_ids) != len(labels):
        raise ValueError("SpaDTA spot IDs and cluster labels have different lengths")
    return pd.Series(labels.to_numpy(), index=spot_ids.to_numpy(), name="spadta_cluster")


def normalize_log1p(matrix: sparse.spmatrix, target_sum: float = 1e4) -> sparse.csr_matrix:
    matrix = matrix.tocsr().astype(np.float32)
    library_size = np.asarray(matrix.sum(axis=1)).ravel()
    scale = np.divide(
        target_sum,
        library_size,
        out=np.zeros_like(library_size, dtype=np.float32),
        where=library_size > 0,
    )
    matrix = sparse.diags(scale, format="csr") @ matrix
    matrix.data = np.log1p(matrix.data)
    return matrix.tocsr()


def load_modality(
    filename: str,
    common_spots: pd.Index,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    adata = ad.read_h5ad(DATA_DIR / filename)
    adata.obs_names = adata.obs_names.astype(str)
    positions = adata.obs_names.get_indexer(common_spots)
    if (positions < 0).any():
        raise ValueError(f"{filename} is missing aligned spots")
    matrix = normalize_log1p(adata.X[positions])
    features = adata.var_names.astype(str).to_numpy()
    return matrix, features


def load_selected_modality(
    filename: str,
    common_spots: pd.Index,
    selected_features: list[str],
    chunk_size: int = 256,
) -> sparse.csr_matrix:
    """Load only selected columns while using full-library sizes for normalization."""
    adata = ad.read_h5ad(DATA_DIR / filename, backed="r")
    try:
        adata.obs_names = adata.obs_names.astype(str)
        positions = adata.obs_names.get_indexer(common_spots)
        if (positions < 0).any():
            raise ValueError(f"{filename} is missing aligned spots")
        all_features = pd.Index(adata.var_names.astype(str))
        first_occurrence = ~all_features.duplicated(keep="first")
        unique_features = all_features[first_occurrence]
        unique_positions = np.flatnonzero(first_occurrence)
        selected_in_unique = unique_features.get_indexer(selected_features)
        feature_positions = np.where(
            selected_in_unique >= 0,
            unique_positions[np.maximum(selected_in_unique, 0)],
            -1,
        )
        if (feature_positions < 0).any():
            missing = np.asarray(selected_features)[feature_positions < 0].tolist()
            raise KeyError(f"{filename} is missing selected features: {missing}")

        normalized_chunks = []
        for start in range(0, len(positions), chunk_size):
            block = adata.X[positions[start:start + chunk_size], :].tocsr().astype(np.float32)
            library_size = np.asarray(block.sum(axis=1)).ravel()
            selected = block[:, feature_positions].tocsr()
            scale = np.divide(
                1e4,
                library_size,
                out=np.zeros_like(library_size, dtype=np.float32),
                where=library_size > 0,
            )
            selected = sparse.diags(scale, format="csr") @ selected
            selected.data = np.log1p(selected.data)
            normalized_chunks.append(selected)
        return sparse.vstack(normalized_chunks, format="csr")
    finally:
        adata.file.close()


def eligible_regions(
    annotation: pd.DataFrame,
    common_spots: pd.Index,
    regions: list[tuple[str, str]],
) -> tuple[list[tuple[str, str]], pd.DataFrame]:
    truth = annotation.loc[common_spots, "cluster"].astype(str)
    counts = truth.value_counts()
    audit = pd.DataFrame(
        [
            {
                "region_id": region_id,
                "region_name": region_name,
                "spots": int(counts.get(region_name, 0)),
                "used_for_marker_heatmap": int(counts.get(region_name, 0)) >= MIN_REGION_SPOTS,
            }
            for region_id, region_name in regions
        ]
    )
    selected = [
        (region_id, region_name)
        for region_id, region_name in regions
        if int(counts.get(region_name, 0)) >= MIN_REGION_SPOTS
    ]
    return selected, audit


def select_region_markers(
    matrix: sparse.csr_matrix,
    features: np.ndarray,
    truth: np.ndarray,
    regions: list[tuple[str, str]],
    modality: str,
) -> tuple[list[int], list[dict[str, object]], list[tuple[str, int, int]]]:
    if modality == "RNA":
        duplicate = pd.Index(features).duplicated(keep="first")
        excluded_name = np.array([
            bool(re.match(r"^(mt-|Rpl\d|Rps\d)", feature, flags=re.IGNORECASE))
            for feature in features
        ])
        eligible_feature = ~(duplicate | excluded_name)
    else:
        eligible_feature = np.ones(len(features), dtype=bool)

    total_nonzero = np.asarray(matrix.getnnz(axis=0)).ravel()
    eligible_feature &= total_nonzero >= 10
    selected_indices: list[int] = []
    records: list[dict[str, object]] = []
    group_ranges: list[tuple[str, int, int]] = []
    used_features: set[str] = set()

    for _, region_name in regions:
        in_group = truth == region_name
        out_group = ~in_group
        mean_in = np.asarray(matrix[in_group].mean(axis=0)).ravel()
        mean_out = np.asarray(matrix[out_group].mean(axis=0)).ravel()
        score = mean_in - mean_out
        detected_in = np.asarray(matrix[in_group].getnnz(axis=0)).ravel()
        minimum_detected = max(3, int(np.ceil(in_group.sum() * 0.05)))
        valid = eligible_feature & (detected_in >= minimum_detected) & (score > 0)
        ranked = np.flatnonzero(valid)
        ranked = ranked[np.argsort(score[ranked])[::-1]]

        start = len(selected_indices)
        for feature_index in ranked:
            feature = str(features[feature_index])
            if feature in used_features:
                continue
            used_features.add(feature)
            selected_indices.append(int(feature_index))
            records.append({
                "modality": modality,
                "region_name": region_name,
                "feature": feature,
                "mean_in": float(mean_in[feature_index]),
                "mean_out": float(mean_out[feature_index]),
                "mean_difference": float(score[feature_index]),
                "detected_spots_in_region": int(detected_in[feature_index]),
                "region_spots": int(in_group.sum()),
            })
            if len(selected_indices) - start == MARKERS_PER_REGION:
                break
        if len(selected_indices) - start != MARKERS_PER_REGION:
            raise RuntimeError(
                f"Could not select {MARKERS_PER_REGION} {modality} markers for {region_name}"
            )
        group_ranges.append((region_name, start, len(selected_indices) - 1))
    return selected_indices, records, group_ranges


def aggregate_by_cluster(
    matrix: sparse.csr_matrix,
    feature_indices: list[int],
    clusters: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    cluster_order = sorted(pd.unique(clusters).astype(str), key=natural_key)
    selected = matrix[:, feature_indices]
    means = np.vstack([
        np.asarray(selected[clusters == cluster].mean(axis=0)).ravel()
        for cluster in cluster_order
    ])
    center = means.mean(axis=0, keepdims=True)
    scale = means.std(axis=0, keepdims=True)
    zscore = np.divide(means - center, scale, out=np.zeros_like(means), where=scale > 0)
    return np.clip(zscore, -1.0, 1.0), cluster_order


def wrapped_region_name(name: str) -> str:
    if len(name) <= 16:
        return name
    words = name.replace("and ", "and\n").split(" ")
    if len(words) > 2 and "\n" not in name:
        midpoint = len(words) // 2
        return " ".join(words[:midpoint]) + "\n" + " ".join(words[midpoint:])
    return " ".join(words)


def plot_heatmap(
    values: np.ndarray,
    cluster_order: list[str],
    feature_names: list[str],
    group_ranges: list[tuple[str, int, int]],
    region_colors: dict[str, str],
    modality: str,
    output_stem: Path,
) -> None:
    if modality == "RNA":
        cmap = LinearSegmentedColormap.from_list("rna", ["#F7FBFF", "#6BAED6", "#08306B"])
        colorbar_label = "Mean expression in cluster"
    else:
        cmap = LinearSegmentedColormap.from_list("atac", ["#F7FCF5", "#74C476", "#00441B"])
        colorbar_label = "Mean accessibility in cluster"

    fig_width = max(16.5, len(feature_names) * 0.72)
    fig, ax = plt.subplots(figsize=(fig_width, 9.2), dpi=220)
    image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
    ax.set_yticks(np.arange(len(cluster_order)))
    ax.set_yticklabels(cluster_order, fontsize=14)
    ax.set_ylabel("SpaDTA cluster", fontsize=17)
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_xticklabels(
        feature_names,
        rotation=55,
        ha="right",
        va="top",
        rotation_mode="anchor",
        fontsize=17,
    )
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#333333")

    for region_name, start, end in group_ranges:
        midpoint = (start + end) / 2
        ax.plot(
            [start - 0.42, start - 0.42, end + 0.42, end + 0.42],
            [1.055, 1.075, 1.075, 1.055],
            transform=ax.get_xaxis_transform(),
            color=region_colors[region_name],
            linewidth=2.0,
            clip_on=False,
        )
        ax.text(
            midpoint,
            1.095,
            wrapped_region_name(region_name),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=13.5,
            color="#222222",
            linespacing=0.9,
            clip_on=False,
        )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.025, ticks=[-1, 0, 1])
    colorbar.set_label(colorbar_label, fontsize=15)
    colorbar.ax.tick_params(labelsize=12, length=3)
    fig.subplots_adjust(left=0.075, right=0.91, bottom=0.43, top=0.74)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(output_stem.with_suffix(".png"), dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def process_modality(
    modality: str,
    filename: str,
    common_spots: pd.Index,
    truth: np.ndarray,
    clusters: np.ndarray,
    regions: list[tuple[str, str]],
    region_colors: dict[str, str],
    output_stem: Path,
) -> list[dict[str, object]]:
    matrix, features = load_modality(filename, common_spots)
    selected, records, group_ranges = select_region_markers(
        matrix, features, truth, regions, modality
    )
    values, cluster_order = aggregate_by_cluster(matrix, selected, clusters)
    plot_heatmap(
        values,
        cluster_order,
        [str(features[index]) for index in selected],
        group_ranges,
        region_colors,
        modality,
        output_stem,
    )
    return records


def replot_selected_modality(
    modality: str,
    filename: str,
    common_spots: pd.Index,
    clusters: np.ndarray,
    regions: list[tuple[str, str]],
    region_colors: dict[str, str],
    marker_table: pd.DataFrame,
    output_stem: Path,
) -> None:
    modality_markers = marker_table.loc[marker_table["modality"].eq(modality)].copy()
    features: list[str] = []
    group_ranges: list[tuple[str, int, int]] = []
    for _, region_name in regions:
        region_features = modality_markers.loc[
            modality_markers["region_name"].eq(region_name), "feature"
        ].astype(str).tolist()
        if len(region_features) != MARKERS_PER_REGION:
            raise ValueError(
                f"Cached marker table has {len(region_features)} {modality} markers for {region_name}"
            )
        start = len(features)
        features.extend(region_features)
        group_ranges.append((region_name, start, len(features) - 1))

    cache_path = output_stem.with_name(output_stem.name + "_heatmap_values.csv")
    if cache_path.exists():
        cached = pd.read_csv(cache_path, index_col=0)
        cached = cached.loc[:, features]
        values = cached.to_numpy(dtype=float)
        cluster_order = cached.index.astype(str).tolist()
    else:
        matrix = load_selected_modality(filename, common_spots, features)
        values, cluster_order = aggregate_by_cluster(
            matrix, list(range(len(features))), clusters
        )
        pd.DataFrame(values, index=cluster_order, columns=features).to_csv(
            cache_path, float_format="%.6f"
        )

    plot_heatmap(
        values,
        cluster_order,
        features,
        group_ranges,
        region_colors,
        modality,
        output_stem,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot E18 RNA marker-gene and ATAC marker-peak heatmaps by SpaDTA cluster."
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--recompute-markers",
        action="store_true",
        help="Repeat full marker selection instead of reusing selected_region_markers.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotation, all_regions, region_colors = load_annotation()
    spadta_labels = load_spadta_labels()
    common_spots = annotation.index.intersection(spadta_labels.index, sort=False)
    truth = annotation.loc[common_spots, "cluster"].astype(str).to_numpy()
    clusters = spadta_labels.loc[common_spots].astype(str).to_numpy()
    regions, region_audit = eligible_regions(annotation, common_spots, all_regions)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    marker_path = args.output_dir / "selected_region_markers.csv"
    if marker_path.exists() and not args.recompute_markers:
        marker_table = pd.read_csv(marker_path)
        replot_selected_modality(
            "RNA", "adata_RNA.h5ad", common_spots, clusters,
            regions, region_colors, marker_table,
            args.output_dir / "fig4i_rna_marker_genes",
        )
        replot_selected_modality(
            "ATAC", "adata_ATAC.h5ad", common_spots, clusters,
            regions, region_colors, marker_table,
            args.output_dir / "fig4j_atac_marker_peaks",
        )
    else:
        records = []
        records.extend(process_modality(
            "RNA", "adata_RNA.h5ad", common_spots, truth, clusters,
            regions, region_colors, args.output_dir / "fig4i_rna_marker_genes",
        ))
        records.extend(process_modality(
            "ATAC", "adata_ATAC.h5ad", common_spots, truth, clusters,
            regions, region_colors, args.output_dir / "fig4j_atac_marker_peaks",
        ))
        pd.DataFrame(records).to_csv(marker_path, index=False)
    region_audit.to_csv(args.output_dir / "region_inclusion_audit.csv", index=False)
    print(f"Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
