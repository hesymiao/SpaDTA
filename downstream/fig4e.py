from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.collections import PolyCollection
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from fig2g import alpha_shape_boundaries, smooth_closed_boundary
from fig4b_cluster import hex_polygons, save_figure, tissue_canvas


ROOT = Path("/data/user/hesy/projects/SpatialMETA")
DATA_DIR = Path(
    "/bigdat2/user/hesy/spatialmeta/SpatialMETA/smart/SMART_data/Mouse_Brain_E18_S1"
)
RUN_DIR = ROOT / "SpaDTA_718/runs/ATAC/Mouse_Brain_E18_S1"
FIG4D_DIR = ROOT / "SpaDTA_718/runs/atac_downstream/fig4d"
OUTPUT_DIR = ROOT / "SpaDTA_718/runs/atac_downstream/fig4e"

TARGET_CLUSTER = "8"
TARGET_REGION = "C11 / DPallm"
FEATURE_PAIRS = [
    {
        "gene": "Neurod2",
        "peak": "chr11:98329222-98330071",
        "tss_distance_bp": 1,
        "biological_context": "背侧端脑神经元分化",
    },
    {
        "gene": "Sox11",
        "peak": "chr12:27342497-27343356",
        "tss_distance_bp": 353,
        "biological_context": "神经发生与神经元分化",
    },
    {
        "gene": "Grin2b",
        "peak": "chr6:136173140-136174060",
        "tss_distance_bp": 90,
        "biological_context": "谷氨酸能突触和皮层神经元成熟",
    },
    {
        "gene": "Zbtb18",
        "peak": "chr1:177441653-177442571",
        "tss_distance_bp": 238,
        "biological_context": "皮层神经发生与神经元命运决定",
    },
]


def load_spatial_and_labels() -> tuple[pd.Index, pd.DataFrame, np.ndarray]:
    template = ad.read_h5ad(DATA_DIR / "unused_gt.h5ad")
    template.obs_names = template.obs_names.astype(str)
    coords = pd.DataFrame(
        np.asarray(template.obsm["spatial"], dtype=float)[:, :2],
        index=template.obs_names,
        columns=["x", "y"],
    )
    spot_ids = pd.Index(
        pd.read_csv(
            RUN_DIR / "saved_epoch_embeddings/epoch_0300/spot_ids.csv", dtype=str
        )["spot_id"].astype(str)
    )
    labels = pd.read_csv(
        RUN_DIR / "final_protocol/epoch_0300/spot_labels.csv", dtype=str
    )["mclust_label"].astype(str).to_numpy()
    if len(spot_ids) != len(labels):
        raise ValueError("SpaDTA spot IDs and labels have different lengths")
    common = coords.index.intersection(spot_ids, sort=False)
    positions = spot_ids.get_indexer(common)
    return common, coords.loc[common], labels[positions]


def load_feature(filename: str, spot_ids: pd.Index, feature: str, binary: bool) -> np.ndarray:
    adata = ad.read_h5ad(DATA_DIR / filename, backed="r")
    try:
        adata.obs_names = adata.obs_names.astype(str)
        row_positions = adata.obs_names.get_indexer(spot_ids)
        features = pd.Index(adata.var_names.astype(str))
        feature_positions = np.flatnonzero(features == feature)
        if len(feature_positions) == 0:
            raise KeyError(f"{feature!r} is absent from {filename}")
        feature_position = int(feature_positions[0])
        with h5py.File(DATA_DIR / filename, "r") as handle:
            indptr = handle["X/indptr"][:]
            indices = handle["X/indices"][:]
            data = handle["X/data"][:]
        matching_entries = np.flatnonzero(indices == feature_position)
        matching_rows = np.searchsorted(indptr, matching_entries, side="right") - 1
        row_values = np.zeros(len(indptr) - 1, dtype=np.float32)
        row_values[matching_rows] = data[matching_entries].astype(np.float32, copy=False)
        values = row_values[row_positions]
        if binary:
            return (values > 0).astype(np.float32)
        library_column = "gex_umis_count" if filename == "adata_RNA.h5ad" else "atac_peak_region_fragments"
        library_size = pd.to_numeric(
            adata.obs.iloc[row_positions][library_column], errors="coerce"
        ).fillna(0).to_numpy(dtype=np.float32)
        scale = np.divide(
            1e4,
            library_size,
            out=np.zeros_like(library_size, dtype=np.float32),
            where=library_size > 0,
        )
        return np.log1p(values * scale)
    finally:
        adata.file.close()


def lookup_statistics(gene: str, peak: str) -> tuple[dict[str, float], dict[str, float]]:
    def find_row(path: Path, feature: str) -> pd.Series:
        for chunk in pd.read_csv(path, dtype={"spadta_cluster": str}, chunksize=50_000):
            match = chunk.loc[
                chunk["spadta_cluster"].eq(TARGET_CLUSTER) & chunk["feature"].eq(feature)
            ]
            if not match.empty:
                return match.iloc[0]
        raise KeyError(f"No differential row for cluster {TARGET_CLUSTER}, feature {feature}")

    rna_row = find_row(FIG4D_DIR / "rna_cluster_differential.csv.gz", gene)
    atac_row = find_row(FIG4D_DIR / "atac_cluster_differential.csv.gz", peak)
    rna_stats = {
        "mean_in": float(rna_row["mean_in"]),
        "mean_out": float(rna_row["mean_out"]),
        "difference": float(rna_row["mean_difference"]),
        "fdr": float(rna_row["fdr"]),
    }
    atac_stats = {
        "fraction_in": float(atac_row["detected_fraction_in"]),
        "fraction_out": float(atac_row["detected_fraction_out"]),
        "difference": float(atac_row["fraction_difference"]),
        "fdr": float(atac_row["fdr"]),
    }
    return rna_stats, atac_stats


def fdr_label(value: float) -> str:
    return f"{value:.1e}"


def draw_fast_boundary(ax: plt.Axes, target_points: np.ndarray, radius: float) -> None:
    boundaries = alpha_shape_boundaries(target_points, radius * 2.35)
    if not boundaries:
        boundaries = alpha_shape_boundaries(target_points, radius * 3.5)
    for boundary in boundaries:
        boundary = smooth_closed_boundary(boundary)
        ax.plot(
            boundary[:, 0],
            boundary[:, 1],
            color="#FFFFFF",
            linewidth=1.8,
            solid_joinstyle="round",
            solid_capstyle="round",
            zorder=5,
        )


def plot_feature(
    coords: pd.DataFrame,
    labels: np.ndarray,
    values: np.ndarray,
    title: str,
    subtitle: str,
    colorbar_label: str,
    output_stem: Path,
    binary: bool = False,
) -> None:
    canvas, points, radius = tissue_canvas(coords)
    polygons = hex_polygons(points, radius * 1.002)
    if binary:
        value_norm = Normalize(vmin=0.0, vmax=1.0)
    else:
        positive = values[values > 0]
        vmax = float(np.quantile(positive, 0.98)) if len(positive) else 1.0
        value_norm = Normalize(vmin=0.0, vmax=max(vmax, 1e-6))
    cmap = matplotlib.colormaps["viridis"]
    facecolors = cmap(value_norm(values))

    fig, ax = plt.subplots(figsize=(8.6, 8.2), dpi=130)
    ax.imshow(canvas, origin="upper", interpolation="none")
    ax.add_collection(
        PolyCollection(
            polygons,
            facecolors=facecolors,
            edgecolors="#FFFFFF",
            linewidths=0.30,
            antialiaseds=True,
            zorder=3,
        )
    )
    target = labels == TARGET_CLUSTER
    draw_fast_boundary(ax, points[target], radius)
    ax.set_xlim(-0.5, canvas.shape[1] - 0.5)
    ax.set_ylim(canvas.shape[0] - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=20, fontweight="bold", pad=29)
    ax.text(
        0.5,
        1.015,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=15,
    )
    ax.axis("off")
    colorbar = fig.colorbar(
        ScalarMappable(norm=value_norm, cmap=cmap),
        ax=ax,
        fraction=0.035,
        pad=0.018,
        shrink=0.72,
    )
    colorbar.set_label(colorbar_label, fontsize=15)
    colorbar.ax.tick_params(labelsize=13)
    if binary:
        colorbar.set_ticks([0, 1])
        colorbar.set_ticklabels(["Closed", "Accessible"])
    fig.subplots_adjust(left=0.02, right=0.88, bottom=0.02, top=0.90)
    save_figure(fig, output_stem)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    spot_ids, coords, labels = load_spatial_and_labels()
    statistics_rows = []
    pair_summaries = []
    for pair in FEATURE_PAIRS:
        gene = str(pair["gene"])
        peak = str(pair["peak"])
        distance = int(pair["tss_distance_bp"])
        rna_values = load_feature("adata_RNA.h5ad", spot_ids, gene, binary=False)
        atac_values = load_feature("adata_ATAC.h5ad", spot_ids, peak, binary=True)
        rna_stats, atac_stats = lookup_statistics(gene, peak)

        plot_feature(
            coords,
            labels,
            rna_values,
            gene,
            (
                f"SpaDTA cluster {TARGET_CLUSTER} ({TARGET_REGION}) vs other domains, "
                f"FDR = {fdr_label(rna_stats['fdr'])}"
            ),
            "Normalized expression",
            OUTPUT_DIR / f"fig4e_rna_{gene}",
        )
        plot_feature(
            coords,
            labels,
            atac_values,
            f"{peak} ({gene} promoter)",
            (
                f"{distance} bp from {gene} TSS; cluster {TARGET_CLUSTER} vs other domains, "
                f"FDR = {fdr_label(atac_stats['fdr'])}"
            ),
            "Peak state",
            OUTPUT_DIR / f"fig4e_atac_{gene}_promoter_peak",
            binary=True,
        )
        statistics_rows.extend(
            [
                {
                    "modality": "RNA",
                    "feature": gene,
                    "linked_gene": gene,
                    "spadta_cluster": TARGET_CLUSTER,
                    "majority_region": TARGET_REGION,
                    **rna_stats,
                },
                {
                    "modality": "ATAC",
                    "feature": peak,
                    "linked_gene": gene,
                    "tss_distance_bp": distance,
                    "spadta_cluster": TARGET_CLUSTER,
                    "majority_region": TARGET_REGION,
                    **atac_stats,
                },
            ]
        )
        pair_summaries.append({**pair, "rna": rna_stats, "atac": atac_stats})

    pd.DataFrame(statistics_rows).to_csv(
        OUTPUT_DIR / "fig4e_selected_gene_peak_statistics.csv", index=False
    )
    (OUTPUT_DIR / "fig4e_summary.json").write_text(
        json.dumps(
            {
                "interpretation": (
                    "SpaDTA cluster 8, aligned mainly to C11/DPallm, shows concordant RNA expression "
                    "and promoter-proximal chromatin accessibility for four neuronal genes."
                ),
                "selection": (
                    "Pairs were selected from full cluster-vs-rest differential scans because both modalities "
                    "are significant, each ATAC peak is within 500 bp of the linked gene TSS, and the genes are "
                    "biologically relevant to dorsal-pallium neuronal development."
                ),
                "pairs": pair_summaries,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
