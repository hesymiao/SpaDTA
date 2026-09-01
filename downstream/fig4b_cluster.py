from __future__ import annotations

from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull
from sklearn.preprocessing import StandardScaler
import umap


ROOT = Path("/data/user/hesy/projects/SpatialMETA")
RESULT = ROOT / "SpaDTA_718/runs/atac_result"
SPADTA_RUN = ROOT / "SpaDTA_718/runs/ATAC"
OUT = ROOT / "SpaDTA_718/runs/atac_downstream/fig4b_cluster"
DATA = Path("/bigdat2/user/hesy/spatialmeta/SpatialMETA/smart/SMART_data")
SAMPLE = "Mouse_Brain_E18_S1"

METHODS = [
    "SpaDTA", "PRESENT", "SMART", "WNN", "MOFA+", "SNF", "CellCharter",
    "SpatialGlue", "MEFISTO", "MultiVI", "COSMOS", "scMM", "MISO",
]

MCLUST_SOURCES = {
    "PRESENT": "present_seed2020",
    "SMART": "smart_existing_uniform_mclust",
    "MOFA+": "mofa",
    "CellCharter": "uniform_mclust_recheck/cellcharter",
    "SpatialGlue": "spatialglue",
    "MEFISTO": "mefisto",
    "MultiVI": "multivi",
    "COSMOS": "uniform_mclust_recheck/cosmos",
    "scMM": "scmm",
    "MISO": "uniform_mclust_recheck/miso",
}

# Distinct, colorblind-conscious colors assembled from Tableau and Okabe-Ito.
CLUSTER_COLORS = [
    "#4E79A7", "#E15759", "#59A14F", "#F28E2B", "#B07AA1",
    "#76B7B2", "#EDC948", "#FF9DA7", "#9C755F", "#7F7F7F",
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9",
]
UNMATCHED_COLORS = ["#222222", "#8DD3C7", "#B3DE69", "#FCCDE5", "#BC80BD"]


def natural_key(value: str) -> tuple[object, ...]:
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value))


def safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def load_slice() -> tuple[pd.Index, pd.DataFrame, pd.Series, list[tuple[str, str]]]:
    template = sc.read_h5ad(DATA / SAMPLE / "unused_gt.h5ad")
    template.obs_names = template.obs_names.astype(str)
    coords = pd.DataFrame(
        np.asarray(template.obsm["spatial"], dtype=float)[:, :2],
        index=template.obs_names,
        columns=["x", "y"],
    )
    annotation = pd.read_csv(DATA / SAMPLE / "anno.csv", dtype=str)
    region_id_column = "0" if "0" in annotation.columns else None
    if "cluster" not in annotation.columns:
        raise KeyError(f"{SAMPLE} anno.csv has no cluster region-name column")
    if region_id_column is None:
        annotation["region_id"] = annotation["cluster"]
        region_id_column = "region_id"
    region_pairs = (
        annotation[[region_id_column, "cluster"]]
        .drop_duplicates()
        .sort_values(region_id_column, key=lambda values: values.map(natural_key))
    )
    if region_pairs[region_id_column].duplicated().any() or region_pairs["cluster"].duplicated().any():
        raise ValueError("Ground-truth region IDs and names must have a one-to-one mapping")
    regions = list(region_pairs.itertuples(index=False, name=None))
    ground_truth = annotation.set_index("barcode")["cluster"]
    common = coords.index.intersection(ground_truth.index, sort=False)
    return common, coords.loc[common], ground_truth.loc[common], regions


def load_method(method: str) -> tuple[np.ndarray, pd.Index, np.ndarray]:
    if method == "SpaDTA":
        embedding_dir = SPADTA_RUN / SAMPLE / "saved_epoch_embeddings/epoch_0300"
        embedding = np.load(embedding_dir / "branch_scaled_full.npy")
        spot_ids = pd.Index(pd.read_csv(embedding_dir / "spot_ids.csv")["spot_id"].astype(str))
        labels = pd.read_csv(
            SPADTA_RUN / SAMPLE / "final_protocol/epoch_0300/spot_labels.csv",
            dtype={"mclust_label": str},
        )["mclust_label"].to_numpy()
        return embedding, spot_ids, labels

    if method in MCLUST_SOURCES:
        method_dir = RESULT / MCLUST_SOURCES[method] / SAMPLE
        packed = np.load(method_dir / "mclust_input.npz", allow_pickle=True)
        embedding = np.asarray(packed["embedding"], dtype=float)
        spot_ids = pd.Index(packed["obs_names"].astype(str))
        labels = np.load(method_dir / "mclust_labels.npy", allow_pickle=True).astype(str)
        return embedding, spot_ids, labels

    if method in {"WNN", "SNF"}:
        method_dir = RESULT / method.lower() / SAMPLE
        adata = sc.read_h5ad(method_dir / f"adata_{method.lower()}.h5ad")
        adata.obs_names = adata.obs_names.astype(str)
        embedding = np.hstack([
            np.asarray(adata.obsm["X_RNA"], dtype=float),
            np.asarray(adata.obsm["X_ATAC"], dtype=float),
        ])
        return embedding, pd.Index(adata.obs_names), adata.obs["paper_cluster"].astype(str).to_numpy()

    raise KeyError(method)


def align_method(
    embedding: np.ndarray,
    spot_ids: pd.Index,
    labels: np.ndarray,
    slice_ids: pd.Index,
) -> tuple[np.ndarray, pd.Index, np.ndarray]:
    if len(embedding) != len(spot_ids) or len(labels) != len(spot_ids):
        raise ValueError("Embedding, barcode, and cluster-label lengths differ")
    if spot_ids.has_duplicates:
        raise ValueError("Duplicate method barcodes are not supported")
    common = slice_ids.intersection(spot_ids, sort=False)
    positions = spot_ids.get_indexer(common)
    if (positions < 0).any():
        raise RuntimeError("Failed to align method barcodes")
    return embedding[positions], common, np.asarray(labels)[positions].astype(str)


def ground_truth_colors(regions: list[tuple[str, str]]) -> dict[str, str]:
    if len(regions) > len(CLUSTER_COLORS):
        extra = [to_hex(color) for color in plt.get_cmap("tab20").colors]
        palette = CLUSTER_COLORS + [color for color in extra if color not in CLUSTER_COLORS]
    else:
        palette = CLUSTER_COLORS
    return {region_name: palette[index] for index, (_, region_name) in enumerate(regions)}


def match_cluster_colors(
    labels: np.ndarray,
    truth: np.ndarray,
    regions: list[tuple[str, str]],
    gt_colors: dict[str, str],
) -> tuple[list[str], dict[str, str], list[dict[str, object]]]:
    categories = sorted(pd.unique(labels).astype(str), key=natural_key)
    region_names = [region_name for _, region_name in regions]
    overlap = pd.crosstab(
        pd.Categorical(labels, categories=categories),
        pd.Categorical(truth, categories=region_names),
        dropna=False,
    ).reindex(index=categories, columns=region_names, fill_value=0)
    rows, columns = linear_sum_assignment(-overlap.to_numpy(dtype=float))
    matched_regions = {
        categories[row]: region_names[column]
        for row, column in zip(rows, columns)
        if int(overlap.iloc[row, column]) > 0
    }
    unmatched_index = 0
    color_map: dict[str, str] = {}
    records = []
    for category in categories:
        region_name = matched_regions.get(category)
        if region_name is None:
            color = UNMATCHED_COLORS[unmatched_index % len(UNMATCHED_COLORS)]
            unmatched_index += 1
            matched_spots = 0
        else:
            color = gt_colors[region_name]
            matched_spots = int(overlap.loc[category, region_name])
        cluster_size = int((labels == category).sum())
        color_map[category] = color
        records.append({
            "predicted_cluster": category,
            "matched_gt_region": region_name or "unmatched",
            "color": color,
            "overlap_spots": matched_spots,
            "cluster_size": cluster_size,
            "overlap_fraction": matched_spots / cluster_size if cluster_size else np.nan,
        })
    return categories, color_map, records


def style_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def fitted_lattice(
    coords: pd.DataFrame,
    width: int,
    height: int,
    margin: float,
) -> tuple[np.ndarray, float]:
    grid = coords[["x", "y"]].to_numpy(dtype=float)
    columns = grid[:, 0] - grid[:, 0].min()
    rows = grid[:, 1] - grid[:, 1].min()
    # Pointy-top hex lattice: adjacent rows are staggered by half a cell.
    lattice = np.column_stack([
        np.sqrt(3.0) * (columns + 0.5 * (rows.astype(int) % 2)),
        1.5 * rows,
    ])
    low = lattice.min(axis=0)
    high = lattice.max(axis=0)
    span = np.maximum(high - low, 1.0)
    radius = min((width - 2 * margin) / span[0], (height - 2 * margin) / span[1])
    fitted = margin + (lattice - low) * radius
    return fitted, radius


def tissue_canvas(
    slice_coords: pd.DataFrame,
    width: int = 900,
    height: int = 900,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Create a neutral raster slice canvas when the ATAC artifact has no tissue image."""
    margin = 28.0
    fitted, radius = fitted_lattice(slice_coords, width, height, margin)
    yy, xx = np.mgrid[0:height, 0:width]
    # A low-contrast, deterministic tissue-like field keeps the slice footprint
    # visible without inventing biological structures absent from the ATAC files.
    field = gaussian_filter(np.random.default_rng(18).normal(0, 1, (height, width)), sigma=70)
    field = field / max(float(np.max(np.abs(field))), 1e-8)
    radial = ((xx - width * 0.5) / width) ** 2 + ((yy - height * 0.5) / height) ** 2
    canvas = np.empty((height, width, 3), dtype=float)
    canvas[..., 0] = 0.925 + 0.030 * field - 0.020 * radial
    canvas[..., 1] = 0.895 + 0.024 * field - 0.014 * radial
    canvas[..., 2] = 0.925 + 0.032 * field
    hull = ConvexHull(fitted)
    mask = MplPath(fitted[hull.vertices]).contains_points(
        np.column_stack([xx.ravel(), yy.ravel()])
    ).reshape(height, width)
    canvas[~mask] = np.array([0.965, 0.965, 0.965])
    return canvas.clip(0, 1), fitted, radius


def hex_polygons(points: np.ndarray, radius: float) -> list[np.ndarray]:
    angles = np.arange(6, dtype=float) * np.pi / 3.0 + np.pi / 6.0
    offsets = np.column_stack([np.cos(angles), np.sin(angles)]) * radius
    return [point[None, :] + offsets for point in points]


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    dpi = 170
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white", pil_kwargs={"optimize": True})
    plt.close(fig)
    png = stem.with_suffix(".png")
    if png.stat().st_size >= 1_000_000:
        from PIL import Image

        image = Image.open(png).convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
        image.save(png, optimize=True)
    if png.stat().st_size >= 1_000_000:
        raise RuntimeError(f"PNG exceeds 1000 KB: {png} ({png.stat().st_size} bytes)")


def plot_spatial(
    title: str,
    filename: str,
    coords: pd.DataFrame,
    slice_coords: pd.DataFrame,
    labels: np.ndarray,
    colors: dict[str, str],
) -> None:
    canvas, all_points, radius = tissue_canvas(slice_coords)
    point_frame = pd.DataFrame(all_points, index=slice_coords.index, columns=["x", "y"])
    points = point_frame.loc[coords.index].to_numpy(dtype=float)
    polygons = hex_polygons(points, radius * 1.002)
    facecolors = [colors[str(label)] for label in labels]
    fig = plt.figure(figsize=(7.0, 7.0), dpi=100, frameon=False)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(canvas, origin="upper", alpha=1.0, interpolation="none")
    ax.add_collection(PolyCollection(
        polygons, facecolors=facecolors, edgecolors="#FFFFFF",
        linewidths=0.34, antialiaseds=True, zorder=3,
    ))
    ax.set_xlim(-0.5, canvas.shape[1] - 0.5)
    ax.set_ylim(canvas.shape[0] - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    save_figure(fig, OUT / filename)


def plot_embedding(
    method: str,
    embedding: np.ndarray,
    labels: np.ndarray,
    categories: list[str],
    colors: dict[str, str],
) -> None:
    scaled = StandardScaler().fit_transform(np.nan_to_num(embedding, copy=False))
    projected = umap.UMAP(
        n_neighbors=15, min_dist=0.35, metric="euclidean",
        random_state=42, n_jobs=1,
    ).fit_transform(scaled)
    fig = plt.figure(figsize=(5.0, 5.0), frameon=False)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    for category in categories:
        mask = labels == category
        ax.scatter(
            projected[mask, 0], projected[mask, 1], s=9,
            c=colors[category], edgecolors="none", alpha=0.9,
            rasterized=False,
        )
    ax.margins(0.02)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    save_figure(fig, OUT / f"{safe_name(method)}_embedding")


def plot_ground_truth_legend(
    regions: list[tuple[str, str]],
    colors: dict[str, str],
) -> None:
    handles = [
        Patch(
            facecolor=colors[region_name],
            edgecolor="#444444",
            linewidth=0.6,
            label=f"{region_id}  {region_name}",
        )
        for region_id, region_name in regions
    ]
    fig, ax = plt.subplots(figsize=(10.5, 4.2), dpi=120)
    ax.axis("off")
    ax.legend(
        handles=handles,
        title="Ground truth regions",
        loc="center",
        ncol=2,
        frameon=False,
        fontsize=11,
        title_fontsize=13,
        handlelength=1.8,
        handleheight=1.1,
        handletextpad=0.7,
        columnspacing=2.0,
        labelspacing=0.8,
    )
    fig.tight_layout()
    save_figure(fig, OUT / "ground_truth_region_legend")


def main() -> None:
    slice_ids, slice_coords, ground_truth, regions = load_slice()
    gt_colors = ground_truth_colors(regions)
    plot_spatial(
        "Ground truth", "ground_truth_spatial",
        slice_coords, slice_coords, ground_truth.to_numpy(dtype=str),
        gt_colors,
    )
    plot_ground_truth_legend(regions, gt_colors)
    audit = []
    color_audit = []
    for method in METHODS:
        embedding, spot_ids, labels = load_method(method)
        embedding, aligned_ids, labels = align_method(embedding, spot_ids, labels, slice_ids)
        coords = slice_coords.loc[aligned_ids]
        categories, colors, match_records = match_cluster_colors(
            labels,
            ground_truth.loc[aligned_ids].to_numpy(dtype=str),
            regions,
            gt_colors,
        )
        plot_spatial(
            f"{method} spatial clusters", f"{safe_name(method)}_spatial",
            coords, slice_coords, labels,
            colors,
        )
        plot_embedding(method, embedding, labels, categories, colors)
        color_audit.extend({"method": method, **record} for record in match_records)
        audit.append({
            "method": method,
            "spots": len(aligned_ids),
            "clusters": int(pd.Series(labels).nunique()),
            "embedding_dim": int(embedding.shape[1]),
        })
    pd.DataFrame(audit).to_csv(OUT / "plot_audit.csv", index=False)
    pd.DataFrame(color_audit).to_csv(OUT / "cluster_to_gt_color_mapping.csv", index=False)


if __name__ == "__main__":
    main()
