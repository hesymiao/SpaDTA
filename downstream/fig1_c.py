from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
from matplotlib.patches import RegularPolygon
from scipy import sparse
from scipy.spatial import cKDTree


ROOT = Path("/data/user/hesy/projects/SpatialMETA")
RUN_ROOT = ROOT / "SpaDTA_718" / "runs" / "sm_downstream"
INPUT_H5AD = RUN_ROOT / "inputs" / "Y7_T" / "Y7_T_output.h5ad"
OUTPUT_DIR = RUN_ROOT / "fig1_c"

METABOLITE_FEATURE = "309.27992033396646"
SOURCE_RECONSTRUCTION_LAYER = "reconstruction_decalign_linear"
SELECTED_GENE_FEATURE = "CD74"


def ensure_vlag_colormap() -> None:
    try:
        matplotlib.colormaps["vlag"]
    except KeyError:
        matplotlib.colormaps.register(
            sns.color_palette("vlag", as_cmap=True),
            name="vlag",
        )
    try:
        matplotlib.colormaps["red_white_purple"]
    except KeyError:
        matplotlib.colormaps.register(
            LinearSegmentedColormap.from_list(
                "red_white_purple",
                ["#5a2a83", "#f5efec", "#c43d3d"],
            ),
            name="red_white_purple",
        )


def prepare_paper_style_adata(input_h5ad: Path) -> sc.AnnData:
    adata = sc.read_h5ad(input_h5ad)
    if SOURCE_RECONSTRUCTION_LAYER not in adata.layers:
        raise KeyError(f"Missing reconstruction layer: {SOURCE_RECONSTRUCTION_LAYER}")
    if METABOLITE_FEATURE not in adata.var_names:
        raise KeyError(f"Missing feature: {METABOLITE_FEATURE}")

    adata.layers["reconstruction"] = adata.layers[SOURCE_RECONSTRUCTION_LAYER].copy()
    return adata


def to_dense_1d(values) -> np.ndarray:
    if sparse.issparse(values):
        return np.asarray(values.toarray()).ravel()
    return np.asarray(values).ravel()


def get_feature_values(adata: sc.AnnData, feature: str, layer: str) -> np.ndarray:
    matrix = adata[:, [feature]].layers[layer]
    values = to_dense_1d(matrix).astype(np.float64, copy=False)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def get_feature_pcc(adata: sc.AnnData, feature: str, truth_layer: str, pred_layer: str) -> float:
    truth = get_feature_values(adata, feature, truth_layer)
    pred = get_feature_values(adata, feature, pred_layer)
    if np.std(truth) < 1e-12 or np.std(pred) < 1e-12:
        return float("nan")
    return float(np.corrcoef(truth, pred)[0, 1])


def select_high_pcc_gene(adata: sc.AnnData, truth_layer: str, pred_layer: str) -> tuple[str, float]:
    var_types = adata.var["type"].astype(str).to_numpy()
    st_mask = var_types == "ST"
    if not np.any(st_mask):
        raise ValueError("No ST features found when selecting high-PCC gene.")

    truth = np.asarray(adata.layers[truth_layer][:, st_mask].toarray() if sparse.issparse(adata.layers[truth_layer]) else adata.layers[truth_layer][:, st_mask], dtype=np.float64)
    pred = np.asarray(adata.layers[pred_layer][:, st_mask].toarray() if sparse.issparse(adata.layers[pred_layer]) else adata.layers[pred_layer][:, st_mask], dtype=np.float64)

    truth_centered = truth - truth.mean(axis=0, keepdims=True)
    pred_centered = pred - pred.mean(axis=0, keepdims=True)
    denom = np.sqrt((truth_centered * truth_centered).sum(axis=0) * (pred_centered * pred_centered).sum(axis=0))

    pcc = np.full(truth.shape[1], np.nan, dtype=np.float64)
    valid = denom > 1e-12
    pcc[valid] = (truth_centered[:, valid] * pred_centered[:, valid]).sum(axis=0) / denom[valid]
    if not np.any(np.isfinite(pcc)):
        raise ValueError("No finite ST feature PCC values found.")

    gene_names = adata.var_names.to_numpy()[st_mask]
    selected_idx = np.where(gene_names == SELECTED_GENE_FEATURE)[0]
    if selected_idx.size == 0:
        raise ValueError(f"Selected ST feature not found: {SELECTED_GENE_FEATURE}")
    idx = int(selected_idx[0])
    return str(gene_names[idx]), float(pcc[idx])


def get_spatial_image_and_coords(adata: sc.AnnData) -> tuple[np.ndarray, np.ndarray]:
    library_id = next(iter(adata.uns["spatial"]))
    spatial_info = adata.uns["spatial"][library_id]
    image = np.asarray(spatial_info["images"]["hires"])
    scale = float(spatial_info["scalefactors"]["tissue_hires_scalef"])
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64) * scale
    return image, coords


def estimate_hex_radius(coords: np.ndarray) -> float:
    tree = cKDTree(coords)
    nearest_distances, _ = tree.query(coords, k=2)
    center_spacing = float(np.median(nearest_distances[:, 1]))
    return center_spacing / np.sqrt(3.0) * 1.02


def plot_hex_panel(
    ax: plt.Axes,
    image: np.ndarray,
    coords: np.ndarray,
    values: np.ndarray,
    feature_name: str,
    cmap_name: str,
    hex_radius: float,
) -> tuple[PatchCollection, matplotlib.image.AxesImage]:
    image_artist = ax.imshow(image, origin="upper", rasterized=True)

    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-8
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.colormaps[cmap_name]

    patches = [
        RegularPolygon(
            (x, y),
            numVertices=6,
            radius=hex_radius,
            orientation=np.pi / 6.0,
        )
        for x, y in coords
    ]
    collection = PatchCollection(
        patches,
        cmap=cmap,
        norm=norm,
        linewidth=0.0,
        edgecolor="none",
        alpha=0.95,
    )
    collection.set_array(values)
    ax.add_collection(collection)

    ax.set_title(feature_name)
    ax.set_xlim(-0.5, image.shape[1] - 0.5)
    ax.set_ylim(image.shape[0] - 0.5, -0.5)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    plt.colorbar(collection, ax=ax, fraction=0.046, pad=0.04)
    return collection, image_artist


def plot_hex_spatial_pair(adata: sc.AnnData, layer: str, output_path: Path) -> None:
    image, coords = get_spatial_image_and_coords(adata)
    hex_radius = estimate_hex_radius(coords)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    collections = []
    gene_feature, gene_pcc = select_high_pcc_gene(
        adata,
        truth_layer="normalized",
        pred_layer="reconstruction",
    )
    metabolite_pcc = get_feature_pcc(
        adata,
        METABOLITE_FEATURE,
        truth_layer="normalized",
        pred_layer="reconstruction",
    )
    feature_specs = [
        (gene_feature, "vlag"),
        (METABOLITE_FEATURE, "red_white_purple"),
    ]
    for ax, (feature_name, cmap_name) in zip(axes, feature_specs):
        values = get_feature_values(adata, feature_name, layer)
        collection, _ = plot_hex_panel(
            ax=ax,
            image=image,
            coords=coords,
            values=values,
            feature_name=feature_name,
            cmap_name=cmap_name,
            hex_radius=hex_radius,
        )
        collections.append(collection)
    fig.suptitle(
        f"{gene_feature} (PCC={gene_pcc:.4f}) | {METABOLITE_FEATURE} (PCC={metabolite_pcc:.4f})",
        fontsize=14,
        y=0.98,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    for collection in collections:
        collection.set_rasterized(False)
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_vlag_colormap()
    adata = prepare_paper_style_adata(INPUT_H5AD)

    print("plotting normalized")
    plot_hex_spatial_pair(adata, layer="normalized", output_path=OUTPUT_DIR / "fig1_c_original.png")
    print("saved normalized")

    print("plotting reconstruction")
    plot_hex_spatial_pair(adata, layer="reconstruction", output_path=OUTPUT_DIR / "fig1_c_denoised.png")
    print("saved reconstruction")

    print(OUTPUT_DIR / "fig1_c_original.png")
    print(OUTPUT_DIR / "fig1_c_denoised.png")


if __name__ == "__main__":
    main()
