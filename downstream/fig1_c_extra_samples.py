from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
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
OUTPUT_DIR = RUN_ROOT / "fig1_c"
SOURCE_RECONSTRUCTION_LAYER = "reconstruction_decalign_linear"

SAMPLE_CONFIGS = {
    "248_T": {
        "input_h5ad": RUN_ROOT / "inputs" / "248_T" / "248_T_output.h5ad",
        "gene_feature": "SEC61G",
        "metabolite_feature": "932.6503304838307",
    },
    "m3_FMP": {
        "input_h5ad": RUN_ROOT / "inputs" / "m3_FMP" / "m3_FMP_output.h5ad",
        "gene_feature": "Fth1",
        "metabolite_feature": "554.21582",
    },
}


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


def prepare_paper_style_adata(input_h5ad: Path, metabolite_feature: str) -> sc.AnnData:
    adata = sc.read_h5ad(input_h5ad)
    if SOURCE_RECONSTRUCTION_LAYER not in adata.layers:
        raise KeyError(f"Missing reconstruction layer: {SOURCE_RECONSTRUCTION_LAYER}")
    if metabolite_feature not in adata.var_names:
        raise KeyError(f"Missing feature: {metabolite_feature}")

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


def get_selected_feature_pcc(
    adata: sc.AnnData,
    feature_type: str,
    selected_feature: str,
    truth_layer: str,
    pred_layer: str,
) -> float:
    var_types = adata.var["type"].astype(str).to_numpy()
    feature_mask = var_types == feature_type
    if not np.any(feature_mask):
        raise ValueError(f"No {feature_type} features found when selecting PCC feature.")

    truth = np.asarray(
        adata.layers[truth_layer][:, feature_mask].toarray()
        if sparse.issparse(adata.layers[truth_layer])
        else adata.layers[truth_layer][:, feature_mask],
        dtype=np.float64,
    )
    pred = np.asarray(
        adata.layers[pred_layer][:, feature_mask].toarray()
        if sparse.issparse(adata.layers[pred_layer])
        else adata.layers[pred_layer][:, feature_mask],
        dtype=np.float64,
    )

    truth_centered = truth - truth.mean(axis=0, keepdims=True)
    pred_centered = pred - pred.mean(axis=0, keepdims=True)
    denom = np.sqrt((truth_centered * truth_centered).sum(axis=0) * (pred_centered * pred_centered).sum(axis=0))

    pcc = np.full(truth.shape[1], np.nan, dtype=np.float64)
    valid = denom > 1e-12
    pcc[valid] = (truth_centered[:, valid] * pred_centered[:, valid]).sum(axis=0) / denom[valid]
    if not np.any(np.isfinite(pcc)):
        raise ValueError(f"No finite {feature_type} PCC values found.")

    feature_names = adata.var_names.to_numpy()[feature_mask]
    selected_idx = np.where(feature_names == selected_feature)[0]
    if selected_idx.size == 0:
        raise ValueError(f"Selected {feature_type} feature not found: {selected_feature}")
    return float(pcc[int(selected_idx[0])])


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


def plot_hex_spatial_pair(
    adata: sc.AnnData,
    sample_name: str,
    gene_feature: str,
    metabolite_feature: str,
    layer: str,
    output_path: Path,
) -> None:
    image, coords = get_spatial_image_and_coords(adata)
    hex_radius = estimate_hex_radius(coords)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    collections = []
    gene_pcc = get_selected_feature_pcc(
        adata,
        feature_type="ST",
        selected_feature=gene_feature,
        truth_layer="normalized",
        pred_layer="reconstruction",
    )
    metabolite_pcc = get_selected_feature_pcc(
        adata,
        feature_type="SM",
        selected_feature=metabolite_feature,
        truth_layer="normalized",
        pred_layer="reconstruction",
    )
    feature_specs = [
        (gene_feature, "vlag"),
        (metabolite_feature, "red_white_purple"),
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
        f"{sample_name} | {gene_feature} (PCC={gene_pcc:.4f}) | {metabolite_feature} (PCC={metabolite_pcc:.4f})",
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for sample_name, config in SAMPLE_CONFIGS.items():
        adata = prepare_paper_style_adata(
            input_h5ad=config["input_h5ad"],
            metabolite_feature=config["metabolite_feature"],
        )

        print(f"plotting {sample_name} normalized")
        plot_hex_spatial_pair(
            adata=adata,
            sample_name=sample_name,
            gene_feature=config["gene_feature"],
            metabolite_feature=config["metabolite_feature"],
            layer="normalized",
            output_path=OUTPUT_DIR / f"fig1_c_{sample_name}_original.png",
        )
        print(f"saved {sample_name} normalized")

        print(f"plotting {sample_name} reconstruction")
        plot_hex_spatial_pair(
            adata=adata,
            sample_name=sample_name,
            gene_feature=config["gene_feature"],
            metabolite_feature=config["metabolite_feature"],
            layer="reconstruction",
            output_path=OUTPUT_DIR / f"fig1_c_{sample_name}_denoised.png",
        )
        print(f"saved {sample_name} reconstruction")

        print(OUTPUT_DIR / f"fig1_c_{sample_name}_original.png")
        print(OUTPUT_DIR / f"fig1_c_{sample_name}_denoised.png")


if __name__ == "__main__":
    main()
