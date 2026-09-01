from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import RegularPolygon
from scipy import sparse
from scipy.spatial import cKDTree


ROOT = Path("/data/user/hesy/projects/SpatialMETA")
RUN_ROOT = ROOT / "SpaDTA_718" / "runs" / "sm_downstream"
INPUT_ROOT = RUN_ROOT / "inputs"
OUTPUT_DIR = RUN_ROOT / "fig1c"
RECONSTRUCTION_LAYER = "reconstruction_decalign_linear"
OBSERVED_LAYER = "normalized"

SAMPLE_FEATURES = {
    "X49_T": {
        "gene": "CD74",
        "metabolite": "227.10823423028484",
    },
    "248_T": {
        "gene": "COL4A1",
        "metabolite": "238.7857738747654",
    },
    "m1_FMP": {
        "gene": "Mbp",
        "metabolite": "578.21224",
    },
}


def register_colormaps() -> None:
    try:
        matplotlib.colormaps["vlag"]
    except KeyError:
        matplotlib.colormaps.register(sns.color_palette("vlag", as_cmap=True), name="vlag")
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


def to_dense_1d(values) -> np.ndarray:
    if sparse.issparse(values):
        return np.asarray(values.toarray()).ravel()
    return np.asarray(values).ravel()


def feature_values(adata: sc.AnnData, feature: str, layer: str) -> np.ndarray:
    values = to_dense_1d(adata[:, [feature]].layers[layer]).astype(np.float64, copy=False)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def feature_pcc(observed: np.ndarray, reconstructed: np.ndarray) -> float:
    if np.std(observed) < 1e-12 or np.std(reconstructed) < 1e-12:
        return float("nan")
    return float(np.corrcoef(observed, reconstructed)[0, 1])


def spatial_context(adata: sc.AnnData) -> tuple[np.ndarray, np.ndarray, float]:
    library_id = next(iter(adata.uns["spatial"]))
    spatial_info = adata.uns["spatial"][library_id]
    image = np.asarray(spatial_info["images"]["hires"])
    scale = float(spatial_info["scalefactors"]["tissue_hires_scalef"])
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64) * scale
    nearest_distances, _ = cKDTree(coords).query(coords, k=2)
    radius = float(np.median(nearest_distances[:, 1])) / np.sqrt(3.0) * 1.02
    return image, coords, radius


def panel_norm(values: np.ndarray) -> Normalize:
    vmin, vmax = np.nanpercentile(values, [1.0, 99.0])
    vmin = min(float(vmin), 0.0)
    vmax = float(vmax)
    if not np.isfinite(vmax) or np.isclose(vmin, vmax):
        vmax = vmin + 1e-8
    return Normalize(vmin=vmin, vmax=vmax, clip=True)


def add_spatial_panel(
    ax: plt.Axes,
    image: np.ndarray,
    coords: np.ndarray,
    radius: float,
    values: np.ndarray,
    norm: Normalize,
    cmap_name: str,
    title: str,
) -> PatchCollection:
    ax.imshow(image, origin="upper", rasterized=True)
    patches = [
        RegularPolygon((x, y), numVertices=6, radius=radius, orientation=np.pi / 6.0)
        for x, y in coords
    ]
    collection = PatchCollection(
        patches,
        cmap=matplotlib.colormaps[cmap_name],
        norm=norm,
        linewidth=0.0,
        edgecolor="none",
        alpha=0.95,
    )
    collection.set_array(values)
    ax.add_collection(collection)
    ax.set_title(title, fontsize=20, fontweight="bold", pad=10)
    ax.set_xlim(-0.5, image.shape[1] - 0.5)
    ax.set_ylim(image.shape[0] - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    return collection


def load_sample(sample: str, gene: str, metabolite: str) -> tuple[sc.AnnData, dict[str, object]]:
    input_path = INPUT_ROOT / sample / f"{sample}_output.h5ad"
    adata = sc.read_h5ad(input_path)
    if RECONSTRUCTION_LAYER not in adata.layers:
        raise KeyError(f"{sample}: missing layer {RECONSTRUCTION_LAYER}")
    for feature, expected_type in ((gene, "ST"), (metabolite, "SM")):
        if feature not in adata.var_names:
            raise KeyError(f"{sample}: missing feature {feature}")
        actual_type = str(adata.var.loc[feature, "type"])
        if actual_type != expected_type:
            raise ValueError(f"{sample}: {feature} has type {actual_type}, expected {expected_type}")

    values: dict[str, dict[str, np.ndarray]] = {}
    metrics: dict[str, object] = {
        "sample": sample,
        "input_h5ad": str(input_path.resolve()),
        "n_spots": int(adata.n_obs),
    }
    for label, feature in (("gene", gene), ("metabolite", metabolite)):
        observed = feature_values(adata, feature, OBSERVED_LAYER)
        reconstructed = feature_values(adata, feature, RECONSTRUCTION_LAYER)
        values[label] = {"observed": observed, "reconstructed": reconstructed}
        metrics[f"{label}_feature"] = feature
        metrics[f"{label}_pcc"] = feature_pcc(observed, reconstructed)
        metrics[f"{label}_observed_mean"] = float(np.mean(observed))
        metrics[f"{label}_reconstructed_mean"] = float(np.mean(reconstructed))
        metrics[f"{label}_rmse"] = float(np.sqrt(np.mean((observed - reconstructed) ** 2)))
    metrics["values"] = values
    return adata, metrics


def save_sample_figures(adata: sc.AnnData, metrics: dict[str, object]) -> None:
    sample = str(metrics["sample"])
    image, coords, radius = spatial_context(adata)
    sample_dir = OUTPUT_DIR / sample
    sample_dir.mkdir(parents=True, exist_ok=True)

    values = metrics["values"]
    specs = [
        ("gene", str(metrics["gene_feature"]), "vlag"),
        ("metabolite", str(metrics["metabolite_feature"]), "red_white_purple"),
    ]
    for state, state_title in (("observed", "Observed"), ("reconstructed", "Reconstructed")):
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.3))
        collections = []
        for ax, (label, feature, cmap_name) in zip(axes, specs):
            collection = add_spatial_panel(
                ax,
                image,
                coords,
                radius,
                values[label][state],
                panel_norm(values[label][state]),
                cmap_name,
                feature,
            )
            fig.colorbar(collection, ax=ax, fraction=0.046, pad=0.03)
            collections.append(collection)
        fig.tight_layout()
        output_path = sample_dir / f"{sample}_{state}.png"
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        for collection in collections:
            collection.set_rasterized(False)
        fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
        plt.close(fig)


def save_overview(samples: list[tuple[sc.AnnData, dict[str, object]]]) -> None:
    fig, axes = plt.subplots(len(samples), 4, figsize=(18, 5.0 * len(samples)), squeeze=False)
    collections: list[PatchCollection] = []
    for row, (adata, metrics) in enumerate(samples):
        image, coords, radius = spatial_context(adata)
        values = metrics["values"]
        gene = str(metrics["gene_feature"])
        metabolite = str(metrics["metabolite_feature"])
        panel_specs = [
            ("gene", "observed", "vlag", f"{gene} | Observed"),
            ("gene", "reconstructed", "vlag", f"{gene} | Reconstructed"),
            ("metabolite", "observed", "red_white_purple", f"m/z {metabolite} | Observed"),
            (
                "metabolite",
                "reconstructed",
                "red_white_purple",
                f"m/z {metabolite} | Reconstructed",
            ),
        ]
        for col, (label, state, cmap_name, title) in enumerate(panel_specs):
            collection = add_spatial_panel(
                axes[row, col],
                image,
                coords,
                radius,
                values[label][state],
                panel_norm(values[label][state]),
                cmap_name,
                title,
            )
            fig.colorbar(collection, ax=axes[row, col], fraction=0.046, pad=0.025)
            collections.append(collection)
        axes[row, 0].set_ylabel(
            f"ST PCC={metrics['gene_pcc']:.3f}\nSM PCC={metrics['metabolite_pcc']:.3f}",
            fontsize=16,
            rotation=90,
            labelpad=12,
        )

    fig.suptitle("SpaDTA reconstruction of spatial gene and metabolite patterns", fontsize=17, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    output_path = OUTPUT_DIR / "fig1c_three_sm_reconstruction_overview.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    for collection in collections:
        collection.set_rasterized(False)
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    register_colormaps()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    samples: list[tuple[sc.AnnData, dict[str, object]]] = []
    metric_rows: list[dict[str, object]] = []
    for sample, features in SAMPLE_FEATURES.items():
        adata, metrics = load_sample(sample, features["gene"], features["metabolite"])
        save_sample_figures(adata, metrics)
        samples.append((adata, metrics))
        metric_rows.append({key: value for key, value in metrics.items() if key != "values"})
        print(
            f"[fig1c] {sample}: {features['gene']} PCC={metrics['gene_pcc']:.4f}; "
            f"{features['metabolite']} PCC={metrics['metabolite_pcc']:.4f}",
            flush=True,
        )

    save_overview(samples)
    pd.DataFrame(metric_rows).to_csv(OUTPUT_DIR / "reconstruction_metrics.csv", index=False)
    (OUTPUT_DIR / "feature_selection.json").write_text(
        json.dumps(SAMPLE_FEATURES, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[fig1c] wrote {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
