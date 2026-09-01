from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc


project_root = Path("/data/user/hesy/projects/SpatialMETA")
sample_name = "Y27_T"
input_h5ad = project_root / "SpaDTA_718" / "runs" / "downstream_runs" / "sm_to_st_generation_spatial_top_third" / sample_name / "superres_eval_fixed_train_mean_libsize" / "celllevel_generated_st_from_sm.h5ad"
layer = "generated_st_log1p_from_sm_celllevel"
gene = "RBP4"
output_dir = project_root / "SpaDTA_718" / "runs" / "downstream_runs" / "sm_to_st_generation_spatial_top_third" / sample_name / "top_third_region_plots"
title = None
point_size = 6.0
background_point_size = 2.0
dpi = 220
cmap = "viridis"
vmin = None
vmax = None
clip_quantile = 0.99
region_axis = "y"
region_side = "top"
region_fraction = 1.0 / 3.0
threshold = None
show_background = True
background_color = "#d9d9d9"
background_alpha = 0.35


def region_mask(coords: np.ndarray, axis: str, side: str, threshold: float) -> np.ndarray:
    axis_idx = 0 if axis == "x" else 1
    values = coords[:, axis_idx]
    if side in {"top", "left"}:
        return values <= threshold
    return values >= threshold


def compute_region_threshold(coords: np.ndarray, axis: str, side: str, region_fraction: float) -> float:
    axis_idx = 0 if axis == "x" else 1
    values = coords[:, axis_idx].astype(np.float32, copy=False)
    value_min = float(np.min(values))
    value_max = float(np.max(values))
    value_range = value_max - value_min
    frac = float(region_fraction)
    if side in {"top", "left"}:
        return value_min + value_range * frac
    return value_max - value_range * frac


def plot_layer_gene_on_spatial_region(
    input_h5ad: Path,
    layer: str,
    gene: str,
    output_dir: Path,
    title: str | None = None,
    point_size: float = 6.0,
    background_point_size: float = 2.0,
    dpi: int = 220,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    clip_quantile: float = 0.99,
    region_axis: str = "y",
    region_side: str = "top",
    region_fraction: float = 1.0 / 3.0,
    threshold: float | None = None,
    show_background: bool = True,
    background_color: str = "#d9d9d9",
    background_alpha: float = 0.35,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(input_h5ad)
    if layer not in adata.layers:
        raise KeyError(f"Missing layer: {layer}")
    if "spatial" not in adata.obsm:
        raise KeyError("adata.obsm['spatial'] is required")

    var_names = adata.var_names.astype(str).to_numpy()
    matches = np.where(var_names == str(gene))[0]
    if len(matches) == 0:
        raise KeyError(f"Gene not found: {gene}")
    gene_idx = int(matches[0])

    values = np.asarray(adata.layers[layer][:, gene_idx]).reshape(-1).astype(np.float32, copy=False)
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    region_threshold = compute_region_threshold(coords, axis=region_axis, side=region_side, region_fraction=region_fraction) if threshold is None else float(threshold)
    mask = region_mask(coords, axis=region_axis, side=region_side, threshold=region_threshold)
    if not np.any(mask):
        raise ValueError("Selected region contains 0 points")

    region_values = values[mask]
    region_coords = coords[mask]

    plot_vmin = float(np.nanmin(region_values)) if vmin is None else float(vmin)
    if vmax is None:
        plot_vmax = float(np.nanquantile(region_values, clip_quantile))
        plot_vmax = max(plot_vmax, plot_vmin + 1e-8)
    else:
        plot_vmax = float(vmax)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    if show_background:
        bg_coords = coords[~mask]
        if len(bg_coords) > 0:
            ax.scatter(
                bg_coords[:, 0],
                bg_coords[:, 1],
                c=background_color,
                s=background_point_size,
                alpha=background_alpha,
                linewidths=0.0,
                rasterized=True,
            )

    scatter = ax.scatter(
        region_coords[:, 0],
        region_coords[:, 1],
        c=region_values,
        s=point_size,
        cmap=cmap,
        vmin=plot_vmin,
        vmax=plot_vmax,
        linewidths=0.0,
        rasterized=True,
    )
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title or f"{gene} [{layer}] {region_side} {region_fraction:.3f}", fontsize=12)
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(gene, rotation=90)
    fig.tight_layout()

    region_tag = f"{region_side}_{region_axis}_{str(region_threshold).replace('.', 'p')}"
    stem = f"{gene}_{layer}_{region_tag}"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "input_h5ad": str(Path(input_h5ad).resolve()),
        "layer": layer,
        "gene": gene,
        "region_axis": region_axis,
        "region_side": region_side,
        "region_fraction": float(region_fraction),
        "threshold": float(region_threshold),
        "n_obs_total": int(adata.n_obs),
        "n_obs_region": int(mask.sum()),
        "value_min_region": float(np.nanmin(region_values)),
        "value_max_region": float(np.nanmax(region_values)),
        "value_mean_region": float(np.nanmean(region_values)),
        "value_median_region": float(np.nanmedian(region_values)),
        "clip_quantile": float(clip_quantile),
        "plot_vmin": plot_vmin,
        "plot_vmax": plot_vmax,
        "png": str(png_path),
        "pdf": str(pdf_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    plot_layer_gene_on_spatial_region(
        input_h5ad=input_h5ad,
        layer=layer,
        gene=gene,
        output_dir=output_dir,
        title=title,
        point_size=point_size,
        background_point_size=background_point_size,
        dpi=dpi,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        clip_quantile=clip_quantile,
        region_axis=region_axis,
        region_side=region_side,
        region_fraction=region_fraction,
        threshold=threshold,
        show_background=show_background,
        background_color=background_color,
        background_alpha=background_alpha,
    )
