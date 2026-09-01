import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib import pyplot as plt
from scipy import sparse


project_root = Path("/data/user/hesy/projects/SpatialMETA")
sample_name = "Y27_T"
eval_h5ad = project_root / "compare_method" / "ours" / "runs" / sample_name / "sm_to_st_eval.h5ad"
background_h5ad = project_root / "compare_method" / "ours" / "runs" / sample_name / f"{sample_name}_ours_domains.h5ad"
pred_layer = "generated_st"
true_layer = "true_st"
output_dir = project_root / "SpaDTA_718" / "runs" / "downstream_runs" / f"{sample_name}_spot_pcc"
title = f"{sample_name} spot PCC"
point_size = 22.0
background_point_size = 18.0
cmap = "viridis"
vmin = -1.0
vmax = 1.0
dpi = 220


def to_dense_float32(values):
    if sparse.issparse(values):
        return values.toarray().astype(np.float32, copy=False)
    return np.asarray(values, dtype=np.float32)


def per_spot_pcc(y_true, y_pred):
    result = np.full(y_true.shape[0], np.nan, dtype=np.float32)
    for idx in range(y_true.shape[0]):
        current_true = y_true[idx]
        current_pred = y_pred[idx]
        if np.std(current_true) < 1e-8 or np.std(current_pred) < 1e-8:
            continue
        corr = np.corrcoef(current_true, current_pred)[0, 1]
        if np.isfinite(corr):
            result[idx] = float(corr)
    return result


def plot_spot_pcc_on_slice(
    eval_h5ad: Path,
    output_dir: Path,
    pred_layer: str,
    true_layer: str,
    background_h5ad: Path | None = None,
    title: str | None = None,
    point_size: float = 22.0,
    background_point_size: float = 18.0,
    cmap: str = "viridis",
    vmin: float = -1.0,
    vmax: float = 1.0,
    dpi: int = 220,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(eval_h5ad)
    if "spatial" not in adata.obsm:
        raise ValueError("input h5ad must contain adata.obsm['spatial']")
    if pred_layer not in adata.layers:
        raise ValueError(f"missing pred layer: {pred_layer}")
    if true_layer not in adata.layers:
        raise ValueError(f"missing true layer: {true_layer}")

    y_pred = to_dense_float32(adata.layers[pred_layer])
    y_true = to_dense_float32(adata.layers[true_layer])
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    pcc = per_spot_pcc(y_true, y_pred)

    full_coords = None
    if background_h5ad is not None:
        background_adata = sc.read_h5ad(background_h5ad)
        if "spatial" not in background_adata.obsm:
            raise ValueError("background h5ad must contain adata.obsm['spatial']")
        full_coords = np.asarray(background_adata.obsm["spatial"], dtype=np.float32)

    df = pd.DataFrame(
        {
            "obs_name": adata.obs_names.astype(str),
            "spot_name": adata.obs["spot_name"].astype(str).to_numpy() if "spot_name" in adata.obs.columns else adata.obs_names.astype(str),
            "x": coords[:, 0],
            "y": coords[:, 1],
            "spot_pcc": pcc,
        }
    )
    csv_path = output_dir / "spot_pcc.csv"
    df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    if full_coords is not None:
        ax.scatter(
            full_coords[:, 0],
            full_coords[:, 1],
            s=background_point_size,
            c="#d7d3ea",
            alpha=0.9,
            linewidths=0.0,
            zorder=1,
        )
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=pcc,
        s=point_size,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.15,
        edgecolors="white",
        zorder=2,
    )
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title or eval_h5ad.stem, fontsize=11)
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Spot PCC", rotation=90)
    fig.tight_layout()

    png_path = output_dir / "spot_pcc_on_slice.png"
    pdf_path = output_dir / "spot_pcc_on_slice.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "input_h5ad": str(eval_h5ad.resolve()),
        "background_h5ad": str(background_h5ad.resolve()) if background_h5ad is not None else None,
        "pred_layer": pred_layer,
        "true_layer": true_layer,
        "n_spots": int(adata.n_obs),
        "n_background_spots": int(full_coords.shape[0]) if full_coords is not None else None,
        "n_valid_spots": int(np.isfinite(pcc).sum()),
        "spot_pcc_mean": float(np.nanmean(pcc)),
        "spot_pcc_median": float(np.nanmedian(pcc)),
        "spot_pcc_min": float(np.nanmin(pcc)),
        "spot_pcc_max": float(np.nanmax(pcc)),
        "spot_pcc_csv": str(csv_path),
        "png": str(png_path),
        "pdf": str(pdf_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    plot_spot_pcc_on_slice(
        eval_h5ad=eval_h5ad,
        output_dir=output_dir,
        pred_layer=pred_layer,
        true_layer=true_layer,
        background_h5ad=background_h5ad,
        title=title,
        point_size=point_size,
        background_point_size=background_point_size,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        dpi=dpi,
    )
