import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from matplotlib.colors import TwoSlopeNorm


project_root = Path("/data/user/hesy/projects/SpatialMETA")
sample_name = "Y27_T"
run_name = "current_localcontext_contrastive_w002_fixed_targetcount_20260427_localized"
config_name = "ablate_no_hete_homo_splitrecon_sharedhalf_balance"
input_h5ad = (
    project_root
    / "SpaDTA_718"
    / "runs"
    / "model_runs"
    / run_name
    / config_name
    / f"{sample_name}_{config_name}.h5ad"
)
obs_key = "contribution_st_decalign_linear"
output_dir = input_h5ad.parent / f"{sample_name}_obs_value_plots"
title = f"{sample_name} {obs_key}"
point_size = 24.0
dpi = 220
cmap = "coolwarm"
center_zero = False
vmin = None
vmax = None


def plot_obs_value_on_full_slice(
    input_h5ad: Path,
    obs_key: str,
    output_dir: Path,
    title: str | None = None,
    point_size: float = 24.0,
    dpi: int = 220,
    cmap: str = "coolwarm",
    center_zero: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(input_h5ad)
    if obs_key not in adata.obs.columns:
        raise KeyError(f"Missing obs key: {obs_key}")
    if "spatial" not in adata.obsm:
        raise KeyError("adata.obsm['spatial'] is required")

    values = adata.obs[obs_key].astype(float).to_numpy()
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)

    plot_vmin = float(np.nanmin(values)) if vmin is None else float(vmin)
    plot_vmax = float(np.nanmax(values)) if vmax is None else float(vmax)
    if center_zero:
        lim = max(abs(plot_vmin), abs(plot_vmax))
        plot_vmin, plot_vmax = -lim, lim
        norm = TwoSlopeNorm(vmin=plot_vmin, vcenter=0.0, vmax=plot_vmax)
    else:
        norm = None

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=values,
        s=point_size,
        cmap=cmap,
        vmin=None if norm is not None else plot_vmin,
        vmax=None if norm is not None else plot_vmax,
        norm=norm,
        linewidths=0.15,
        edgecolors="white",
        rasterized=True,
    )
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title or obs_key, fontsize=12)
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(obs_key, rotation=90)
    fig.tight_layout()

    png_path = output_dir / f"{obs_key}_full_slice.png"
    pdf_path = output_dir / f"{obs_key}_full_slice.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "input_h5ad": str(input_h5ad.resolve()),
        "obs_key": obs_key,
        "n_spots": int(adata.n_obs),
        "value_min": float(np.nanmin(values)),
        "value_max": float(np.nanmax(values)),
        "value_mean": float(np.nanmean(values)),
        "value_median": float(np.nanmedian(values)),
        "png": str(png_path),
        "pdf": str(pdf_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    plot_obs_value_on_full_slice(
        input_h5ad=input_h5ad,
        obs_key=obs_key,
        output_dir=output_dir,
        title=title,
        point_size=point_size,
        dpi=dpi,
        cmap=cmap,
        center_zero=center_zero,
        vmin=vmin,
        vmax=vmax,
    )
