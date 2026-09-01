import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


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
obs_key_a = "contribution_st_decalign_linear"
obs_key_b = "contribution_sm_decalign_linear"
output_dir = input_h5ad.parent / f"{sample_name}_obs_diff_plots"
title = f"{sample_name} {obs_key_a} - {obs_key_b}"
point_size = 24.0
dpi = 220
cmap = "ylwhgn"


def plot_obs_diff_on_full_slice(
    input_h5ad: Path,
    obs_key_a: str,
    obs_key_b: str,
    output_dir: Path,
    title: str | None = None,
    point_size: float = 24.0,
    dpi: int = 220,
    cmap: str = "ylwhgn",
):
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(input_h5ad)
    if obs_key_a not in adata.obs.columns or obs_key_b not in adata.obs.columns:
        raise KeyError("Missing one of the requested obs keys.")
    if "spatial" not in adata.obsm:
        raise KeyError("adata.obsm['spatial'] is required")

    value_a = adata.obs[obs_key_a].astype(float).to_numpy()
    value_b = adata.obs[obs_key_b].astype(float).to_numpy()
    diff = value_a - value_b
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)

    lim = float(np.max(np.abs(diff)))
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)

    plot_cmap = cmap
    if plot_cmap == "ylwhgn":
        plot_cmap = LinearSegmentedColormap.from_list("ylwhgn", ["#f2c14e", "#fffdf6", "#18a87a"])
    elif plot_cmap == "ylwhgn_soft":
        plot_cmap = LinearSegmentedColormap.from_list("ylwhgn_soft", ["#e6c15b", "#fbfaf4", "#55b88d"])

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=diff,
        s=point_size,
        cmap=plot_cmap,
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
    ax.set_title(title or f"{obs_key_a} - {obs_key_b}", fontsize=12)
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(f"{obs_key_a} - {obs_key_b}", rotation=90)
    fig.tight_layout()

    stem = f"{obs_key_a}_minus_{obs_key_b}"
    png_path = output_dir / f"{stem}_full_slice.png"
    pdf_path = output_dir / f"{stem}_full_slice.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "input_h5ad": str(input_h5ad.resolve()),
        "obs_key_a": obs_key_a,
        "obs_key_b": obs_key_b,
        "n_spots": int(adata.n_obs),
        "diff_min": float(np.min(diff)),
        "diff_max": float(np.max(diff)),
        "diff_mean": float(np.mean(diff)),
        "diff_median": float(np.median(diff)),
        "png": str(png_path),
        "pdf": str(pdf_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    plot_obs_diff_on_full_slice(
        input_h5ad=input_h5ad,
        obs_key_a=obs_key_a,
        obs_key_b=obs_key_b,
        output_dir=output_dir,
        title=title,
        point_size=point_size,
        dpi=dpi,
        cmap=cmap,
    )
