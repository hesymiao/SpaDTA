from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import scanpy as sc


project_root = Path("/data/user/hesy/projects/SpatialMETA")
sample_name = "m3_FMP"
run_root = project_root / "SpaDTA_718" / "runs" / "sm_downstream"
input_h5ad = run_root / "inputs" / sample_name / f"{sample_name}_output.h5ad"
output_dir = run_root / "fig2b" / sample_name

contribution_st_key = "contribution_st_decalign_linear"
contribution_sm_key = "contribution_sm_decalign_linear"
embedding_homo_st_key = "X_emb_homo_st_decalign_linear"
embedding_homo_sm_key = "X_emb_homo_sm_decalign_linear"
embedding_homo_joint_key = "X_emb_homo_joint_decalign_linear"

plot_st_key = "contribution_st"
plot_sm_key = "contribution_sm"
plot_title = sample_name
img_key = "hires"
alpha_img = 0.10
spot_size = 1.5
dpi = 220


def rowwise_cosine(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    x1 = np.asarray(x1, dtype=np.float32)
    x2 = np.asarray(x2, dtype=np.float32)
    dot = np.sum(x1 * x2, axis=1)
    denom = np.linalg.norm(x1, axis=1) * np.linalg.norm(x2, axis=1)
    cosine = np.divide(
        dot,
        denom,
        out=np.zeros_like(dot, dtype=np.float32),
        where=denom > 1e-12,
    )
    return np.clip(cosine, -1.0, 1.0)


def angular_similarity_from_cos(cosine: np.ndarray) -> np.ndarray:
    cosine = np.clip(np.asarray(cosine, dtype=np.float32), -1.0, 1.0)
    return 1.0 - (np.arccos(cosine) / np.pi)


def compute_spatialmeta_like_contributions(
    homo_st: np.ndarray,
    homo_sm: np.ndarray,
    homo_joint: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cos_st = rowwise_cosine(homo_st, homo_joint)
    cos_sm = rowwise_cosine(homo_sm, homo_joint)
    angular_st = angular_similarity_from_cos(cos_st)
    angular_sm = angular_similarity_from_cos(cos_sm)
    contribution_st = np.clip(angular_st - angular_sm + 0.5, 0.0, 1.0)
    contribution_sm = 1.0 - contribution_st
    return contribution_st.astype(np.float32), contribution_sm.astype(np.float32)


def ensure_contribution_columns(adata: sc.AnnData) -> None:
    has_obs = contribution_st_key in adata.obs.columns and contribution_sm_key in adata.obs.columns
    if has_obs:
        adata.obs[plot_st_key] = adata.obs[contribution_st_key].astype(float).to_numpy()
        adata.obs[plot_sm_key] = adata.obs[contribution_sm_key].astype(float).to_numpy()
        return

    required_obsm = (
        embedding_homo_st_key,
        embedding_homo_sm_key,
        embedding_homo_joint_key,
    )
    if not all(key in adata.obsm for key in required_obsm):
        raise KeyError(
            "Missing contribution columns and missing homo/joint embeddings required to recompute them: "
            f"{required_obsm}"
        )

    contribution_st, contribution_sm = compute_spatialmeta_like_contributions(
        homo_st=np.asarray(adata.obsm[embedding_homo_st_key], dtype=np.float32),
        homo_sm=np.asarray(adata.obsm[embedding_homo_sm_key], dtype=np.float32),
        homo_joint=np.asarray(adata.obsm[embedding_homo_joint_key], dtype=np.float32),
    )
    adata.obs[plot_st_key] = contribution_st
    adata.obs[plot_sm_key] = contribution_sm


def plot_spatial_contributions(
    input_h5ad: Path,
    output_dir: Path,
) -> dict[str, str]:
    if not input_h5ad.exists():
        raise FileNotFoundError(input_h5ad)

    output_dir.mkdir(parents=True, exist_ok=True)
    adata = sc.read_h5ad(input_h5ad)
    if "spatial" not in adata.obsm:
        raise KeyError("adata.obsm['spatial'] is required")
    if "spatial" not in adata.uns:
        raise KeyError("adata.uns['spatial'] is required for slice background plotting")

    ensure_contribution_columns(adata)

    contribution_cmap = LinearSegmentedColormap.from_list(
        "contribution_map",
        ["#2ec4b6", "#ffffff", "#ff9f1c"],
    )

    plt.close("all")
    sc.pl.spatial(
        adata,
        img_key=img_key,
        color=[plot_st_key, plot_sm_key],
        ncols=2,
        color_map=contribution_cmap,
        alpha_img=alpha_img,
        size=spot_size,
        show=False,
        title=[f"{plot_title} ST contribution", f"{plot_title} SM contribution"],
    )
    fig = plt.gcf()

    png_path = output_dir / "spatial_contributions.png"
    pdf_path = output_dir / "spatial_contributions.pdf"
    svg_path = output_dir / "spatial_contributions.svg"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight", format="svg")
    plt.close(fig)

    summary = {
        "input_h5ad": str(input_h5ad.resolve()),
        "output_dir": str(output_dir.resolve()),
        "st_plot_key": plot_st_key,
        "sm_plot_key": plot_sm_key,
        "st_value_min": float(np.nanmin(adata.obs[plot_st_key].astype(float).to_numpy())),
        "st_value_max": float(np.nanmax(adata.obs[plot_st_key].astype(float).to_numpy())),
        "sm_value_min": float(np.nanmin(adata.obs[plot_sm_key].astype(float).to_numpy())),
        "sm_value_max": float(np.nanmax(adata.obs[plot_sm_key].astype(float).to_numpy())),
        "png": str(png_path.resolve()),
        "pdf": str(pdf_path.resolve()),
        "svg": str(svg_path.resolve()),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    plot_spatial_contributions(
        input_h5ad=input_h5ad,
        output_dir=output_dir,
    )
