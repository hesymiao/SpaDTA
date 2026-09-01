import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


project_root = Path("/data/user/hesy/projects/SpatialMETA")
sample_name = "Y27_T"
eval_h5ad = project_root / "compare_method" / "ours" / "runs" / sample_name / "sm_to_st_eval.h5ad"
background_h5ad = project_root / "compare_method" / "ours" / "runs" / sample_name / f"{sample_name}_ours_domains.h5ad"
true_layer = "true_st"
pred_layer = "generated_st"
features = ["Plp1", "Mbp", "Gfap"]
output_dir = project_root / "SpaDTA_718" / "runs" / "downstream_runs" / f"{sample_name}_generated_vs_true"
title = f"{sample_name} generated vs true"
background_point_size = 24.0
foreground_point_size = 30.0
dpi = 220
cmap = "viridis"


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


def plot_generated_vs_true_features_on_slice(
    eval_h5ad: Path,
    background_h5ad: Path,
    true_layer: str,
    pred_layer: str,
    features: list[str],
    output_dir: Path,
    title: str | None = None,
    background_point_size: float = 24.0,
    foreground_point_size: float = 30.0,
    dpi: int = 220,
    cmap: str = "viridis",
):
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_adata = sc.read_h5ad(eval_h5ad)
    bg_adata = sc.read_h5ad(background_h5ad)
    if "spatial" not in eval_adata.obsm or "spatial" not in bg_adata.obsm:
        raise ValueError("Both eval and background h5ad must contain obsm['spatial'].")

    eval_adata.obs_names = eval_adata.obs_names.astype(str)
    bg_adata.obs_names = bg_adata.obs_names.astype(str)

    y_true = to_dense_float32(eval_adata.layers[true_layer])
    y_pred = to_dense_float32(eval_adata.layers[pred_layer])
    eval_coords = np.asarray(eval_adata.obsm["spatial"], dtype=np.float32)
    bg_coords = np.asarray(bg_adata.obsm["spatial"], dtype=np.float32)
    spot_pcc = per_spot_pcc(y_true, y_pred)

    feature_rows = []
    valid_features = []
    feature_indices = []
    for feature in features:
        index = np.where(eval_adata.var_names.astype(str) == str(feature))[0]
        if len(index) == 0:
            continue
        valid_features.append(str(feature))
        feature_indices.append(int(index[0]))
    if not valid_features:
        raise ValueError("None of the requested features were found in eval_adata.var_names.")

    n_cols = len(valid_features)
    fig, axes = plt.subplots(2, n_cols, figsize=(4.5 * n_cols, 8.2), squeeze=False)

    for col_idx, (feature, feat_idx) in enumerate(zip(valid_features, feature_indices)):
        true_values = y_true[:, feat_idx]
        pred_values = y_pred[:, feat_idx]
        vmax_plot = float(np.quantile(np.concatenate([true_values, pred_values]), 0.99))
        vmax_plot = max(vmax_plot, 1e-6)

        for row_idx, (values, row_label) in enumerate(((true_values, "True"), (pred_values, "Generated"))):
            ax = axes[row_idx, col_idx]
            ax.scatter(
                bg_coords[:, 0],
                bg_coords[:, 1],
                s=background_point_size,
                c="#ddd8ea",
                alpha=0.9,
                linewidths=0.0,
                zorder=1,
            )
            scatter = ax.scatter(
                eval_coords[:, 0],
                eval_coords[:, 1],
                c=values,
                s=foreground_point_size,
                cmap=cmap,
                vmin=0.0,
                vmax=vmax_plot,
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
            if row_idx == 0:
                ax.set_title(feature, fontsize=12)
            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=12)
            colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.02)
            colorbar.ax.tick_params(labelsize=8)

        feature_corr = np.nan
        if np.std(true_values) >= 1e-8 and np.std(pred_values) >= 1e-8:
            feature_corr = float(np.corrcoef(true_values, pred_values)[0, 1])
        feature_rows.append(
            {
                "feature": feature,
                "feature_corr": feature_corr,
                "true_mean": float(np.mean(true_values)),
                "pred_mean": float(np.mean(pred_values)),
                "vmax_plot": vmax_plot,
            }
        )
        axes[1, col_idx].text(
            0.02,
            0.98,
            f"Feature PCC={feature_corr:.3f}",
            transform=axes[1, col_idx].transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 2},
        )

    fig.suptitle(title or eval_h5ad.stem, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    png_path = output_dir / "generated_vs_true_on_slice.png"
    pdf_path = output_dir / "generated_vs_true_on_slice.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    feature_df = pd.DataFrame(feature_rows)
    feature_csv = output_dir / "feature_plot_summary.csv"
    feature_df.to_csv(feature_csv, index=False)

    summary = {
        "eval_h5ad": str(eval_h5ad.resolve()),
        "background_h5ad": str(background_h5ad.resolve()),
        "true_layer": true_layer,
        "pred_layer": pred_layer,
        "features": valid_features,
        "n_eval_spots": int(eval_adata.n_obs),
        "n_background_spots": int(bg_adata.n_obs),
        "spot_pcc_mean": float(np.nanmean(spot_pcc)),
        "feature_plot_summary_csv": str(feature_csv),
        "png": str(png_path),
        "pdf": str(pdf_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    plot_generated_vs_true_features_on_slice(
        eval_h5ad=eval_h5ad,
        background_h5ad=background_h5ad,
        true_layer=true_layer,
        pred_layer=pred_layer,
        features=features,
        output_dir=output_dir,
        title=title,
        background_point_size=background_point_size,
        foreground_point_size=foreground_point_size,
        dpi=dpi,
        cmap=cmap,
    )
