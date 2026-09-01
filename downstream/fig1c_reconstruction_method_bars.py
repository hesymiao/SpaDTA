from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import h5py
from scipy import sparse


ROOT = Path("/data/user/hesy/projects/SpatialMETA")
SAMPLES = ("X49_T", "248_T", "m1_FMP")
OUTPUT_DIR = ROOT / "SpaDTA_718" / "runs" / "sm_downstream" / "fig1c" / "reconstruction_method_bars"

METHODS = {
    "SpaDTA": {
        "kind": "h5ad",
        "path": ROOT / "SpaDTA_718" / "runs" / "sm_downstream" / "inputs" / "{sample}" / "{sample}_output.h5ad",
        "layer": "reconstruction_decalign_linear",
    },
    "spaMultiVAE": {
        "kind": "spamultivae_txt",
        "input_path": ROOT / "compare_method" / "spaMultiVAE_official_runner" / "all9_20260717_float64" / "{sample}" / "input.h5",
        "gene_path": ROOT / "compare_method" / "spaMultiVAE_official_runner" / "all9_20260717_float64" / "{sample}" / "gene_denoised_counts.txt",
        "protein_path": ROOT / "compare_method" / "spaMultiVAE_official_runner" / "all9_20260717_float64" / "{sample}" / "protein_denoised_counts.txt",
        "248_T_input_path": ROOT / "compare_method" / "common" / "full_248_rerun_20260715" / "SpaPeakVAE" / "248_T" / "inputs" / "248_T_spamultivae_input.h5",
        "248_T_gene_path": ROOT / "compare_method" / "spaMultiVAE_official_runner" / "full_248_T_20260716" / "gene_denoised_counts.txt",
        "248_T_protein_path": ROOT / "compare_method" / "spaMultiVAE_official_runner" / "full_248_T_20260716" / "protein_denoised_counts.txt",
    },
    "SpatialMETA": {
        "kind": "h5ad",
        "path": ROOT / "compare_method" / "spatialmeta" / "runs" / "{sample}" / "{sample}_spatialmeta_domains.h5ad",
        "248_T_path": ROOT / "compare_method" / "common" / "full_248_rerun_20260715" / "SpatialMETA" / "248_T" / "248_T_spatialmeta_domains.h5ad",
        "layer": "reconstruction",
    },
    "totalVI": {
        "kind": "totalvi_npz",
        "path": ROOT / "compare_method" / "totalVI" / "runs" / "{sample}" / "{sample}_totalvi_reconstruction_joint.npz",
        "248_T_path": ROOT / "compare_method" / "common" / "full_248_rerun_20260715" / "totalVI" / "248_T" / "248_T_totalvi_reconstruction_joint.npz",
    },
}

METHOD_COLORS = {
    "SpaDTA": "#4F5D95",
    "SpatialMETA": "#8873A4",
    "totalVI": "#9E365C",
    "spaMultiVAE": "#2A9D8F",
}


def to_dense(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray(matrix.toarray(), dtype=np.float64)
    return np.asarray(matrix, dtype=np.float64)


def best_feature_names(adata: sc.AnnData, reference: pd.Index | None = None) -> pd.Index:
    index_names = pd.Index(adata.var_names.astype(str))
    if "name" not in adata.var.columns:
        return index_names
    column_names = pd.Index(adata.var["name"].astype(str))
    if column_names.has_duplicates:
        return index_names
    if reference is None:
        return column_names if index_names.str.fullmatch(r"\d+").mean() > 0.5 else index_names
    index_overlap = int(index_names.isin(reference).sum())
    column_overlap = int(column_names.isin(reference).sum())
    return column_names if column_overlap > index_overlap else index_names


def spatial_frame(adata: sc.AnnData, path: Path) -> pd.DataFrame:
    if "spatial" not in adata.obsm:
        raise ValueError(f"{path}: missing obsm['spatial'] for version audit")
    coordinates = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    return pd.DataFrame(coordinates, index=adata.obs_names.astype(str))


def read_h5ad_layer_df(
    path: Path, layer: str, reference: pd.Index | None = None
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    adata = sc.read_h5ad(path)
    try:
        names = best_feature_names(adata, reference)
        if names.has_duplicates:
            raise ValueError(f"{path}: duplicate feature names after normalization")
        matrix = to_dense(adata.layers[layer])
        frame = pd.DataFrame(matrix, index=adata.obs_names.astype(str), columns=names)
        if "type" in adata.var.columns:
            var_types = pd.Series(adata.var["type"].astype(str).to_numpy(), index=names)
        else:
            var_types = pd.Series(index=names, dtype=str)
        return frame, var_types, spatial_frame(adata, path)
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()


def read_totalvi_df(path: Path) -> tuple[pd.DataFrame, str, pd.DataFrame]:
    z = np.load(path, allow_pickle=True)
    export_mode = str(np.atleast_1d(z["export_mode"])[0])
    if export_mode != "decoder_reconstruction_for_loss":
        raise ValueError(f"legacy totalVI export: {export_mode}")
    obs_names = pd.Index(np.asarray(z["obs_names"]).astype(str))
    gene_names = pd.Index(np.asarray(z["gene_names"]).astype(str))
    protein_names = pd.Index(np.asarray(z["protein_names"]).astype(str))
    matrix = np.concatenate(
        [
            np.asarray(z["gene_reconstruction"], dtype=np.float64),
            np.asarray(z["protein_reconstruction"], dtype=np.float64),
        ],
        axis=1,
    )
    names = gene_names.append(protein_names)
    if names.has_duplicates:
        raise ValueError(f"{path}: duplicate totalVI feature names")
    input_path = path.with_name(path.name.replace("_reconstruction_joint.npz", "_input.h5ad"))
    input_adata = sc.read_h5ad(input_path)
    try:
        coordinates = spatial_frame(input_adata, input_path)
    finally:
        if getattr(input_adata, "file", None) is not None:
            input_adata.file.close()
    return pd.DataFrame(matrix, index=obs_names, columns=names), export_mode, coordinates


def read_spamultivae_df(
    input_path: Path,
    gene_path: Path,
    protein_path: Path,
    truth_index: pd.Index,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    with h5py.File(input_path, "r") as handle:
        gene_names = pd.Index(np.asarray(handle["gene_names"]).astype(str))
        protein_names = pd.Index(np.asarray(handle["protein_names"]).astype(str))
        coordinates = np.asarray(handle["pos"], dtype=np.float64)
    gene = np.loadtxt(gene_path, delimiter=",", ndmin=2).astype(np.float64)
    protein = np.loadtxt(protein_path, delimiter=",", ndmin=2).astype(np.float64)
    if gene.shape != (len(truth_index), len(gene_names)):
        raise ValueError(f"spaMultiVAE gene reconstruction shape {gene.shape} does not match {(len(truth_index), len(gene_names))}")
    if protein.shape != (len(truth_index), len(protein_names)):
        raise ValueError(f"spaMultiVAE protein reconstruction shape {protein.shape} does not match {(len(truth_index), len(protein_names))}")
    names = gene_names.append(protein_names)
    if names.has_duplicates:
        raise ValueError("spaMultiVAE has duplicate feature names")
    matrix = np.concatenate([gene, protein], axis=1)
    spatial = pd.DataFrame(coordinates, index=truth_index)
    return pd.DataFrame(matrix, index=truth_index, columns=names), spatial


def load_prediction(
    sample: str,
    method: str,
    cfg: dict[str, object],
    truth_columns: pd.Index,
    truth_index: pd.Index,
) -> tuple[pd.DataFrame, Path, str, pd.DataFrame]:
    path_template = cfg.get(f"{sample}_path", cfg.get("path"))
    path = Path(str(path_template).format(sample=sample)) if path_template is not None else Path(".")
    if cfg["kind"] != "spamultivae_txt" and not path.exists():
        raise FileNotFoundError(path)
    if cfg["kind"] == "h5ad":
        layer = str(cfg["layer"])
        frame, _, coordinates = read_h5ad_layer_df(path, layer, truth_columns)
        return frame, path, layer, coordinates
    if cfg["kind"] == "totalvi_npz":
        pred_df, export_mode, coordinates = read_totalvi_df(path)
        return pred_df, path, export_mode, coordinates
    if cfg["kind"] == "spamultivae_txt":
        input_path = Path(str(cfg.get(f"{sample}_input_path", cfg["input_path"])).format(sample=sample))
        gene_path = Path(str(cfg.get(f"{sample}_gene_path", cfg["gene_path"])).format(sample=sample))
        protein_path = Path(str(cfg.get(f"{sample}_protein_path", cfg["protein_path"])).format(sample=sample))
        pred_df, coordinates = read_spamultivae_df(input_path, gene_path, protein_path, truth_index)
        return pred_df, gene_path, "gene/protein_denoised_counts", coordinates
    raise ValueError(f"Unsupported reconstruction source for {method}: {cfg['kind']}")


def mean_feature_pcc(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth_centered = truth - truth.mean(axis=0, keepdims=True)
    pred_centered = prediction - prediction.mean(axis=0, keepdims=True)
    denom = np.sqrt(np.sum(truth_centered**2, axis=0) * np.sum(pred_centered**2, axis=0))
    valid = denom > 1e-12
    if not np.any(valid):
        return float("nan")
    values = np.sum(truth_centered[:, valid] * pred_centered[:, valid], axis=0) / denom[valid]
    return float(np.mean(values))


def mean_feature_cosine(truth: np.ndarray, prediction: np.ndarray) -> float:
    denom = np.linalg.norm(truth, axis=0) * np.linalg.norm(prediction, axis=0)
    valid = denom > 1e-12
    if not np.any(valid):
        return float("nan")
    values = np.sum(truth[:, valid] * prediction[:, valid], axis=0) / denom[valid]
    return float(np.mean(values))


def evaluate_sample(sample: str) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    truth_path = Path(str(METHODS["SpaDTA"]["path"]).format(sample=sample))
    truth_df, var_types, truth_spatial = read_h5ad_layer_df(truth_path, "normalized")

    rows: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []
    predictions: dict[str, tuple[pd.DataFrame, Path, str, pd.DataFrame]] = {}
    for method, cfg in METHODS.items():
        try:
            prediction = load_prediction(
                sample,
                method,
                cfg,
                truth_df.columns,
                truth_df.index,
            )
            pred_df, _, _, pred_spatial = prediction
            if len(pred_df) != len(truth_df):
                raise ValueError(
                    f"dataset version mismatch: prediction has {len(pred_df)} spots, truth has {len(truth_df)}"
                )
            if not pred_df.index.equals(truth_df.index):
                raise ValueError("dataset version mismatch: spot names/order differ from truth")
            if not pred_spatial.index.equals(truth_spatial.index):
                raise ValueError("dataset version mismatch: coordinate spot names/order differ from truth")
            if pred_spatial.shape != truth_spatial.shape or not np.allclose(
                pred_spatial.to_numpy(), truth_spatial.to_numpy(), rtol=0.0, atol=1e-6
            ):
                max_difference = float(
                    np.max(np.abs(pred_spatial.to_numpy() - truth_spatial.to_numpy()))
                )
                raise ValueError(
                    f"dataset version mismatch: spatial coordinates differ (max abs difference={max_difference:.6g})"
                )
            predictions[method] = prediction
        except Exception as exc:
            skipped.append(
                {
                    "sample": sample,
                    "method": method,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )

    if not predictions:
        return rows, skipped

    common_obs = truth_df.index
    common_vars = truth_df.columns
    for pred_df, _, _, _ in predictions.values():
        common_obs = common_obs.intersection(pred_df.index, sort=False)
        common_vars = common_vars.intersection(pred_df.columns, sort=False)
    st_columns = common_vars[var_types.reindex(common_vars).eq("ST").to_numpy()]
    sm_columns = common_vars[var_types.reindex(common_vars).eq("SM").to_numpy()]
    if len(common_obs) == 0 or len(st_columns) == 0 or len(sm_columns) == 0:
        raise ValueError(
            f"{sample}: empty shared evaluation space: spots={len(common_obs)}, ST={len(st_columns)}, SM={len(sm_columns)}"
        )

    truth_st = truth_df.loc[common_obs, st_columns].to_numpy(dtype=np.float64)
    truth_sm = truth_df.loc[common_obs, sm_columns].to_numpy(dtype=np.float64)
    truth_joint = np.concatenate([truth_st, truth_sm], axis=1)
    for method, (pred_df, source_path, source_layer, _) in predictions.items():
        try:
            pred_st = pred_df.loc[common_obs, st_columns].to_numpy(dtype=np.float64)
            pred_sm = pred_df.loc[common_obs, sm_columns].to_numpy(dtype=np.float64)
            pred_joint = np.concatenate([pred_st, pred_sm], axis=1)
            row = {
                "sample": sample,
                "method": method,
                "n_spots": int(len(common_obs)),
                "n_st_features": int(len(st_columns)),
                "n_sm_features": int(len(sm_columns)),
                "n_features": int(len(common_vars)),
                "st_feature_pcc_mean": mean_feature_pcc(truth_st, pred_st),
                "sm_feature_pcc_mean": mean_feature_pcc(truth_sm, pred_sm),
                "joint_feature_pcc_mean": mean_feature_pcc(truth_joint, pred_joint),
                "st_feature_cosine_mean": mean_feature_cosine(truth_st, pred_st),
                "sm_feature_cosine_mean": mean_feature_cosine(truth_sm, pred_sm),
                "joint_feature_cosine_mean": mean_feature_cosine(truth_joint, pred_joint),
                "evaluation_space": "current_spadta_normalized_shared_spots_and_features",
                "reconstruction_source": str(source_path.resolve()),
                "reconstruction_layer_or_mode": source_layer,
            }
            row["feature_pcc_mean"] = (
                row["st_feature_pcc_mean"] + row["sm_feature_pcc_mean"]
            ) / 2.0
            row["feature_cosine_mean"] = (
                row["st_feature_cosine_mean"] + row["sm_feature_cosine_mean"]
            ) / 2.0
            rows.append(row)
            print(
                f"[fig1c-bars] {sample} {method}: "
                f"PCC={row['feature_pcc_mean']:.4f}, cosine={row['feature_cosine_mean']:.4f}",
                flush=True,
            )
        except Exception as exc:
            skipped.append(
                {
                    "sample": sample,
                    "method": method,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[fig1c-bars] skip {sample} {method}: {exc}", flush=True)
    return rows, skipped


def add_bars(ax: plt.Axes, data: pd.DataFrame, metric: str, title: str) -> None:
    data = data.loc[data[metric].notna()].copy()
    if data.empty:
        ax.text(0.5, 0.5, "No valid reconstruction", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=20, fontweight="bold")
        ax.set_xticks([])
        return
    methods = data["method"].tolist()
    values = data[metric].astype(float).to_numpy()
    x = np.arange(len(methods))
    bars = ax.bar(x, values, width=0.56, color=[METHOD_COLORS.get(method, "#888888") for method in methods])
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=24, ha="right", fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    for tick, method in zip(ax.get_xticklabels(), methods):
        tick.set_color(METHOD_COLORS.get(method, "#222222"))
        tick.set_fontweight("bold")
    y_min = min(0.0, float(np.min(values)) - 0.05)
    y_max = min(1.05, max(0.7, float(np.max(values)) + 0.12))
    ax.set_ylim(y_min, y_max)
    ax.set_title(title, fontsize=20, fontweight="bold")
    ax.grid(axis="y", alpha=0.2, linewidth=0.8)
    ax.set_axisbelow(True)
    if y_min < 0:
        ax.axhline(0.0, color="#666666", linewidth=0.8)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.018,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=12,
        )


def save_per_sample_figures(metrics: pd.DataFrame) -> None:
    for sample in SAMPLES:
        sample_df = metrics.loc[metrics["sample"].eq(sample)].copy()
        sample_dir = OUTPUT_DIR / sample
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_df.to_csv(sample_dir / f"{sample}_reconstruction_metric_averages.csv", index=False)
        for metric, title, stem in (
            ("feature_pcc_mean", "PCC", "feature_mean_pcc_by_method"),
            ("feature_cosine_mean", "Cosine Similarity", "feature_mean_cosine_by_method"),
        ):
            fig, ax = plt.subplots(figsize=(5.4, 6.2), dpi=200)
            add_bars(ax, sample_df, metric, title)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            output_path = sample_dir / f"{sample}_{stem}.png"
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
            plt.close(fig)


def save_overview(metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(len(SAMPLES), 2, figsize=(12, 5.4 * len(SAMPLES)), squeeze=False)
    for row, sample in enumerate(SAMPLES):
        sample_df = metrics.loc[metrics["sample"].eq(sample)].copy()
        add_bars(axes[row, 0], sample_df, "feature_pcc_mean", "PCC")
        add_bars(axes[row, 1], sample_df, "feature_cosine_mean", "Cosine Similarity")
    fig.suptitle("Reconstruction accuracy across methods", fontsize=25, fontweight="bold", y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    output_path = OUTPUT_DIR / "three_samples_reconstruction_feature_mean_bars.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []
    for sample in SAMPLES:
        sample_rows, sample_skipped = evaluate_sample(sample)
        rows.extend(sample_rows)
        skipped.extend(sample_skipped)

    metrics = pd.DataFrame(rows)
    if metrics.empty:
        raise RuntimeError("No reconstruction metrics were computed")
    metrics_path = OUTPUT_DIR / "reconstruction_feature_mean_metrics.csv"
    averages_path = OUTPUT_DIR / "reconstruction_metric_averages.csv"
    skipped_path = OUTPUT_DIR / "skipped_methods.csv"
    metrics.to_csv(metrics_path, index=False)
    metrics[
        [
            "sample",
            "method",
            "n_spots",
            "n_st_features",
            "n_sm_features",
            "st_feature_pcc_mean",
            "sm_feature_pcc_mean",
            "feature_pcc_mean",
            "st_feature_cosine_mean",
            "sm_feature_cosine_mean",
            "feature_cosine_mean",
        ]
    ].to_csv(averages_path, index=False)
    pd.DataFrame(skipped, columns=["sample", "method", "reason"]).to_csv(skipped_path, index=False)
    save_per_sample_figures(metrics)
    save_overview(metrics)
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "samples": list(SAMPLES),
                "methods_with_reconstruction": list(METHODS),
                "evaluation_space": "truth=current SpaDTA input layers['normalized']; shared spots and features across methods",
                "bar_definition": "mean(ST feature-wise metric, SM feature-wise metric)",
                "metrics_csv": str(metrics_path.resolve()),
                "averages_csv": str(averages_path.resolve()),
                "skipped_csv": str(skipped_path.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[fig1c-bars] wrote {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
