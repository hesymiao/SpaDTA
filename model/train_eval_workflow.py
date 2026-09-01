from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch

from .cluster_eval_utils import compute_metrics, load_gt_from_annotation_csv, load_gt_from_h5ad
from .model import DecAlignSpatialMetaLinear
from .preprocess import validate_spadta_model_input
from .workflow import seed_everything


DEFAULT_RSCRIPT = Path("/data/user/hesy/miniconda3/envs/renv/bin/Rscript")


def normalize_train_overrides(train_overrides: dict[str, object] | None) -> dict[str, object]:
    normalized = {} if train_overrides is None else dict(train_overrides)
    if "spatial_fourier_scales" in normalized and isinstance(normalized["spatial_fourier_scales"], list):
        normalized["spatial_fourier_scales"] = tuple(float(value) for value in normalized["spatial_fourier_scales"])
    return normalized


def serialize_table_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return json.dumps(list(value), ensure_ascii=False)
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def pca_project_local(matrix: np.ndarray, n_components: int) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"expected a 2D embedding matrix, got shape={values.shape}")
    centered = values - values.mean(axis=0, keepdims=True)
    max_components = min(centered.shape[0], centered.shape[1])
    if max_components < 1:
        raise ValueError("embedding matrix is empty")
    use_components = max(1, min(int(n_components), max_components))
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:use_components].T


def normalize_branch_mean_variance(branch: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(branch, dtype=np.float64)
    centered = values - values.mean(axis=0, keepdims=True)
    per_dimension_variance = np.var(centered, axis=0, ddof=0)
    mean_dimension_variance = float(per_dimension_variance.mean())
    if mean_dimension_variance <= eps:
        raise ValueError(f"Branch mean dimension variance is too small: {mean_dimension_variance}")
    scale = float(np.sqrt(mean_dimension_variance))
    return centered / scale


def build_branch_scaled_full(
    q_mu_shared: np.ndarray,
    q_mu_st: np.ndarray,
    q_mu_sm: np.ndarray,
    *,
    shared_weight: float = 1.0,
    st_weight: float = 1.0,
    sm_weight: float = 1.0,
) -> np.ndarray:
    return np.concatenate(
        [
            float(shared_weight) * normalize_branch_mean_variance(q_mu_shared),
            float(st_weight) * normalize_branch_mean_variance(q_mu_st),
            float(sm_weight) * normalize_branch_mean_variance(q_mu_sm),
        ],
        axis=1,
    )


def build_raw_full_q_mu(q_mu_shared: np.ndarray, q_mu_st: np.ndarray, q_mu_sm: np.ndarray) -> np.ndarray:
    return np.concatenate([q_mu_shared, q_mu_st, q_mu_sm], axis=1)


def build_eval_embedding(
    *,
    q_mu_shared: np.ndarray,
    q_mu_st: np.ndarray,
    q_mu_sm: np.ndarray,
    embedding_eval_mode: str,
    branch_scaled_shared_weight: float = 1.0,
    branch_scaled_st_weight: float = 1.0,
    branch_scaled_sm_weight: float = 1.0,
) -> tuple[str, np.ndarray]:
    mode = str(embedding_eval_mode).strip().lower()
    if mode == "branch_scaled_full":
        return mode, build_branch_scaled_full(
            q_mu_shared=q_mu_shared,
            q_mu_st=q_mu_st,
            q_mu_sm=q_mu_sm,
            shared_weight=float(branch_scaled_shared_weight),
            st_weight=float(branch_scaled_st_weight),
            sm_weight=float(branch_scaled_sm_weight),
        )
    if mode == "raw_full_q_mu":
        return mode, build_raw_full_q_mu(q_mu_shared=q_mu_shared, q_mu_st=q_mu_st, q_mu_sm=q_mu_sm)
    if mode == "q_mu_shared":
        return mode, np.asarray(q_mu_shared, dtype=np.float64)
    raise ValueError(
        f"Unsupported embedding_eval_mode={embedding_eval_mode!r}; "
        "expected one of {'branch_scaled_full', 'raw_full_q_mu', 'q_mu_shared'}."
    )


def summarize_branch_variance(branch: np.ndarray) -> dict[str, float]:
    values = np.asarray(branch, dtype=np.float64)
    centered = values - values.mean(axis=0, keepdims=True)
    per_dimension_variance = np.var(centered, axis=0, ddof=0)
    return {
        "total_variance": float(per_dimension_variance.sum()),
        "mean_dimension_variance": float(per_dimension_variance.mean()),
        "mean_vector_norm": float(np.linalg.norm(centered, axis=1).mean()),
    }


def resolve_rscript(rscript: Path | None) -> Path:
    candidates: list[Path] = []
    if rscript is not None:
        candidates.append(Path(rscript))
    candidates.extend(
        [
            DEFAULT_RSCRIPT,
            Path("/data/user/hesy/miniconda3/envs/stabmap_official_r42/bin/Rscript"),
            Path("/data/user/hesy/miniconda3/envs/spanjy/bin/Rscript"),
            Path("/data/user/hesy/miniconda3/envs/unitcr/bin/Rscript"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Rscript not found; pass --rscript explicitly")


def run_mclust_fixed_k(
    points: np.ndarray,
    n_clusters: int,
    random_seed: int,
    rscript: Path,
    work_dir: Path,
    model_name: str,
) -> np.ndarray:
    if points.shape[0] < n_clusters:
        raise ValueError("number of clusters exceeds number of observations")

    work_dir.mkdir(parents=True, exist_ok=True)
    csv_handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".csv",
        prefix="mclust_input_",
        dir=work_dir,
        delete=False,
    )
    out_handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".csv",
        prefix="mclust_output_",
        dir=work_dir,
        delete=False,
    )
    script_handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".R",
        prefix="mclust_run_",
        dir=work_dir,
        delete=False,
    )
    csv_path = Path(csv_handle.name)
    out_path = Path(out_handle.name)
    script_path = Path(script_handle.name)
    csv_handle.close()
    out_handle.close()
    script_handle.close()

    try:
        pd.DataFrame(points).to_csv(csv_path, index=False)
        script_path.write_text(
            """
args <- commandArgs(trailingOnly=TRUE)
if (length(args) < 5) {
  stop("expected 5 trailing arguments for mclust invocation")
}
input_csv <- args[1]
output_csv <- args[2]
num_cluster <- as.integer(args[3])
model_name <- args[4]
seed_value <- as.integer(args[5])
suppressPackageStartupMessages(library(mclust))
set.seed(seed_value)
dat <- read.csv(input_csv, check.names=FALSE)
if (nrow(dat) < num_cluster) {
  stop("number of rows is smaller than requested clusters")
}
res <- Mclust(dat, G=num_cluster, modelNames=model_name)
if (is.null(res$classification) || length(res$classification) == 0) {
  stop("mclust returned empty classification")
}
out_df <- data.frame(cluster=as.integer(res$classification))
write.csv(out_df, file=output_csv, row.names=FALSE, quote=FALSE)
""",
            encoding="utf-8",
        )
        completed = subprocess.run(
            [
                os.fspath(rscript),
                os.fspath(script_path),
                os.fspath(csv_path),
                os.fspath(out_path),
                str(int(n_clusters)),
                str(model_name),
                str(int(random_seed)),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if not out_path.exists() or out_path.stat().st_size == 0:
            stderr = completed.stderr.strip() if completed.stderr else ""
            stdout = completed.stdout.strip() if completed.stdout else ""
            raise RuntimeError(
                "mclust finished without writing a usable output file; "
                f"stdout={stdout!r} stderr={stderr!r}"
            )
        result = pd.read_csv(out_path)
        if "cluster" not in result.columns:
            raise RuntimeError("mclust output is missing 'cluster' column")
        labels = result["cluster"].astype(int).to_numpy()
        if labels.shape[0] != points.shape[0]:
            raise RuntimeError("mclust returned a different number of labels than input rows")
        return labels
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ""
        stdout = exc.stdout.strip() if exc.stdout else ""
        raise RuntimeError(f"mclust failed via {rscript}: {stderr or stdout or exc}") from exc
    finally:
        for path in (csv_path, out_path, script_path):
            if path.exists():
                path.unlink()


def build_train_kwargs(
    *,
    base_train_kwargs: dict[str, object],
    device: str,
    cluster_resolution: float,
    max_epoch: int,
    n_per_batch: int,
    train_overrides: dict[str, object] | None,
) -> dict[str, object]:
    train_kwargs = dict(base_train_kwargs)
    train_kwargs["device"] = device
    train_kwargs["cluster_resolution"] = float(cluster_resolution)
    train_kwargs["max_epoch"] = int(max_epoch)
    train_kwargs["n_per_batch"] = int(n_per_batch)
    train_kwargs.update(normalize_train_overrides(train_overrides))
    train_kwargs.setdefault("spatial_contrastive_early_stop_enabled", True)
    train_kwargs.setdefault("spatial_contrastive_early_stop_window_epochs", 70)
    train_kwargs.setdefault("spatial_contrastive_early_stop_slope_threshold", 1.0e-4)
    train_kwargs.setdefault("spatial_contrastive_early_stop_min_epoch", 400)
    train_kwargs.setdefault("spatial_contrastive_early_stop_patience", 20)
    train_kwargs.setdefault("shared_kl_weight_scale", 1.0)
    train_kwargs.setdefault("private_kl_weight_scale", 1.0)
    train_kwargs.setdefault("late_kl_start_epoch", 0)
    train_kwargs.setdefault("late_kl_ramp_epochs", 0)
    train_kwargs.setdefault("late_shared_kl_weight_scale", train_kwargs["shared_kl_weight_scale"])
    train_kwargs.setdefault("late_private_kl_weight_scale", train_kwargs["private_kl_weight_scale"])
    train_kwargs.setdefault("late_reconstruction_start_epoch", 0)
    train_kwargs.setdefault("late_reconstruction_ramp_epochs", 0)
    train_kwargs.setdefault("late_reconstruction_st_weight_scale", 1.0)
    train_kwargs.setdefault("late_reconstruction_sm_weight_scale", 1.0)
    train_kwargs.setdefault("spatial_negative_margin_weight", 0.0)
    train_kwargs.setdefault("spatial_contrastive_stop_epoch", 0)
    train_kwargs.setdefault("spatial_negative_margin_warmup_epochs", 16)
    train_kwargs.setdefault("spatial_negative_margin_stop_epoch", 0)
    train_kwargs.setdefault("spatial_negative_margin_decay_epochs", 0)
    train_kwargs.setdefault("shared_latent_std_weight", 0.0)
    train_kwargs.setdefault("shared_latent_cov_weight", 0.0)
    train_kwargs.setdefault("shared_latent_geometry_warmup_epochs", 16)
    train_kwargs.setdefault("shared_latent_std_target", 1.0)
    train_kwargs.setdefault("private_latent_ceiling_weight", 0.0)
    train_kwargs.setdefault("private_latent_ceiling_ratio", 0.9)
    train_kwargs.setdefault("private_latent_ceiling_start_epoch", 0)
    train_kwargs.setdefault("private_latent_ceiling_ramp_epochs", 0)
    train_kwargs.setdefault("branch_scaled_shared_weight", 1.0)
    train_kwargs.setdefault("branch_scaled_st_weight", 1.0)
    train_kwargs.setdefault("branch_scaled_sm_weight", 1.0)
    train_kwargs.setdefault("save_embedding_epochs", [])
    return train_kwargs


def build_template_adata(adata: sc.AnnData) -> sc.AnnData:
    template = sc.AnnData(obs=adata.obs.loc[:, []].copy())
    template.obs_names = adata.obs_names.copy()
    if "spatial" not in adata.obsm:
        raise KeyError("Expected 'spatial' coordinates in adata.obsm for evaluation template")
    template.obsm["spatial"] = np.asarray(adata.obsm["spatial"], dtype=np.float32).copy()
    return template


def build_model(
    adata: sc.AnnData,
    train_kwargs: dict[str, object],
) -> DecAlignSpatialMetaLinear:
    return DecAlignSpatialMetaLinear(
        adata,
        proj_dim=int(train_kwargs["proj_dim"]),
        token_dim=int(train_kwargs["token_dim"]),
        n_latent=int(train_kwargs["n_latent"]),
        num_prototypes=int(train_kwargs["num_prototypes"]),
        dropout_rate=float(train_kwargs["dropout_rate"]),
        device=str(train_kwargs["device"]),
        reconstruction_method_st=str(train_kwargs["reconstruction_method_st"]),
        reconstruction_method_sm=str(train_kwargs["reconstruction_method_sm"]),
        standardize_inputs=bool(train_kwargs["standardize_inputs"]),
        use_standardized_reconstruction=bool(train_kwargs["standardized_reconstruction"]),
        feature_input_mode=bool(train_kwargs["feature_input_mode"]),
        spatial_hidden_dim=int(train_kwargs["spatial_coord_hidden_dim"]),
        spatial_context_hidden_dim=int(train_kwargs["spatial_context_hidden_dim"]),
        spatial_context_k=int(train_kwargs["spatial_context_k"]),
        spatial_encoder_mode=str(train_kwargs["spatial_encoder_mode"]),
        spatial_fourier_scales=tuple(float(value) for value in train_kwargs["spatial_fourier_scales"]),
        spatial_token_scale=float(train_kwargs["spatial_token_scale"]),
        spatial_token_dropout=float(train_kwargs["spatial_token_dropout"]),
        spatial_contrastive_pos_k=int(train_kwargs["spatial_contrastive_pos_k"]),
        spatial_contrastive_neg_k=int(train_kwargs["spatial_contrastive_neg_k"]),
        spatial_contrastive_temperature=float(train_kwargs["spatial_contrastive_temperature"]),
        spatial_contrastive_neg_strategy=str(train_kwargs["spatial_contrastive_neg_strategy"]),
        spatial_contrastive_mode=str(train_kwargs["spatial_contrastive_mode"]),
        spatial_negative_margin=float(train_kwargs["spatial_negative_margin"]),
        spatial_positive_weighting=str(train_kwargs["spatial_positive_weighting"]),
        spatial_positive_aggregation=str(train_kwargs["spatial_positive_aggregation"]),
        spatial_positive_weight_temperature=float(train_kwargs["spatial_positive_weight_temperature"]),
        decoder_hidden_dim=int(train_kwargs.get("decoder_hidden_dim", train_kwargs["proj_dim"])),
        decoder_num_layers=int(train_kwargs.get("decoder_num_layers", 1)),
        decoder_private_feature_masking=bool(train_kwargs["decoder_private_feature_masking"]),
        decoder_private_mask_probability=float(train_kwargs["decoder_private_mask_probability"]),
        decoder_private_mask_warmup_start=int(train_kwargs["decoder_private_mask_warmup_start"]),
        decoder_private_mask_warmup_end=int(train_kwargs["decoder_private_mask_warmup_end"]),
        private_encoder_num_layers=int(train_kwargs.get("private_encoder_num_layers", 1)),
        private_encoder_activation=str(train_kwargs.get("private_encoder_activation", "none")),
        shared_graph_mode=str(train_kwargs.get("shared_graph_mode", "praga_fused")),
    )


def resolve_gt_labels_for_template(
    *,
    adata_template: sc.AnnData,
    gt_h5ad: Path | None,
    annotation_csv: Path | None,
    gt_key: str | None,
) -> np.ndarray:
    pred_coords = np.asarray(adata_template.obsm["spatial"], dtype=np.float64)[:, :2]
    if gt_h5ad is not None and gt_h5ad.exists():
        labels = load_gt_from_h5ad(
            gt_h5ad,
            adata_template.obs_names,
            pred_coords,
            gt_key,
        )
    elif annotation_csv is not None and annotation_csv.exists():
        labels = load_gt_from_annotation_csv(annotation_csv, adata_template.obs_names)
    else:
        raise FileNotFoundError("neither ground-truth h5ad nor annotation csv is available")
    return labels.astype(object).to_numpy()


def select_labeled_embedding(
    embedding: np.ndarray,
    labels_true: np.ndarray,
    target_n_clusters: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    labels = pd.Series(labels_true, dtype=object)
    valid = labels.notna().to_numpy()
    matched_spots = int(valid.sum())
    if matched_spots < 2:
        raise RuntimeError(f"too few matched ground-truth labels: {matched_spots}")

    matched_labels = labels.iloc[np.flatnonzero(valid)].astype(str).to_numpy()
    observed_gt_classes = int(pd.Series(matched_labels).nunique())
    audit = {
        "total_eval_spots": float(len(labels)),
        "matched_eval_spots": float(matched_spots),
        "unlabeled_eval_spots": float(len(labels) - matched_spots),
        "observed_gt_classes": float(observed_gt_classes),
        "target_pred_clusters": float(target_n_clusters),
    }
    return np.asarray(embedding)[valid], matched_labels, audit


def evaluate_embedding_fast(
    *,
    q_mu_shared: np.ndarray,
    q_mu_st: np.ndarray,
    q_mu_sm: np.ndarray,
    embedding_eval_mode: str,
    branch_scaled_shared_weight: float = 1.0,
    branch_scaled_st_weight: float = 1.0,
    branch_scaled_sm_weight: float = 1.0,
    labels_true: np.ndarray,
    target_n_clusters: int,
    pca_components: int,
    random_seed: int,
    rscript: Path,
    work_dir: Path,
    mclust_model_name: str,
) -> tuple[str, dict[str, float]]:
    embedding_name, eval_embedding = build_eval_embedding(
        q_mu_shared=q_mu_shared,
        q_mu_st=q_mu_st,
        q_mu_sm=q_mu_sm,
        embedding_eval_mode=embedding_eval_mode,
        branch_scaled_shared_weight=branch_scaled_shared_weight,
        branch_scaled_st_weight=branch_scaled_st_weight,
        branch_scaled_sm_weight=branch_scaled_sm_weight,
    )
    eval_embedding, labels_true, label_audit = select_labeled_embedding(
        eval_embedding,
        labels_true,
        target_n_clusters,
    )
    projected = pca_project_local(eval_embedding, pca_components)

    labels_pred = run_mclust_fixed_k(
        projected,
        n_clusters=target_n_clusters,
        random_seed=random_seed,
        rscript=rscript,
        work_dir=work_dir,
        model_name=mclust_model_name,
    )

    # Old fixed-k Leiden path kept for quick rollback.
    # labels_pred, resolution = run_leiden_fixed_k(
    #     projected,
    #     n_clusters=target_n_clusters,
    #     random_seed=random_seed,
    #     n_neighbors=15,
    #     metric="euclidean",
    # )

    metrics = compute_metrics(labels_true, labels_pred)
    metrics["observed_pred_clusters"] = float(pd.Series(labels_pred).nunique())
    metrics.update(label_audit)
    metrics["cluster_model_name"] = str(mclust_model_name)
    metrics = {key: float(value) if isinstance(value, (int, float, np.floating)) else value for key, value in metrics.items()}
    return embedding_name, metrics


def current_warmup_weight(target: float, warmup_epochs: int, epoch_num: int) -> float:
    if warmup_epochs <= 0:
        return float(target)
    scale = min(max(epoch_num - 1, 0) / float(warmup_epochs), 1.0)
    return float(target) * float(scale)


def capture_rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def update_global_metrics(global_csv_path: Path, row: dict[str, object]) -> None:
    row_df = pd.DataFrame([row])
    if global_csv_path.exists():
        all_df = pd.read_csv(global_csv_path)
        all_df = all_df[
            ~(
                (all_df["config_name"] == row["config_name"])
                & (all_df["sample_name"] == row["sample_name"])
                & (all_df["epoch"] == row["epoch"])
            )
        ]
        all_df = pd.concat([all_df, row_df], ignore_index=True)
    else:
        all_df = row_df
    all_df = all_df.sort_values(["sample_name", "config_name", "epoch"]).reset_index(drop=True)
    all_df.to_csv(global_csv_path, index=False)


def save_checkpoint(
    *,
    checkpoint_path: Path,
    model: DecAlignSpatialMetaLinear,
    epoch: int,
    config_name: str,
    sample_name: str,
    input_h5ad_path: Path,
    train_kwargs: dict[str, object],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": int(epoch),
        "config_name": config_name,
        "sample_name": sample_name,
        "input_h5ad_path": str(input_h5ad_path),
        "train_kwargs": dict(train_kwargs),
        "model_state_dict": model.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)


def save_embedding_snapshot(
    *,
    output_root: Path,
    epoch: int,
    spot_ids: pd.Index,
    q_mu_shared: np.ndarray,
    q_mu_st: np.ndarray,
    q_mu_sm: np.ndarray,
    raw_full_q_mu: np.ndarray,
    branch_scaled_full: np.ndarray,
    metrics: dict[str, float],
) -> Path:
    snapshot_dir = output_root / "saved_epoch_embeddings" / f"epoch_{int(epoch):04d}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    np.save(snapshot_dir / "q_mu_shared.npy", np.asarray(q_mu_shared, dtype=np.float32))
    np.save(snapshot_dir / "q_mu_st.npy", np.asarray(q_mu_st, dtype=np.float32))
    np.save(snapshot_dir / "q_mu_sm.npy", np.asarray(q_mu_sm, dtype=np.float32))
    np.save(snapshot_dir / "raw_full_q_mu.npy", np.asarray(raw_full_q_mu, dtype=np.float32))
    np.save(snapshot_dir / "branch_scaled_full.npy", np.asarray(branch_scaled_full, dtype=np.float32))
    pd.DataFrame({"spot_id": pd.Index(spot_ids).astype(str)}).to_csv(snapshot_dir / "spot_ids.csv", index=False)
    (snapshot_dir / "metrics.json").write_text(
        json.dumps(
            {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float, np.floating))},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return snapshot_dir


def save_named_embedding_snapshot(
    *,
    output_root: Path,
    snapshot_name: str,
    epoch: int,
    spot_ids: pd.Index,
    q_mu_shared: np.ndarray,
    q_mu_st: np.ndarray,
    q_mu_sm: np.ndarray,
    raw_full_q_mu: np.ndarray,
    branch_scaled_full: np.ndarray,
    metrics: dict[str, float],
) -> Path:
    snapshot_dir = output_root / "saved_best_embeddings" / snapshot_name
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    np.save(snapshot_dir / "q_mu_shared.npy", np.asarray(q_mu_shared, dtype=np.float32))
    np.save(snapshot_dir / "q_mu_st.npy", np.asarray(q_mu_st, dtype=np.float32))
    np.save(snapshot_dir / "q_mu_sm.npy", np.asarray(q_mu_sm, dtype=np.float32))
    np.save(snapshot_dir / "raw_full_q_mu.npy", np.asarray(raw_full_q_mu, dtype=np.float32))
    np.save(snapshot_dir / "branch_scaled_full.npy", np.asarray(branch_scaled_full, dtype=np.float32))
    pd.DataFrame({"spot_id": pd.Index(spot_ids).astype(str)}).to_csv(snapshot_dir / "spot_ids.csv", index=False)
    metrics_payload = {
        key: float(value) for key, value in metrics.items() if isinstance(value, (int, float, np.floating))
    }
    metrics_payload["epoch"] = int(epoch)
    (snapshot_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return snapshot_dir


def build_train_eval_parser(
    *,
    default_sample: str,
    default_processed_root: Path,
    default_output_root: Path,
    default_gt_h5ad: Path,
    default_annotation_csv: Path,
    default_device: str,
    default_cluster_resolution: float,
    default_target_n_clusters: int,
    default_pca_components: int,
    default_cluster_random_state: int,
    default_eval_every: int,
    default_max_epoch: int,
    default_n_per_batch: int,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Spatial multi-omic train/eval entry with periodic mclust evaluation.")
    parser.add_argument("--sample-name", default=default_sample)
    parser.add_argument("--processed-root", type=Path, default=default_processed_root)
    parser.add_argument("--output-root", type=Path, default=default_output_root)
    parser.add_argument("--config-name", default=None)
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--cluster-resolution", type=float, default=default_cluster_resolution)
    parser.add_argument("--max-epoch", type=int, default=default_max_epoch)
    parser.add_argument("--n-per-batch", type=int, default=default_n_per_batch)
    parser.add_argument("--eval-every", type=int, default=default_eval_every)
    parser.add_argument("--target-n-clusters", type=int, default=default_target_n_clusters)
    parser.add_argument("--pca-components", type=int, default=default_pca_components)
    parser.add_argument("--cluster-random-state", type=int, default=default_cluster_random_state)
    parser.add_argument("--leiden-random-state", dest="cluster_random_state", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--mclust-model-name", default="EEE")
    parser.add_argument("--rscript", type=Path, default=None)
    parser.add_argument("--gt-h5ad", type=Path, default=default_gt_h5ad)
    parser.add_argument("--annotation-csv", type=Path, default=default_annotation_csv)
    parser.add_argument("--gt-key", default=None)
    parser.add_argument("--train-overrides-json", default=None)
    parser.add_argument("--save-checkpoint-epochs", type=int, nargs="*", default=None)
    parser.add_argument("--skip-online-cluster-eval", action="store_true")
    return parser


def run_train_eval_workflow(
    *,
    package_root: Path,
    args: argparse.Namespace,
    base_train_kwargs: dict[str, object],
) -> None:
    train_overrides = None
    if args.train_overrides_json:
        train_overrides = json.loads(args.train_overrides_json)
        if not isinstance(train_overrides, dict):
            raise TypeError("--train-overrides-json must decode to a JSON object")

    gt_h5ad = args.gt_h5ad if args.gt_h5ad.exists() else None
    annotation_csv = args.annotation_csv if args.annotation_csv.exists() else None
    rscript = resolve_rscript(args.rscript)

    train_kwargs = build_train_kwargs(
        base_train_kwargs=base_train_kwargs,
        device=args.device,
        cluster_resolution=args.cluster_resolution,
        max_epoch=args.max_epoch,
        n_per_batch=args.n_per_batch,
        train_overrides=train_overrides,
    )
    config_name = args.config_name or (
        f"{args.sample_name}_batch{train_kwargs['n_per_batch']}_epoch{train_kwargs['max_epoch']}"
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    input_h5ad_path = args.processed_root / f"{args.sample_name}.h5ad"
    config_snapshot_path = args.output_root / "config.json"
    mclust_work_dir = args.output_root / "_mclust_tmp"

    print(f"[entry] sample={args.sample_name}", flush=True)
    print(f"[entry] input_h5ad={input_h5ad_path}", flush=True)
    print(f"[entry] config_name={config_name}", flush=True)
    print(f"[entry] output_root={args.output_root}", flush=True)
    print(f"[entry] device={args.device}", flush=True)
    print(f"[entry] batch_size={train_kwargs['n_per_batch']}", flush=True)
    print(f"[entry] max_epoch={train_kwargs['max_epoch']}", flush=True)
    print(f"[entry] eval_every={args.eval_every}", flush=True)
    print(
        f"[entry] train_overrides={json.dumps(normalize_train_overrides(train_overrides), ensure_ascii=False, sort_keys=True)}",
        flush=True,
    )
    config_snapshot = {
        "config_name": config_name,
        "sample_name": args.sample_name,
        "input_h5ad_path": str(input_h5ad_path),
        "gt_h5ad": str(args.gt_h5ad),
        "annotation_csv": str(args.annotation_csv),
        "target_n_clusters": int(args.target_n_clusters),
        "pca_components": int(args.pca_components),
        "cluster_eval_method": "none" if bool(args.skip_online_cluster_eval) else "mclust_fixed_k",
        "mclust_random_seed": int(args.cluster_random_state),
        "mclust_model_name": str(args.mclust_model_name),
        "rscript": str(rscript),
        "eval_every": int(args.eval_every),
        "train_kwargs": {key: serialize_table_value(value) for key, value in train_kwargs.items()},
    }
    config_snapshot_path.write_text(json.dumps(config_snapshot, ensure_ascii=False, indent=2))

    seed_everything(
        int(train_kwargs["random_seed"]),
        bool(train_kwargs["deterministic"]),
        bool(train_kwargs["deterministic_warn_only"]),
    )
    if int(train_kwargs["max_cells"]) != 0:
        raise ValueError("max_cells is preprocessing and is no longer supported by the training entry")
    print(f"[input] read model-ready h5ad {input_h5ad_path}", flush=True)
    adata_train = sc.read_h5ad(input_h5ad_path)
    validate_spadta_model_input(
        adata_train,
        expression_graph_k=int(train_kwargs["spatial_contrastive_pos_k"]),
        spatial_context_k=int(train_kwargs["spatial_context_k"]),
    )
    adata_template = build_template_adata(adata_train)
    model = build_model(adata_train, train_kwargs)
    cached_gt_labels = resolve_gt_labels_for_template(
        adata_template=adata_template,
        gt_h5ad=gt_h5ad,
        annotation_csv=annotation_csv,
        gt_key=args.gt_key,
    )
    graph_summary_path = args.output_root / "graph_summary.json"
    initial_graph_summary = model.summarize_fused_graphs()
    initial_graph_summary["initial_expression_edge_weight_stats"] = model.get_expression_edge_weight_stats()
    graph_summary_path.write_text(json.dumps(initial_graph_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    best_row: dict[str, object] | None = None
    global_csv_path = args.output_root / "epoch_metrics_all.csv"
    epoch_training_log_path = args.output_root / "epoch_training_log.csv"
    epoch_training_rows: list[dict[str, object]] = []
    checkpoint_epochs = (
        set() if args.save_checkpoint_epochs is None else {int(epoch) for epoch in args.save_checkpoint_epochs}
    )
    save_embedding_epochs = {int(epoch) for epoch in train_kwargs.get("save_embedding_epochs", [])}

    def finalize_metric_row(row: dict[str, object], metrics: dict[str, float]) -> None:
        nonlocal best_row
        row = dict(row)
        row.update(
            {
                "ARI": metrics["ARI"],
                "NMI": metrics["NMI"],
                "AMI": metrics["AMI"],
                "Homo": metrics["Homo"],
                "V-Measure": metrics["V-Measure"],
                "FMI": metrics["FMI"],
                "MI": metrics["MI"],
                "mclust_ari": metrics["ARI"],
                "mclust_nmi": metrics["NMI"],
                "cluster_eval_method": "mclust_fixed_k",
                "cluster_model_name": metrics["cluster_model_name"],
                "observed_pred_clusters": metrics["observed_pred_clusters"],
                "observed_gt_classes": metrics["observed_gt_classes"],
                "target_pred_clusters": metrics["target_pred_clusters"],
                "total_eval_spots": metrics["total_eval_spots"],
                "matched_eval_spots": metrics["matched_eval_spots"],
                "unlabeled_eval_spots": metrics["unlabeled_eval_spots"],
            }
        )
        update_global_metrics(global_csv_path, row)
        if best_row is None or float(row["ARI"]) > float(best_row["ARI"]):
            best_row = dict(row)
        print(
            json.dumps(
                {
                    "epoch": int(row["epoch"]),
                    "ARI": round(float(row["ARI"]), 6),
                    "NMI": round(float(row["NMI"]), 6),
                    "total_loss": round(float(row["total_loss"]), 6),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    def evaluate_checkpoint(epoch_num: int, stats: dict[str, float], history: dict[str, list[float]]) -> None:
        del history
        edge_weight_stats = model.get_expression_edge_weight_stats()
        training_row = {
            "config_name": config_name,
            "sample_name": args.sample_name,
            "epoch": int(epoch_num),
            "reconstruction_loss_st": float(stats["reconstruction_loss_st"]),
            "reconstruction_loss_sm": float(stats["reconstruction_loss_sm"]),
            "kldiv_loss": float(stats["kldiv_loss"]),
            "kldiv_loss_shared": float(stats["kldiv_loss_shared"]),
            "kldiv_loss_st": float(stats["kldiv_loss_st"]),
            "kldiv_loss_sm": float(stats["kldiv_loss_sm"]),
            "weighted_kl_shared_term": float(stats["weighted_kl_shared_term"]),
            "weighted_kl_private_term": float(stats["weighted_kl_private_term"]),
            "current_shared_kl_weight_scale": float(stats["current_shared_kl_weight_scale"]),
            "current_private_kl_weight_scale": float(stats["current_private_kl_weight_scale"]),
            "current_reconstruction_st_weight_scale": float(stats["current_reconstruction_st_weight_scale"]),
            "current_reconstruction_sm_weight_scale": float(stats["current_reconstruction_sm_weight_scale"]),
            "effective_reconstruction_st_weight": float(stats["effective_reconstruction_st_weight"]),
            "effective_reconstruction_sm_weight": float(stats["effective_reconstruction_sm_weight"]),
            "shared_latent_std_loss": float(stats["shared_latent_std_loss"]),
            "shared_latent_cov_loss": float(stats["shared_latent_cov_loss"]),
            "weighted_shared_latent_geometry_term": float(stats["weighted_shared_latent_geometry_term"]),
            "shared_latent_std_mean": float(stats["shared_latent_std_mean"]),
            "shared_latent_std_min": float(stats["shared_latent_std_min"]),
            "shared_latent_std_max": float(stats["shared_latent_std_max"]),
            "shared_latent_anisotropy_ratio": float(stats["shared_latent_anisotropy_ratio"]),
            "shared_latent_cov_offdiag_abs_mean": float(stats["shared_latent_cov_offdiag_abs_mean"]),
            "private_latent_ceiling_loss": float(stats["private_latent_ceiling_loss"]),
            "weighted_private_latent_ceiling_term": float(stats["weighted_private_latent_ceiling_term"]),
            "current_private_latent_ceiling_weight": float(stats["current_private_latent_ceiling_weight"]),
            "private_latent_shared_std_reference": float(stats["private_latent_shared_std_reference"]),
            "private_st_latent_std_mean": float(stats["private_st_latent_std_mean"]),
            "private_sm_latent_std_mean": float(stats["private_sm_latent_std_mean"]),
            "private_st_latent_excess_fraction": float(stats["private_st_latent_excess_fraction"]),
            "private_sm_latent_excess_fraction": float(stats["private_sm_latent_excess_fraction"]),
            "spatial_consistency_loss": float(stats["spatial_consistency_loss"]),
            "spatial_contrastive_loss": float(stats["spatial_contrastive_loss"]),
            "current_spatial_contrastive_weight": float(stats["current_spatial_contrastive_weight"]),
            "spatial_negative_margin_loss": float(stats["spatial_negative_margin_loss"]),
            "weighted_spatial_negative_margin_term": float(stats["weighted_spatial_negative_margin_term"]),
            "current_spatial_negative_margin_weight": float(stats["current_spatial_negative_margin_weight"]),
            "negative_mean_cosine": float(stats["negative_mean_cosine"]),
            "negative_max_cosine": float(stats["negative_max_cosine"]),
            "negative_violation_rate": float(stats["negative_violation_rate"]),
            "effective_negative_pairs": float(stats["effective_negative_pairs"]),
            "positive_count_mean": float(stats["positive_count_mean"]),
            "positive_weight_sum_mean": float(stats["positive_weight_sum_mean"]),
            "positive_weight_mean": float(stats["positive_weight_mean"]),
            "positive_weight_min": float(stats["positive_weight_min"]),
            "positive_weight_max": float(stats["positive_weight_max"]),
            "rank1_weight_mean": float(stats["rank1_weight_mean"]),
            "rank2_weight_mean": float(stats["rank2_weight_mean"]),
            "rank3_weight_mean": float(stats["rank3_weight_mean"]),
            "weighted_positive_distance": float(stats["weighted_positive_distance"]),
            "unweighted_positive_distance": float(stats["unweighted_positive_distance"]),
            "dec_loss": float(stats["dec_loss"]),
            "hete_loss": float(stats["hete_loss"]),
            "homo_loss": float(stats["homo_loss"]),
            "task_loss_shared": float(stats["task_loss_shared"]),
            "task_loss_reconstruction_st": float(stats["task_loss_reconstruction_st"]),
            "task_loss_reconstruction_sm": float(stats["task_loss_reconstruction_sm"]),
            "total_loss": float(stats["total_loss"]),
            "task_weight_shared": float(stats["task_weight_shared"]),
            "task_weight_reconstruction_st": float(stats["task_weight_reconstruction_st"]),
            "task_weight_reconstruction_sm": float(stats["task_weight_reconstruction_sm"]),
            "decoder_private_mask_probability_current": float(stats["decoder_private_mask_probability_current"]),
            "decoder_st_private_actual_mask_fraction": float(stats["decoder_st_private_actual_mask_fraction"]),
            "decoder_sm_private_actual_mask_fraction": float(stats["decoder_sm_private_actual_mask_fraction"]),
            "decoder_st_private_masked_dimensions_mean": float(stats["decoder_st_private_masked_dimensions_mean"]),
            "decoder_sm_private_masked_dimensions_mean": float(stats["decoder_sm_private_masked_dimensions_mean"]),
            "decoder_st_private_kept_dimensions_mean": float(stats["decoder_st_private_kept_dimensions_mean"]),
            "decoder_sm_private_kept_dimensions_mean": float(stats["decoder_sm_private_kept_dimensions_mean"]),
        }
        training_row.update(model.get_graph_fusion_weights())
        training_row.update(edge_weight_stats)
        epoch_training_rows.append(training_row)
        pd.DataFrame(epoch_training_rows).to_csv(epoch_training_log_path, index=False)
        if int(epoch_num) in checkpoint_epochs:
            checkpoint_path = args.output_root / f"checkpoint_epoch_{int(epoch_num):03d}.pt"
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                epoch=int(epoch_num),
                config_name=config_name,
                sample_name=args.sample_name,
                input_h5ad_path=input_h5ad_path,
                train_kwargs=train_kwargs,
            )

        should_eval = (
            (epoch_num % args.eval_every == 0)
            or (epoch_num == int(train_kwargs["max_epoch"]))
            or (int(epoch_num) in save_embedding_epochs)
            or bool(stats.get("spatial_contrastive_early_stop_triggered", 0.0))
        )
        should_save_snapshot = int(epoch_num) in save_embedding_epochs
        should_run_online_eval = bool(should_eval) and (not bool(args.skip_online_cluster_eval))
        if not should_save_snapshot and not should_run_online_eval:
            return

        row = None
        if should_run_online_eval:
            row = {
                "config_name": config_name,
                "sample_name": args.sample_name,
                "epoch": int(epoch_num),
                "n_per_batch": int(train_kwargs["n_per_batch"]),
                "max_epoch": int(train_kwargs["max_epoch"]),
                "mclust_random_seed": int(args.cluster_random_state),
                "reconstruction_loss_st": stats["reconstruction_loss_st"],
                "reconstruction_loss_sm": stats["reconstruction_loss_sm"],
                "kldiv_loss": stats["kldiv_loss"],
                "kldiv_loss_shared": stats["kldiv_loss_shared"],
                "kldiv_loss_st": stats["kldiv_loss_st"],
                "kldiv_loss_sm": stats["kldiv_loss_sm"],
                "kl_weight": current_warmup_weight(
                    target=float(train_kwargs["kl_weight"]),
                    warmup_epochs=int(train_kwargs["n_epochs_kl_warmup"]),
                    epoch_num=int(epoch_num),
                ),
                "shared_kl_weight_scale": float(train_kwargs["shared_kl_weight_scale"]),
                "private_kl_weight_scale": float(train_kwargs["private_kl_weight_scale"]),
                "current_shared_kl_weight_scale": float(stats["current_shared_kl_weight_scale"]),
                "current_private_kl_weight_scale": float(stats["current_private_kl_weight_scale"]),
                "current_reconstruction_st_weight_scale": float(stats["current_reconstruction_st_weight_scale"]),
                "current_reconstruction_sm_weight_scale": float(stats["current_reconstruction_sm_weight_scale"]),
                "effective_reconstruction_st_weight": float(stats["effective_reconstruction_st_weight"]),
                "effective_reconstruction_sm_weight": float(stats["effective_reconstruction_sm_weight"]),
                "weighted_kl_shared_term": float(stats["weighted_kl_shared_term"]),
                "weighted_kl_private_term": float(stats["weighted_kl_private_term"]),
                "weighted_kl_term": float(stats["weighted_kl_shared_term"] + stats["weighted_kl_private_term"]),
                "shared_latent_std_loss": float(stats["shared_latent_std_loss"]),
                "shared_latent_cov_loss": float(stats["shared_latent_cov_loss"]),
                "weighted_shared_latent_geometry_term": float(stats["weighted_shared_latent_geometry_term"]),
                "shared_latent_std_mean": float(stats["shared_latent_std_mean"]),
                "shared_latent_std_min": float(stats["shared_latent_std_min"]),
                "shared_latent_std_max": float(stats["shared_latent_std_max"]),
                "shared_latent_anisotropy_ratio": float(stats["shared_latent_anisotropy_ratio"]),
                "shared_latent_cov_offdiag_abs_mean": float(stats["shared_latent_cov_offdiag_abs_mean"]),
                "private_latent_ceiling_loss": float(stats["private_latent_ceiling_loss"]),
                "weighted_private_latent_ceiling_term": float(stats["weighted_private_latent_ceiling_term"]),
                "current_private_latent_ceiling_weight": float(stats["current_private_latent_ceiling_weight"]),
                "private_latent_shared_std_reference": float(stats["private_latent_shared_std_reference"]),
                "private_st_latent_std_mean": float(stats["private_st_latent_std_mean"]),
                "private_sm_latent_std_mean": float(stats["private_sm_latent_std_mean"]),
                "private_st_latent_excess_fraction": float(stats["private_st_latent_excess_fraction"]),
                "private_sm_latent_excess_fraction": float(stats["private_sm_latent_excess_fraction"]),
                "spatial_consistency_loss": float(stats["spatial_consistency_loss"]),
                "spatial_contrastive_loss": float(stats["spatial_contrastive_loss"]),
                "current_spatial_contrastive_weight": float(stats["current_spatial_contrastive_weight"]),
                "spatial_negative_margin_loss": float(stats["spatial_negative_margin_loss"]),
                "weighted_spatial_negative_margin_term": float(stats["weighted_spatial_negative_margin_term"]),
                "current_spatial_negative_margin_weight": float(stats["current_spatial_negative_margin_weight"]),
                "negative_mean_cosine": float(stats["negative_mean_cosine"]),
                "negative_max_cosine": float(stats["negative_max_cosine"]),
                "negative_violation_rate": float(stats["negative_violation_rate"]),
                "effective_negative_pairs": float(stats["effective_negative_pairs"]),
                "positive_count_mean": float(stats["positive_count_mean"]),
                "positive_weight_sum_mean": float(stats["positive_weight_sum_mean"]),
                "positive_weight_mean": float(stats["positive_weight_mean"]),
                "positive_weight_min": float(stats["positive_weight_min"]),
                "positive_weight_max": float(stats["positive_weight_max"]),
                "rank1_weight_mean": float(stats["rank1_weight_mean"]),
                "rank2_weight_mean": float(stats["rank2_weight_mean"]),
                "rank3_weight_mean": float(stats["rank3_weight_mean"]),
                "weighted_positive_distance": float(stats["weighted_positive_distance"]),
                "unweighted_positive_distance": float(stats["unweighted_positive_distance"]),
                "homo_loss": stats["homo_loss"],
                "task_loss_shared": stats["task_loss_shared"],
                "total_loss": stats["total_loss"],
                "task_weight_shared": stats["task_weight_shared"],
                "task_weight_reconstruction_st": stats["task_weight_reconstruction_st"],
                "task_weight_reconstruction_sm": stats["task_weight_reconstruction_sm"],
                "decoder_private_mask_probability_current": float(stats["decoder_private_mask_probability_current"]),
                "decoder_st_private_actual_mask_fraction": float(stats["decoder_st_private_actual_mask_fraction"]),
                "decoder_sm_private_actual_mask_fraction": float(stats["decoder_sm_private_actual_mask_fraction"]),
                "decoder_st_private_masked_dimensions_mean": float(stats["decoder_st_private_masked_dimensions_mean"]),
                "decoder_sm_private_masked_dimensions_mean": float(stats["decoder_sm_private_masked_dimensions_mean"]),
                "decoder_st_private_kept_dimensions_mean": float(stats["decoder_st_private_kept_dimensions_mean"]),
                "decoder_sm_private_kept_dimensions_mean": float(stats["decoder_sm_private_kept_dimensions_mean"]),
            }
            for key, value in train_kwargs.items():
                row[key] = serialize_table_value(value)

        was_training = model.training
        rng_state = capture_rng_state()
        try:
            contribution_details = model.get_modality_contribution_details()
            q_mu_shared = np.asarray(contribution_details["q_mu_shared"], dtype=np.float64)
            q_mu_st = np.asarray(contribution_details["q_mu_st"], dtype=np.float64)
            q_mu_sm = np.asarray(contribution_details["q_mu_sm"], dtype=np.float64)
            raw_full_q_mu = build_raw_full_q_mu(
                q_mu_shared=q_mu_shared,
                q_mu_st=q_mu_st,
                q_mu_sm=q_mu_sm,
            )
            branch_scaled_full = build_branch_scaled_full(
                q_mu_shared=q_mu_shared,
                q_mu_st=q_mu_st,
                q_mu_sm=q_mu_sm,
                shared_weight=float(train_kwargs["branch_scaled_shared_weight"]),
                st_weight=float(train_kwargs["branch_scaled_st_weight"]),
                sm_weight=float(train_kwargs["branch_scaled_sm_weight"]),
            )

            metrics = None
            if should_save_snapshot:
                save_embedding_snapshot(
                    output_root=args.output_root,
                    epoch=int(epoch_num),
                    spot_ids=adata_template.obs_names,
                    q_mu_shared=q_mu_shared,
                    q_mu_st=q_mu_st,
                    q_mu_sm=q_mu_sm,
                    raw_full_q_mu=raw_full_q_mu,
                    branch_scaled_full=branch_scaled_full,
                    metrics={} if row is None else row,
                )

            if should_run_online_eval:
                embedding_source, metrics = evaluate_embedding_fast(
                    q_mu_shared=q_mu_shared,
                    q_mu_st=q_mu_st,
                    q_mu_sm=q_mu_sm,
                    embedding_eval_mode=str(train_kwargs.get("embedding_eval_mode", "branch_scaled_full")),
                    branch_scaled_shared_weight=float(train_kwargs["branch_scaled_shared_weight"]),
                    branch_scaled_st_weight=float(train_kwargs["branch_scaled_st_weight"]),
                    branch_scaled_sm_weight=float(train_kwargs["branch_scaled_sm_weight"]),
                    labels_true=cached_gt_labels,
                    target_n_clusters=args.target_n_clusters,
                    pca_components=args.pca_components,
                    random_seed=args.cluster_random_state,
                    rscript=rscript,
                    work_dir=mclust_work_dir,
                    mclust_model_name=args.mclust_model_name,
                )

                shared_stats = summarize_branch_variance(q_mu_shared)
                st_stats = summarize_branch_variance(q_mu_st)
                sm_stats = summarize_branch_variance(q_mu_sm)
                full_total_variance = (
                    shared_stats["total_variance"] + st_stats["total_variance"] + sm_stats["total_variance"]
                )
                row.update(
                    {
                        "embedding_source": str(embedding_source),
                        "pca_dimension": int(args.pca_components),
                        "cluster_model_name": str(args.mclust_model_name),
                        "number_of_clusters": int(args.target_n_clusters),
                        "q_mu_shared_mean_dimension_variance": shared_stats["mean_dimension_variance"],
                        "raw_q_mu_shared_total_variance": shared_stats["total_variance"],
                        "raw_q_mu_st_total_variance": st_stats["total_variance"],
                        "raw_q_mu_sm_total_variance": sm_stats["total_variance"],
                        "raw_q_mu_shared_mean_dimension_variance": shared_stats["mean_dimension_variance"],
                        "raw_q_mu_st_mean_dimension_variance": st_stats["mean_dimension_variance"],
                        "raw_q_mu_sm_mean_dimension_variance": sm_stats["mean_dimension_variance"],
                        "raw_q_mu_shared_mean_vector_norm": shared_stats["mean_vector_norm"],
                        "raw_q_mu_st_mean_vector_norm": st_stats["mean_vector_norm"],
                        "raw_q_mu_sm_mean_vector_norm": sm_stats["mean_vector_norm"],
                        "raw_q_mu_shared_variance_fraction": shared_stats["total_variance"] / max(full_total_variance, 1e-12),
                        "raw_q_mu_st_variance_fraction": st_stats["total_variance"] / max(full_total_variance, 1e-12),
                        "raw_q_mu_sm_variance_fraction": sm_stats["total_variance"] / max(full_total_variance, 1e-12),
                    }
                )
                row.update(model.get_graph_fusion_weights())
                row.update(edge_weight_stats)

                prev_best_ari = float(best_row["ARI"]) if best_row is not None else float("-inf")
                current_ari = float(metrics["ARI"])
                if current_ari > prev_best_ari:
                    save_named_embedding_snapshot(
                        output_root=args.output_root,
                        snapshot_name="best_ari",
                        epoch=int(epoch_num),
                        spot_ids=adata_template.obs_names,
                        q_mu_shared=q_mu_shared,
                        q_mu_st=q_mu_st,
                        q_mu_sm=q_mu_sm,
                        raw_full_q_mu=raw_full_q_mu,
                        branch_scaled_full=branch_scaled_full,
                        metrics=metrics,
                    )
                    save_checkpoint(
                        checkpoint_path=args.output_root / "checkpoint_best_ari.pt",
                        model=model,
                        epoch=int(epoch_num),
                        config_name=config_name,
                        sample_name=args.sample_name,
                        input_h5ad_path=input_h5ad_path,
                        train_kwargs=train_kwargs,
                    )
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "epoch": int(epoch_num),
                        "eval_skipped": True,
                        "reason": str(exc),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            return

        finally:
            restore_rng_state(rng_state)
            if was_training:
                model.train()

        if should_run_online_eval and row is not None and metrics is not None:
            finalize_metric_row(row, metrics)

    model.fit(
        max_epoch=int(train_kwargs["max_epoch"]),
        n_per_batch=int(train_kwargs["n_per_batch"]),
        reconstruction_reduction=str(train_kwargs["reconstruction_reduction"]),
        reconstruction_st_weight=float(train_kwargs["reconstruction_st_weight"]),
        reconstruction_sm_weight=float(train_kwargs["reconstruction_sm_weight"]),
        dec_weight=float(train_kwargs["dec_weight"]),
        hete_weight=float(train_kwargs["hete_weight"]),
        homo_weight=float(train_kwargs["homo_weight"]),
        hete_warmup_epochs=int(train_kwargs["hete_warmup_epochs"]),
        homo_warmup_epochs=int(train_kwargs["homo_warmup_epochs"]),
        kl_weight=float(train_kwargs["kl_weight"]),
        n_epochs_kl_warmup=int(train_kwargs["n_epochs_kl_warmup"]),
        shared_kl_weight_scale=float(train_kwargs["shared_kl_weight_scale"]),
        private_kl_weight_scale=float(train_kwargs["private_kl_weight_scale"]),
        late_kl_start_epoch=int(train_kwargs["late_kl_start_epoch"]),
        late_kl_ramp_epochs=int(train_kwargs["late_kl_ramp_epochs"]),
        late_shared_kl_weight_scale=float(train_kwargs["late_shared_kl_weight_scale"]),
        late_private_kl_weight_scale=float(train_kwargs["late_private_kl_weight_scale"]),
        late_reconstruction_start_epoch=int(train_kwargs["late_reconstruction_start_epoch"]),
        late_reconstruction_ramp_epochs=int(train_kwargs["late_reconstruction_ramp_epochs"]),
        late_reconstruction_st_weight_scale=float(train_kwargs["late_reconstruction_st_weight_scale"]),
        late_reconstruction_sm_weight_scale=float(train_kwargs["late_reconstruction_sm_weight_scale"]),
        lr=float(train_kwargs["lr"]),
        weight_decay=float(train_kwargs["weight_decay"]),
        random_seed=int(train_kwargs["random_seed"]),
        balance_start_epoch=int(train_kwargs["balance_start_epoch"]),
        balance_ema=float(train_kwargs["balance_ema"]),
        task_weight_floor=float(train_kwargs["balance_weight_floor"]),
        spatial_consistency_weight=float(train_kwargs["spatial_consistency_weight"]),
        spatial_consistency_warmup_epochs=int(train_kwargs["spatial_consistency_warmup_epochs"]),
        spatial_consistency_use_all_latent=bool(train_kwargs["spatial_consistency_use_all_latent"]),
        spatial_contrastive_weight=float(train_kwargs["spatial_contrastive_weight"]),
        spatial_contrastive_warmup_epochs=int(train_kwargs["spatial_contrastive_warmup_epochs"]),
        spatial_contrastive_stop_epoch=int(train_kwargs["spatial_contrastive_stop_epoch"]),
        spatial_contrastive_use_all_latent=bool(train_kwargs["spatial_contrastive_use_all_latent"]),
        spatial_contrastive_latent_mode=str(train_kwargs.get("spatial_contrastive_latent_mode", "auto")),
        spatial_negative_margin_weight=float(train_kwargs["spatial_negative_margin_weight"]),
        spatial_negative_margin_warmup_epochs=int(train_kwargs["spatial_negative_margin_warmup_epochs"]),
        spatial_negative_margin_stop_epoch=int(train_kwargs["spatial_negative_margin_stop_epoch"]),
        spatial_negative_margin_decay_epochs=int(train_kwargs["spatial_negative_margin_decay_epochs"]),
        shared_latent_std_weight=float(train_kwargs["shared_latent_std_weight"]),
        shared_latent_cov_weight=float(train_kwargs["shared_latent_cov_weight"]),
        shared_latent_geometry_warmup_epochs=int(train_kwargs["shared_latent_geometry_warmup_epochs"]),
        shared_latent_std_target=float(train_kwargs["shared_latent_std_target"]),
        private_latent_ceiling_weight=float(train_kwargs["private_latent_ceiling_weight"]),
        private_latent_ceiling_ratio=float(train_kwargs["private_latent_ceiling_ratio"]),
        private_latent_ceiling_start_epoch=int(train_kwargs["private_latent_ceiling_start_epoch"]),
        private_latent_ceiling_ramp_epochs=int(train_kwargs["private_latent_ceiling_ramp_epochs"]),
        decoder_private_feature_masking=bool(train_kwargs["decoder_private_feature_masking"]),
        decoder_private_mask_probability=float(train_kwargs["decoder_private_mask_probability"]),
        decoder_private_mask_warmup_start=int(train_kwargs["decoder_private_mask_warmup_start"]),
        decoder_private_mask_warmup_end=int(train_kwargs["decoder_private_mask_warmup_end"]),
        spatial_contrastive_early_stop_enabled=bool(train_kwargs["spatial_contrastive_early_stop_enabled"]),
        spatial_contrastive_early_stop_window_epochs=int(train_kwargs["spatial_contrastive_early_stop_window_epochs"]),
        spatial_contrastive_early_stop_slope_threshold=float(
            train_kwargs["spatial_contrastive_early_stop_slope_threshold"]
        ),
        spatial_contrastive_early_stop_min_epoch=int(train_kwargs["spatial_contrastive_early_stop_min_epoch"]),
        spatial_contrastive_early_stop_patience=int(train_kwargs["spatial_contrastive_early_stop_patience"]),
        epoch_end_callback=evaluate_checkpoint,
    )

    final_graph_summary = model.summarize_fused_graphs()
    final_graph_summary["initial_expression_edge_weight_stats"] = initial_graph_summary["initial_expression_edge_weight_stats"]
    final_graph_summary["final_expression_edge_weight_stats"] = model.get_expression_edge_weight_stats()
    final_graph_summary["final_expression_edge_gradient_stats"] = model.get_expression_edge_gradient_stats()
    graph_summary_path.write_text(json.dumps(final_graph_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if best_row is not None:
        print(
            json.dumps(
                {
                    "best_epoch": int(best_row["epoch"]),
                    "best_ARI": round(float(best_row["ARI"]), 6),
                    "best_NMI": round(float(best_row["NMI"]), 6),
                    "global_csv": str(global_csv_path),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    else:
        print(
            json.dumps(
                {
                    "best_epoch": None,
                    "global_csv": str(global_csv_path),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


def run_train_eval_cli(
    *,
    package_root: Path,
    default_sample: str,
    default_processed_root: Path,
    default_output_root: Path,
    default_gt_h5ad: Path,
    default_annotation_csv: Path,
    default_device: str,
    default_cluster_resolution: float,
    default_target_n_clusters: int,
    default_pca_components: int,
    default_cluster_random_state: int,
    default_eval_every: int,
    base_train_kwargs: dict[str, object],
) -> None:
    parser = build_train_eval_parser(
        default_sample=default_sample,
        default_processed_root=default_processed_root,
        default_output_root=default_output_root,
        default_gt_h5ad=default_gt_h5ad,
        default_annotation_csv=default_annotation_csv,
        default_device=default_device,
        default_cluster_resolution=default_cluster_resolution,
        default_target_n_clusters=default_target_n_clusters,
        default_pca_components=default_pca_components,
        default_cluster_random_state=default_cluster_random_state,
        default_eval_every=default_eval_every,
        default_max_epoch=int(base_train_kwargs["max_epoch"]),
        default_n_per_batch=int(base_train_kwargs["n_per_batch"]),
    )
    args = parser.parse_args()
    run_train_eval_workflow(
        package_root=package_root,
        args=args,
        base_train_kwargs=base_train_kwargs,
    )
