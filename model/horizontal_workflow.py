from __future__ import annotations

import json
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from sklearn.metrics import silhouette_score
import torch

from .horizontal_model import DecAlignSpatialMetaLinearHorizontal
from .preprocess import validate_spadta_model_input


warnings.filterwarnings("ignore")

SUPPORTED_POSTHOC_BATCH_METHODS = {"none", "combat", "center", "zscore"}
MODEL_VARIANT = "spaDTA_horizontal_spatialcoords_sharedhalf"


@dataclass
class HorizontalTrainingResult:
    model: DecAlignSpatialMetaLinearHorizontal
    adata: sc.AnnData
    output_h5ad_path: Path
    loss_figure_path: Path
    loss_csv_path: Path
    param_json_path: Path
    loss_df: pd.DataFrame
    input_metadata: dict[str, object]


def seed_everything(seed: int, deterministic: bool, warn_only: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False


def normalize_posthoc_batch_method(method: Optional[str]) -> str:
    value = (method or "none").strip().lower()
    if value not in SUPPORTED_POSTHOC_BATCH_METHODS:
        raise ValueError(
            f"Unsupported posthoc batch method {method!r}; "
            f"expected one of {sorted(SUPPORTED_POSTHOC_BATCH_METHODS)}."
        )
    return value


def corrected_embedding_key(input_key: str, method: Optional[str]) -> str:
    normalized = normalize_posthoc_batch_method(method)
    return input_key if normalized == "none" else f"{input_key}_{normalized}"


def _combat_correct(embedding: np.ndarray, batch_labels: np.ndarray) -> np.ndarray:
    temp = AnnData(X=embedding.copy(), dtype=embedding.dtype)
    temp.obs["batch"] = pd.Categorical(batch_labels.astype(str))
    sc.pp.combat(temp, key="batch")
    return np.asarray(temp.X, dtype=np.float32)


def _center_correct(embedding: np.ndarray, batch_labels: np.ndarray) -> np.ndarray:
    corrected = embedding.copy()
    for batch in pd.unique(batch_labels):
        mask = batch_labels == batch
        corrected[mask] = corrected[mask] - corrected[mask].mean(axis=0, keepdims=True)
    return corrected


def _zscore_correct(embedding: np.ndarray, batch_labels: np.ndarray) -> np.ndarray:
    corrected = embedding.copy()
    for batch in pd.unique(batch_labels):
        mask = batch_labels == batch
        subset = corrected[mask]
        mean = subset.mean(axis=0, keepdims=True)
        std = np.clip(subset.std(axis=0, keepdims=True), a_min=1e-6, a_max=None)
        corrected[mask] = (subset - mean) / std
    return corrected


def compute_sample_silhouette(
    embedding: np.ndarray,
    batch_labels: np.ndarray,
    max_cells: int = 5000,
    random_state: int = 0,
) -> float:
    labels = batch_labels.astype(str)
    if embedding.shape[0] < 3 or pd.unique(labels).shape[0] < 2:
        return float("nan")
    rng = np.random.default_rng(random_state)
    if embedding.shape[0] > max_cells:
        chosen = np.sort(rng.choice(embedding.shape[0], size=max_cells, replace=False))
        embedding = embedding[chosen]
        labels = labels[chosen]
    return float(silhouette_score(embedding, labels, metric="euclidean"))


def apply_posthoc_batch_correction(
    adata,
    input_key: str,
    batch_key: str,
    method: Optional[str],
    output_key: Optional[str] = None,
) -> tuple[str, dict[str, float | str]]:
    normalized = normalize_posthoc_batch_method(method)
    output_key = output_key or corrected_embedding_key(input_key, normalized)

    if input_key not in adata.obsm:
        raise KeyError(f"Missing adata.obsm[{input_key!r}]")
    if batch_key not in adata.obs:
        raise KeyError(f"Missing adata.obs[{batch_key!r}]")

    raw_embedding = np.asarray(adata.obsm[input_key], dtype=np.float32)
    batch_labels = adata.obs[batch_key].astype(str).to_numpy()

    if normalized == "none":
        corrected = raw_embedding.copy()
    elif normalized == "combat":
        corrected = _combat_correct(raw_embedding, batch_labels)
    elif normalized == "center":
        corrected = _center_correct(raw_embedding, batch_labels)
    else:
        corrected = _zscore_correct(raw_embedding, batch_labels)

    adata.obsm[output_key] = corrected
    metrics = {
        "method": normalized,
        "input_key": input_key,
        "output_key": output_key,
        "sample_silhouette_before": compute_sample_silhouette(raw_embedding, batch_labels),
        "sample_silhouette_after": compute_sample_silhouette(corrected, batch_labels),
    }
    return output_key, metrics


def _normalize_output_prefix_name(output_prefix_name: str) -> str:
    path = Path(output_prefix_name).name
    if path.endswith(".h5ad"):
        return Path(path).stem
    return path


def build_horizontal_joint_adata(
    sample_count: int,
    batch_key: str,
    random_seed: int,
    max_cells_per_sample: int = 0,
    input_h5ad_path: str | Path | None = None,
    processed_root: str | Path | None = None,
    sample_names: Optional[list[str]] = None,
) -> tuple[sc.AnnData, dict[str, object]]:
    if sample_count < 2:
        raise ValueError("Horizontal integration requires at least two samples.")

    sample_names = list(sample_names or [])
    if input_h5ad_path is not None:
        input_h5ad_path = Path(input_h5ad_path).expanduser()
        if not input_h5ad_path.exists():
            raise FileNotFoundError(f"找不到输入文件: {input_h5ad_path}")
        joint_adata = sc.read_h5ad(input_h5ad_path)
        if batch_key not in joint_adata.obs.columns:
            raise ValueError(f"输入文件缺少 obs[{batch_key!r}]: {input_h5ad_path}")
        observed_samples = sorted(joint_adata.obs[batch_key].astype(str).unique().tolist())
        if len(observed_samples) != sample_count:
            raise ValueError(
                f"sample_count={sample_count}，但输入文件 obs[{batch_key!r}] 中实际有 {len(observed_samples)} 个样本。"
            )
        if sample_names and sorted(sample_names) != observed_samples:
            raise ValueError(
                f"显式传入的 sample_names 与输入文件中的样本不一致: {sample_names} vs {observed_samples}"
            )
        return joint_adata, {
            "input_mode": "merged_h5ad",
            "input_h5ad": str(input_h5ad_path),
            "samples": observed_samples,
            "sample_count": int(sample_count),
        }

    if processed_root is None:
        raise ValueError("未提供 input_h5ad_path 时，必须显式提供 processed_root。")
    if len(sample_names) != sample_count:
        raise ValueError(
            f"sample_count={sample_count}，但显式传入的 sample_names 数量为 {len(sample_names)}。"
        )

    processed_root = Path(processed_root).expanduser()
    if not processed_root.exists():
        raise FileNotFoundError(f"找不到 processed_root: {processed_root}")

    adata_map: dict[str, sc.AnnData] = {}
    per_sample_n_obs: dict[str, int] = {}
    feature_type_map: dict[str, str] = {}
    highly_variable_map: dict[str, bool] = {}
    rng = np.random.default_rng(random_seed)

    for sample_name in sample_names:
        input_h5ad_path = processed_root / f"{sample_name}.h5ad"
        if not input_h5ad_path.exists():
            raise FileNotFoundError(f"找不到样本文件: {input_h5ad_path}")

        adata = sc.read_h5ad(input_h5ad_path)
        adata.obs_names = adata.obs_names.astype(str)
        adata.obs["original_obs_name"] = adata.obs_names
        adata.var = adata.var.copy()
        adata.var_names = adata.var["name"].astype(str).values
        if adata.var_names.has_duplicates:
            raise ValueError(f"{sample_name} 存在重复特征名，当前无法自动处理。")
        adata.var["name"] = adata.var_names.astype(str)

        if max_cells_per_sample > 0 and adata.n_obs > max_cells_per_sample:
            chosen = np.sort(rng.choice(adata.n_obs, size=max_cells_per_sample, replace=False))
            adata = adata[chosen].copy()

        for feature_name, feature_type in zip(adata.var["name"].astype(str), adata.var["type"].astype(str)):
            previous_type = feature_type_map.get(feature_name)
            if previous_type is not None and previous_type != feature_type:
                raise ValueError(
                    f"特征 {feature_name!r} 在不同样本中的类型不一致: {previous_type} vs {feature_type}"
                )
            feature_type_map[feature_name] = feature_type

        if "highly_variable_moranI" in adata.var.columns:
            for feature_name, feature_flag in zip(
                adata.var["name"].astype(str),
                adata.var["highly_variable_moranI"].astype(bool),
            ):
                highly_variable_map[feature_name] = bool(highly_variable_map.get(feature_name, False) or feature_flag)

        adata_map[sample_name] = adata
        per_sample_n_obs[sample_name] = int(adata.n_obs)

    joint_adata = ad.concat(
        adata_map,
        label=batch_key,
        index_unique="__",
        join="outer",
        merge="same",
        fill_value=0,
    )
    joint_adata.var = joint_adata.var.copy()
    joint_adata.var["name"] = joint_adata.var_names.astype(str)
    joint_adata.var["type"] = [feature_type_map[name] for name in joint_adata.var_names.astype(str)]
    if highly_variable_map:
        joint_adata.var["highly_variable_moranI"] = [
            bool(highly_variable_map.get(name, False)) for name in joint_adata.var_names.astype(str)
        ]
    joint_adata.obs[batch_key] = joint_adata.obs[batch_key].astype(str)

    return joint_adata, {
        "input_mode": "processed_concat",
        "processed_root": str(processed_root),
        "samples": list(sample_names),
        "sample_count": int(sample_count),
        "per_sample_n_obs": per_sample_n_obs,
        "n_features_after_outer_join": int(joint_adata.n_vars),
    }


def train_horizontal_spatial_model(
    output_prefix_path: str | Path,
    sample_count: int,
    batch_key: str,
    device: str,
    max_epoch: int,
    n_per_batch: int,
    proj_dim: int,
    token_dim: int,
    n_latent: int,
    num_prototypes: int,
    max_cells_per_sample: int,
    random_seed: int,
    cluster_random_seed: int,
    dropout_rate: float,
    cluster_n_neighbors: int,
    cluster_resolution: float,
    reconstruction_st_weight: float,
    reconstruction_sm_weight: float,
    dec_weight: float,
    hete_weight: float,
    homo_weight: float,
    horizontal_weight: float,
    hete_warmup_epochs: int,
    homo_warmup_epochs: int,
    horizontal_warmup_epochs: int,
    kl_weight: float,
    n_epochs_kl_warmup: int,
    lr: float,
    weight_decay: float,
    reconstruction_reduction: str,
    reconstruction_method_st: str,
    reconstruction_method_sm: str,
    balance_start_epoch: int,
    balance_ema: float,
    balance_weight_floor: float,
    spatial_coord_hidden_dim: int,
    spatial_context_hidden_dim: int,
    spatial_context_k: int,
    spatial_encoder_mode: str,
    spatial_fourier_scales: tuple[float, ...],
    spatial_token_scale: float,
    spatial_token_dropout: float,
    spatial_consistency_weight: float,
    spatial_consistency_warmup_epochs: int,
    spatial_contrastive_weight: float,
    spatial_contrastive_warmup_epochs: int,
    spatial_contrastive_pos_k: int,
    spatial_contrastive_neg_k: int,
    spatial_contrastive_temperature: float,
    spatial_contrastive_neg_strategy: str,
    standardize_inputs: bool,
    standardized_reconstruction: bool,
    deterministic: bool,
    deterministic_warn_only: bool,
    spatial_consistency_use_all_latent: bool,
    spatial_contrastive_use_all_latent: bool,
    batch_embedding: str,
    batch_hidden_dim: int,
    posthoc_batch_method: str,
    input_h5ad_path: str | Path | None = None,
    processed_root: str | Path | None = None,
    sample_names: Optional[list[str]] = None,
) -> HorizontalTrainingResult:
    output_prefix_path = Path(output_prefix_path).expanduser()
    loss_figure_path = output_prefix_path.parent / f"{output_prefix_path.name}_loss.png"
    output_h5ad_path = output_prefix_path.parent / f"{output_prefix_path.name}.h5ad"
    loss_csv_path = output_prefix_path.parent / f"{output_prefix_path.name}_loss.csv"
    param_json_path = output_prefix_path.parent / f"{output_prefix_path.name}_params.json"

    print(f"[train] output_prefix={output_prefix_path}", flush=True)
    print(f"[train] device={device}", flush=True)
    print(f"[train] batch_key={batch_key}", flush=True)
    print(f"[train] sample_count={sample_count}", flush=True)

    seed_everything(random_seed, deterministic, deterministic_warn_only)
    for save_path in [loss_figure_path, output_h5ad_path, loss_csv_path, param_json_path]:
        save_path.parent.mkdir(parents=True, exist_ok=True)

    print("[train] building merged joint_adata", flush=True)
    joint_adata, input_metadata = build_horizontal_joint_adata(
        sample_count=sample_count,
        batch_key=batch_key,
        random_seed=random_seed,
        max_cells_per_sample=max_cells_per_sample,
        input_h5ad_path=input_h5ad_path,
        processed_root=processed_root,
        sample_names=sample_names,
    )

    validate_spadta_model_input(
        joint_adata,
        expression_graph_k=spatial_contrastive_pos_k,
        spatial_context_k=spatial_context_k,
    )

    print("[train] building horizontal model", flush=True)
    model = DecAlignSpatialMetaLinearHorizontal(
        joint_adata,
        proj_dim=proj_dim,
        token_dim=token_dim,
        n_latent=n_latent,
        num_prototypes=num_prototypes,
        dropout_rate=dropout_rate,
        device=device,
        reconstruction_method_st=reconstruction_method_st,
        reconstruction_method_sm=reconstruction_method_sm,
        standardize_inputs=standardize_inputs,
        use_standardized_reconstruction=standardized_reconstruction,
        spatial_hidden_dim=spatial_coord_hidden_dim,
        spatial_context_hidden_dim=spatial_context_hidden_dim,
        spatial_context_k=spatial_context_k,
        spatial_encoder_mode=spatial_encoder_mode,
        spatial_fourier_scales=spatial_fourier_scales,
        spatial_token_scale=spatial_token_scale,
        spatial_token_dropout=spatial_token_dropout,
        spatial_contrastive_pos_k=spatial_contrastive_pos_k,
        spatial_contrastive_neg_k=spatial_contrastive_neg_k,
        spatial_contrastive_temperature=spatial_contrastive_temperature,
        spatial_contrastive_neg_strategy=spatial_contrastive_neg_strategy,
        batch_keys=[batch_key],
        batch_embedding=batch_embedding,
        batch_hidden_dim=batch_hidden_dim,
    )

    print("[train] start fit", flush=True)
    loss_dict = model.fit(
        max_epoch=max_epoch,
        n_per_batch=n_per_batch,
        reconstruction_reduction=reconstruction_reduction,
        reconstruction_st_weight=reconstruction_st_weight,
        reconstruction_sm_weight=reconstruction_sm_weight,
        dec_weight=dec_weight,
        hete_weight=hete_weight,
        homo_weight=homo_weight,
        horizontal_weight=horizontal_weight,
        hete_warmup_epochs=hete_warmup_epochs,
        homo_warmup_epochs=homo_warmup_epochs,
        horizontal_warmup_epochs=horizontal_warmup_epochs,
        kl_weight=kl_weight,
        n_epochs_kl_warmup=n_epochs_kl_warmup,
        lr=lr,
        weight_decay=weight_decay,
        random_seed=random_seed,
        balance_start_epoch=balance_start_epoch,
        balance_ema=balance_ema,
        task_weight_floor=balance_weight_floor,
        spatial_consistency_weight=spatial_consistency_weight,
        spatial_consistency_warmup_epochs=spatial_consistency_warmup_epochs,
        spatial_consistency_use_all_latent=spatial_consistency_use_all_latent,
        spatial_contrastive_weight=spatial_contrastive_weight,
        spatial_contrastive_warmup_epochs=spatial_contrastive_warmup_epochs,
        spatial_contrastive_use_all_latent=spatial_contrastive_use_all_latent,
    )

    loss_df = pd.DataFrame(loss_dict)
    loss_df.to_csv(loss_csv_path, index=False)

    n_columns = len(loss_df.columns)
    n_plot_cols = 3
    n_plot_rows = int(np.ceil(n_columns / n_plot_cols))
    fig, axes = plt.subplots(n_plot_rows, n_plot_cols, figsize=(20, 3.5 * n_plot_rows))
    axes = np.atleast_1d(axes).flatten()
    for axis, column_name in zip(axes, loss_df.columns):
        axis.plot(loss_df[column_name].values)
        axis.set_title(column_name)
    for axis in axes[len(loss_df.columns):]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(loss_figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("[train] exporting embeddings and clustering", flush=True)
    latent_embedding = model.get_latent_embedding()
    reconstruction = model.get_normalized_expression()
    contribution = model.get_modality_contribution()
    contribution_details = model.get_modality_contribution_details()

    joint_adata.layers["reconstruction_decalign_linear"] = reconstruction
    joint_adata.obsm["X_emb_decalign_linear"] = latent_embedding
    corrected_emb_key, correction_metrics = apply_posthoc_batch_correction(
        joint_adata,
        input_key="X_emb_decalign_linear",
        batch_key=batch_key,
        method=posthoc_batch_method,
        output_key=corrected_embedding_key("X_emb_decalign_linear", posthoc_batch_method),
    )
    joint_adata.obs["contribution_st_decalign_linear"] = contribution
    joint_adata.obs["contribution_sm_decalign_linear"] = contribution_details.get("contribution_sm", 1 - contribution)
    joint_adata.uns["contribution_method_decalign_linear"] = "spatialmeta_like_angular_similarity_to_homo_joint"
    joint_adata.uns["spatial_encoder_mode_decalign_linear"] = spatial_encoder_mode
    joint_adata.uns["horizontal_integration_decalign_linear"] = {
        "enabled": True,
        "batch_key": batch_key,
        "batch_embedding": batch_embedding,
        "batch_hidden_dim": batch_hidden_dim,
        "samples": input_metadata.get("samples", []),
        "sample_count": input_metadata.get("sample_count", sample_count),
        "horizontal_weight": horizontal_weight,
        "horizontal_warmup_epochs": horizontal_warmup_epochs,
        "posthoc_batch_method": posthoc_batch_method,
        "posthoc_batch_output_key": corrected_emb_key,
        "sample_silhouette_before": correction_metrics["sample_silhouette_before"],
        "sample_silhouette_after": correction_metrics["sample_silhouette_after"],
    }

    for key, obs_name in {
        "similarity_st_joint": "similarity_st_joint_decalign_linear",
        "similarity_sm_joint": "similarity_sm_joint_decalign_linear",
        "spatial_gate_mean": "spatial_gate_mean_decalign_linear",
        "spatial_st_gate_mean": "spatial_st_gate_mean_decalign_linear",
        "spatial_sm_gate_mean": "spatial_sm_gate_mean_decalign_linear",
        "spatial_token_scale": "spatial_token_scale_decalign_linear",
    }.items():
        if key in contribution_details:
            joint_adata.obs[obs_name] = contribution_details[key]
    for key, obsm_name in {
        "homo_st_embedding": "X_emb_homo_st_decalign_linear",
        "homo_sm_embedding": "X_emb_homo_sm_decalign_linear",
        "homo_joint_embedding": "X_emb_homo_joint_decalign_linear",
    }.items():
        if key in contribution_details:
            joint_adata.obsm[obsm_name] = contribution_details[key]

    sc.pp.neighbors(
        joint_adata,
        use_rep=corrected_emb_key,
        n_neighbors=cluster_n_neighbors,
        random_state=cluster_random_seed,
    )
    sc.tl.umap(joint_adata, min_dist=1, spread=1, random_state=cluster_random_seed)
    sc.tl.leiden(
        joint_adata,
        resolution=cluster_resolution,
        key_added="decalign_linear_clusters",
        random_state=cluster_random_seed,
    )
    joint_adata.write_h5ad(output_h5ad_path)

    param_json_path.write_text(
        json.dumps(
            {
                "model_variant": MODEL_VARIANT,
                "output_h5ad": str(output_h5ad_path),
                "proj_dim": proj_dim,
                "token_dim": token_dim,
                "n_latent": n_latent,
                "num_prototypes": num_prototypes,
                "dropout_rate": dropout_rate,
                "device": device,
                "max_epoch": max_epoch,
                "n_per_batch": n_per_batch,
                "max_cells_per_sample": max_cells_per_sample,
                "random_seed": random_seed,
                "cluster_random_seed": cluster_random_seed,
                "reconstruction_st_weight": reconstruction_st_weight,
                "reconstruction_sm_weight": reconstruction_sm_weight,
                "dec_weight": dec_weight,
                "hete_weight": hete_weight,
                "homo_weight": homo_weight,
                "horizontal_weight": horizontal_weight,
                "hete_warmup_epochs": hete_warmup_epochs,
                "homo_warmup_epochs": homo_warmup_epochs,
                "horizontal_warmup_epochs": horizontal_warmup_epochs,
                "kl_weight": kl_weight,
                "n_epochs_kl_warmup": n_epochs_kl_warmup,
                "posthoc_batch_method": posthoc_batch_method,
                "posthoc_batch_output_key": corrected_emb_key,
                "sample_silhouette_before": correction_metrics["sample_silhouette_before"],
                "sample_silhouette_after": correction_metrics["sample_silhouette_after"],
                "cluster_n_neighbors": cluster_n_neighbors,
                "cluster_resolution": cluster_resolution,
                "lr": lr,
                "weight_decay": weight_decay,
                "spatial_encoder_mode": spatial_encoder_mode,
                "spatial_context_k": spatial_context_k,
                "spatial_token_scale_initial": spatial_token_scale,
                "spatial_token_dropout": spatial_token_dropout,
                "spatial_contrastive_weight": spatial_contrastive_weight,
                "spatial_contrastive_pos_k": spatial_contrastive_pos_k,
                "spatial_contrastive_neg_k": spatial_contrastive_neg_k,
                "batch_key": batch_key,
                "batch_embedding": batch_embedding,
                "batch_hidden_dim": batch_hidden_dim,
                "input_metadata": input_metadata,
                "fit_metadata": getattr(model, "fit_metadata", {}),
                "loss_figure": str(loss_figure_path),
                "loss_csv": str(loss_csv_path),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"训练完成，结果已保存到: {output_h5ad_path}", flush=True)
    print(f"损失曲线: {loss_figure_path}", flush=True)
    print(f"参数文件: {param_json_path}", flush=True)
    return HorizontalTrainingResult(
        model=model,
        adata=joint_adata,
        output_h5ad_path=output_h5ad_path,
        loss_figure_path=loss_figure_path,
        loss_csv_path=loss_csv_path,
        param_json_path=param_json_path,
        loss_df=loss_df,
        input_metadata=input_metadata,
    )


def run_horizontal_samples(
    sample_count: int,
    output_root: str | Path,
    config_name: str,
    output_prefix_name: str,
    train_kwargs: dict[str, object],
    input_h5ad_path: str | Path | None = None,
    processed_root: str | Path | None = None,
    sample_names: Optional[list[str]] = None,
) -> HorizontalTrainingResult:
    output_root = Path(output_root).expanduser()
    output_prefix_name = _normalize_output_prefix_name(output_prefix_name)
    output_prefix_path = output_root / config_name / output_prefix_name

    print(f"[run] config_name={config_name}", flush=True)
    print(f"[run] output_root={output_root}", flush=True)
    print(f"[run] output_prefix_name={output_prefix_name}", flush=True)
    if input_h5ad_path is not None:
        print(f"[run] input_h5ad={Path(input_h5ad_path).expanduser()}", flush=True)
    else:
        print(f"[run] processed_root={Path(processed_root).expanduser()}", flush=True)
        print(f"[run] sample_names={sample_names}", flush=True)

    return train_horizontal_spatial_model(
        output_prefix_path=output_prefix_path,
        sample_count=sample_count,
        input_h5ad_path=input_h5ad_path,
        processed_root=processed_root,
        sample_names=sample_names,
        **train_kwargs,
    )
