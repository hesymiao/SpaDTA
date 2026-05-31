from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from pathlib import Path
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import torch

from .model import DecAlignSpatialMetaLinear
from .preprocess import normalize_total_joint_adata_sm_st


warnings.filterwarnings("ignore")


@dataclass
class TrainingResult:
    model: DecAlignSpatialMetaLinear
    adata: sc.AnnData
    output_h5ad_path: Path
    loss_figure_path: Path
    loss_csv_path: Path
    loss_df: pd.DataFrame


@dataclass
class SingleRunResult:
    training: TrainingResult
    metrics_df: pd.DataFrame


def seed_everything(seed: int, deterministic: bool, warn_only: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_spatial_model(
    input_h5ad_path: str | Path,
    output_prefix_path: str | Path,
    device: str,
    max_epoch: int,
    n_per_batch: int,
    proj_dim: int,
    token_dim: int,
    n_latent: int,
    num_prototypes: int,
    max_cells: int,
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
    hete_warmup_epochs: int,
    homo_warmup_epochs: int,
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
) -> TrainingResult:
    input_h5ad_path = Path(input_h5ad_path).expanduser()
    output_prefix_path = Path(output_prefix_path).expanduser()
    loss_figure_path = output_prefix_path.parent / f"{output_prefix_path.name}_loss.png"
    output_h5ad_path = output_prefix_path.parent / f"{output_prefix_path.name}.h5ad"
    loss_csv_path = output_prefix_path.parent / f"{output_prefix_path.name}_loss.csv"

    if not input_h5ad_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_h5ad_path}")

    print(f"[train] input_h5ad={input_h5ad_path}", flush=True)
    print(f"[train] output_prefix={output_prefix_path}", flush=True)
    print(f"[train] device={device}", flush=True)

    seed_everything(random_seed, deterministic, deterministic_warn_only)
    for save_path in [loss_figure_path, output_h5ad_path, loss_csv_path]:
        save_path.parent.mkdir(parents=True, exist_ok=True)

    print("[train] loading h5ad", flush=True)
    joint_adata = sc.read_h5ad(input_h5ad_path)
    if max_cells > 0 and joint_adata.n_obs > max_cells:
        joint_adata = joint_adata[:max_cells].copy()

    if "counts" in joint_adata.layers:
        joint_adata.X = joint_adata.layers["counts"].copy()
        normalize_total_joint_adata_sm_st(
            joint_adata,
            target_sum_SM=1e3,
            target_sum_ST=None,
        )

    print("[train] building model", flush=True)
    model = DecAlignSpatialMetaLinear(
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
        hete_warmup_epochs=hete_warmup_epochs,
        homo_warmup_epochs=homo_warmup_epochs,
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

    plot_column_count = len(loss_df.columns)
    plot_cols = 3
    plot_rows = int(np.ceil(plot_column_count / plot_cols))
    fig, axes = plt.subplots(plot_rows, plot_cols, figsize=(20, 3.5 * plot_rows))
    axes = np.atleast_1d(axes).flatten()
    for axis, column_name in zip(axes, loss_df.columns):
        axis.plot(loss_df[column_name].values)
        axis.set_title(column_name)
    for axis in axes[len(loss_df.columns):]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(loss_figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    latent_embedding = model.get_latent_embedding()
    reconstruction = model.get_normalized_expression()
    contribution = model.get_modality_contribution()
    contribution_details = model.get_modality_contribution_details()

    joint_adata.layers["reconstruction_decalign_linear"] = reconstruction
    joint_adata.obsm["X_emb_decalign_linear"] = latent_embedding
    joint_adata.obs["contribution_st_decalign_linear"] = contribution
    joint_adata.obs["contribution_sm_decalign_linear"] = contribution_details.get("contribution_sm", 1 - contribution)
    joint_adata.uns["contribution_method_decalign_linear"] = "spatialmeta_like_angular_similarity_to_homo_joint"
    joint_adata.uns["spatial_encoder_mode_decalign_linear"] = spatial_encoder_mode

    if "homo_st_embedding" in contribution_details:
        joint_adata.obsm["X_emb_homo_st_decalign_linear"] = contribution_details["homo_st_embedding"]
    if "homo_sm_embedding" in contribution_details:
        joint_adata.obsm["X_emb_homo_sm_decalign_linear"] = contribution_details["homo_sm_embedding"]
    if "homo_joint_embedding" in contribution_details:
        joint_adata.obsm["X_emb_homo_joint_decalign_linear"] = contribution_details["homo_joint_embedding"]

    sc.pp.neighbors(
        joint_adata,
        use_rep="X_emb_decalign_linear",
        n_neighbors=cluster_n_neighbors,
        random_state=cluster_random_seed,
    )
    sc.tl.umap(joint_adata, random_state=cluster_random_seed)
    sc.tl.leiden(
        joint_adata,
        resolution=cluster_resolution,
        key_added="decalign_linear_clusters",
        random_state=cluster_random_seed,
    )
    joint_adata.write_h5ad(output_h5ad_path)

    print(f"训练完成，结果已保存到: {output_h5ad_path}")
    print(f"损失曲线: {loss_figure_path}")
    return TrainingResult(
        model=model,
        adata=joint_adata,
        output_h5ad_path=output_h5ad_path,
        loss_figure_path=loss_figure_path,
        loss_csv_path=loss_csv_path,
        loss_df=loss_df,
    )


def compute_clustering_metrics(eval_df: pd.DataFrame, gt_key: str, pred_key: str) -> tuple[float, float]:
    if gt_key not in eval_df or pred_key not in eval_df:
        return np.nan, np.nan

    valid_mask = eval_df[gt_key].notna() & eval_df[pred_key].notna()
    if not valid_mask.any():
        return np.nan, np.nan

    y_true = eval_df.loc[valid_mask, gt_key].astype(str).values
    y_pred = eval_df.loc[valid_mask, pred_key].astype(str).values
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    return ari, nmi


def load_minimal_adata(path: str | Path) -> tuple[sc.AnnData, pd.DataFrame, np.ndarray | None]:
    adata = sc.read_h5ad(path, backed="r")
    obs = adata.obs.copy()
    obs.index = obs.index.astype(str)
    spatial = np.asarray(adata.obsm["spatial"]) if "spatial" in adata.obsm else None
    return adata, obs, spatial


def evaluate_lightweight(
    sample_name: str,
    gt_path: str | Path,
    pred_path: str | Path,
    pred_key: str,
) -> dict[str, object]:
    adata_gt, gt_obs, gt_spatial = load_minimal_adata(gt_path)
    adata_pred, pred_obs, pred_spatial = load_minimal_adata(pred_path)

    try:
        gt_key = "pathological_annotation" if "pathological_annotation" in gt_obs.columns else "annotation"
        if gt_key not in gt_obs.columns:
            raise KeyError("真值文件中不存在 pathological_annotation 或 annotation 列。")
        if pred_key not in pred_obs.columns:
            raise KeyError(f"预测结果中不存在聚类列: {pred_key}")
        if gt_spatial is None or pred_spatial is None:
            raise RuntimeError("缺少 spatial 坐标，无法执行空间匹配。")

        tree = cKDTree(gt_spatial)
        distances, indices = tree.query(pred_spatial, k=1)
        matched_mask = distances < 5.0
        if int(np.sum(matched_mask)) <= 10:
            raise RuntimeError("空间匹配失败，匹配到的 spot 太少。")

        matched_pred_obs = pred_obs.iloc[np.where(matched_mask)[0]].copy()
        matched_pred_obs["GT"] = gt_obs[gt_key].to_numpy()[indices[matched_mask]]
        ari, nmi = compute_clustering_metrics(matched_pred_obs, "GT", pred_key)
        return {
            "sample": sample_name,
            "pred_file": str(pred_path),
            "gt_file": str(gt_path),
            "matched_spots": int(len(matched_pred_obs)),
            "matching_mode": "spatial",
            "pred_key": pred_key,
            "pred_clusters": int(pd.Series(matched_pred_obs[pred_key]).dropna().nunique()),
            "ARI": ari,
            "NMI": nmi,
            "pcc_embedding_self": np.nan,
            "RMSE": np.nan,
            "Cell_PCC": np.nan,
            "Gene_PCC": np.nan,
        }
    finally:
        if getattr(adata_gt, "file", None) is not None:
            adata_gt.file.close()
        if getattr(adata_pred, "file", None) is not None:
            adata_pred.file.close()


def evaluate_clustering(
    sample_name: str,
    gt_path: str | Path,
    pred_path: str | Path,
    output_csv_path: str | Path,
    pred_key: str = "decalign_linear_clusters",
    light_eval: bool = True,
) -> pd.DataFrame:
    gt_path = Path(gt_path).expanduser()
    pred_path = Path(pred_path).expanduser()
    output_csv_path = Path(output_csv_path).expanduser()

    if not gt_path.exists():
        raise FileNotFoundError(f"找不到真值文件: {gt_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"找不到预测结果文件: {pred_path}")

    if light_eval:
        metrics = evaluate_lightweight(
            sample_name=sample_name,
            gt_path=gt_path,
            pred_path=pred_path,
            pred_key=pred_key,
        )
    else:
        adata_gt = sc.read_h5ad(gt_path)
        adata_pred = sc.read_h5ad(pred_path)
        adata_gt.obs_names = adata_gt.obs_names.astype(str)
        adata_pred.obs_names = adata_pred.obs_names.astype(str)

        gt_key = "pathological_annotation" if "pathological_annotation" in adata_gt.obs else "annotation"
        if "spatial" not in adata_gt.obsm or "spatial" not in adata_pred.obsm:
            raise RuntimeError("缺少 spatial 坐标，无法执行空间匹配。")

        tree = cKDTree(adata_gt.obsm["spatial"])
        distances, indices = tree.query(adata_pred.obsm["spatial"], k=1)
        matched_mask = distances < 5.0
        if np.sum(matched_mask) <= 10:
            raise RuntimeError("空间匹配失败，匹配到的 spot 太少。")

        adata_eval = adata_pred[matched_mask].copy()
        adata_eval.obs["GT"] = adata_gt.obs[gt_key].values[indices[matched_mask]]
        if pred_key not in adata_eval.obs:
            raise KeyError(f"预测结果中不存在聚类列: {pred_key}")

        ari, nmi = compute_clustering_metrics(adata_eval.obs, "GT", pred_key)
        metrics = {
            "sample": sample_name,
            "pred_file": str(pred_path),
            "gt_file": str(gt_path),
            "matched_spots": int(adata_eval.n_obs),
            "matching_mode": "spatial",
            "pred_key": pred_key,
            "pred_clusters": int(adata_eval.obs[pred_key].nunique()),
            "ARI": ari,
            "NMI": nmi,
            "pcc_embedding_self": np.nan,
            "RMSE": np.nan,
            "Cell_PCC": np.nan,
            "Gene_PCC": np.nan,
        }

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame([metrics])
    result_df.to_csv(output_csv_path, index=False)
    print(result_df.to_string(index=False))
    print(f"\n结果已保存到: {output_csv_path}")
    return result_df


def run_single_sample(
    sample_name: str,
    processed_root: str | Path,
    gt_root: str | Path,
    output_root: str | Path,
    config_name: str,
    train_kwargs: dict[str, object],
    output_prefix_name: str | None = None,
) -> SingleRunResult:
    processed_root = Path(processed_root)
    gt_root = Path(gt_root)
    output_root = Path(output_root)

    input_h5ad_path = processed_root / f"{sample_name}.h5ad"
    gt_h5ad_path = gt_root / f"adata_joint_{sample_name}_hvf2800.h5ad"
    if output_prefix_name is None:
        output_prefix_name = f"{sample_name}_{config_name}"
    output_prefix_name = Path(output_prefix_name).name
    if output_prefix_name.endswith(".h5ad"):
        output_prefix_name = Path(output_prefix_name).stem
    output_prefix_path = output_root / config_name / output_prefix_name
    output_csv_path = output_root / config_name / f"{sample_name}_metrics_full.csv"

    print(f"[run] sample={sample_name}", flush=True)
    print(f"[run] gt_h5ad={gt_h5ad_path}", flush=True)
    print(f"[run] output_csv={output_csv_path}", flush=True)

    training_result = train_spatial_model(
        input_h5ad_path=input_h5ad_path,
        output_prefix_path=output_prefix_path,
        **train_kwargs,
    )
    print("[run] start evaluation", flush=True)
    metrics_df = evaluate_clustering(
        sample_name=sample_name,
        gt_path=gt_h5ad_path,
        pred_path=training_result.output_h5ad_path,
        output_csv_path=output_csv_path,
        pred_key="decalign_linear_clusters",
        light_eval=True,
    )
    return SingleRunResult(
        training=training_result,
        metrics_df=metrics_df,
    )


def run_all_samples(
    sample_names: list[str],
    train_kwargs_by_sample: dict[str, dict[str, object]],
    processed_root: str | Path,
    gt_root: str | Path,
    output_root: str | Path,
    config_name: str,
) -> pd.DataFrame:
    output_root = Path(output_root)
    summary_rows: list[pd.DataFrame] = []

    for sample_name in sample_names:
        run_single_sample(
            sample_name=sample_name,
            processed_root=processed_root,
            gt_root=gt_root,
            output_root=output_root,
            config_name=config_name,
            train_kwargs=train_kwargs_by_sample[sample_name],
        )
        metrics_path = output_root / config_name / f"{sample_name}_metrics_full.csv"
        if metrics_path.exists():
            summary_rows.append(pd.read_csv(metrics_path))

    if summary_rows:
        summary_df = pd.concat(summary_rows, ignore_index=True)
        summary_df.to_csv(output_root / config_name / "metrics_full.csv", index=False)
        summary_df[["sample", "pred_clusters", "ARI", "NMI"]].to_csv(
            output_root / config_name / "metrics.csv",
            index=False,
        )
        return summary_df
    return pd.DataFrame()


def run_parallel_jobs(
    jobs: list[dict[str, str | float]],
    processed_root: str | Path,
    gt_root: str | Path,
    output_root: str | Path,
    config_name: str,
    max_workers: int = 3,
) -> None:
    processed_root = Path(processed_root)
    gt_root = Path(gt_root)
    output_root = Path(output_root)

    def run_job(job: dict[str, str | float]) -> str:
        sample_name = str(job["sample_name"])
        device = str(job["device"])
        cluster_resolution = float(job["cluster_resolution"])
        run_single_sample(
            sample_name=sample_name,
            processed_root=processed_root,
            gt_root=gt_root,
            output_root=output_root,
            config_name=config_name,
            device=device,
            cluster_resolution=cluster_resolution,
        )
        return sample_name

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(run_job, job): str(job["sample_name"]) for job in jobs}
        for future in concurrent.futures.as_completed(future_map):
            sample_name = future_map[future]
            finished_name = future.result()
            print(f"[done] {sample_name} -> {finished_name}", flush=True)


def as_array(value):
    if sparse.issparse(value):
        return value.toarray()
    return np.asarray(value)


def compare_csv(path_before: Path, path_after: Path, label: str) -> list[str]:
    if not path_before.exists() or not path_after.exists():
        return [f"{label}: missing file"]
    before = pd.read_csv(path_before)
    after = pd.read_csv(path_after)
    if before.equals(after):
        return []
    return [f"{label}: csv mismatch"]


def compare_optional_csv(path_before: Path, path_after: Path, label: str) -> list[str]:
    if not path_before.exists() and not path_after.exists():
        return []
    return compare_csv(path_before, path_after, label)


def compare_h5ad(
    path_before: Path,
    path_after: Path,
    sample_name: str,
    obs_keys: list[str],
    obsm_keys: list[str],
    layer_keys: list[str],
) -> list[str]:
    issues: list[str] = []
    if not path_before.exists() or not path_after.exists():
        return [f"{sample_name}: missing h5ad"]

    before = sc.read_h5ad(path_before)
    after = sc.read_h5ad(path_after)
    try:
        if list(before.obs_names.astype(str)) != list(after.obs_names.astype(str)):
            issues.append(f"{sample_name}: obs_names mismatch")
        if list(before.var_names.astype(str)) != list(after.var_names.astype(str)):
            issues.append(f"{sample_name}: var_names mismatch")

        for key in obs_keys:
            if key in before.obs or key in after.obs:
                if key not in before.obs or key not in after.obs:
                    issues.append(f"{sample_name}: obs[{key}] missing on one side")
                    continue
                before_values = before.obs[key].astype(str).to_numpy() if str(before.obs[key].dtype) == "category" else before.obs[key].to_numpy()
                after_values = after.obs[key].astype(str).to_numpy() if str(after.obs[key].dtype) == "category" else after.obs[key].to_numpy()
                if not np.array_equal(before_values, after_values):
                    issues.append(f"{sample_name}: obs[{key}] mismatch")

        for key in obsm_keys:
            if key in before.obsm or key in after.obsm:
                if key not in before.obsm or key not in after.obsm:
                    issues.append(f"{sample_name}: obsm[{key}] missing on one side")
                    continue
                if not np.array_equal(as_array(before.obsm[key]), as_array(after.obsm[key])):
                    issues.append(f"{sample_name}: obsm[{key}] mismatch")

        for key in layer_keys:
            if key in before.layers or key in after.layers:
                if key not in before.layers or key not in after.layers:
                    issues.append(f"{sample_name}: layers[{key}] missing on one side")
                    continue
                if not np.array_equal(as_array(before.layers[key]), as_array(after.layers[key])):
                    issues.append(f"{sample_name}: layers[{key}] mismatch")
    finally:
        del before
        del after

    return issues


def compare_run_outputs(
    before_root: str | Path,
    after_root: str | Path,
    config_name: str,
    sample_names: list[str],
) -> list[str]:
    before_root = Path(before_root).expanduser()
    after_root = Path(after_root).expanduser()
    obs_keys = [
        "decalign_linear_clusters",
        "contribution_st_decalign_linear",
        "contribution_sm_decalign_linear",
    ]
    obsm_keys = [
        "X_emb_decalign_linear",
        "X_emb_homo_st_decalign_linear",
        "X_emb_homo_sm_decalign_linear",
        "X_emb_homo_joint_decalign_linear",
    ]
    layer_keys = [
        "reconstruction_decalign_linear",
    ]

    issues: list[str] = []
    issues.extend(
        compare_optional_csv(
            before_root / config_name / "metrics.csv",
            after_root / config_name / "metrics.csv",
            "metrics.csv",
        )
    )
    issues.extend(
        compare_optional_csv(
            before_root / config_name / "metrics_full.csv",
            after_root / config_name / "metrics_full.csv",
            "metrics_full.csv",
        )
    )

    for sample_name in sample_names:
        prefix = f"{sample_name}_{config_name}"
        issues.extend(
            compare_csv(
                before_root / config_name / f"{prefix}_loss.csv",
                after_root / config_name / f"{prefix}_loss.csv",
                f"{sample_name}: loss.csv",
            )
        )
        issues.extend(
            compare_h5ad(
                before_root / config_name / f"{prefix}.h5ad",
                after_root / config_name / f"{prefix}.h5ad",
                sample_name,
                obs_keys,
                obsm_keys,
                layer_keys,
            )
        )

    if issues:
        print("MISMATCH")
        for issue in issues:
            print(issue)
    else:
        print("MATCH")
    return issues
