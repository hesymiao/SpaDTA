from __future__ import annotations

from pathlib import Path
import json
import random
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from anndata import AnnData
from scipy import sparse
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors

package_root = Path(__file__).resolve().parents[2]
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))

from SpaDTA_718.model.model import DecAlignSpatialMetaLinear
from SpaDTA_718.model.preprocess import normalize_total_joint_adata_sm_st, prepare_spadta_model_input


project_root = Path("/data/user/hesy/projects/SpatialMETA")
processed_root = Path("/bigdat2/user/hesy/spatialmeta/SpatialMETA/processed")

sample_name = "X49_T"
gene = "FXYD2"

train_input_h5ad = processed_root / f"{sample_name}.h5ad"
highres_sm_h5ad = Path("/bigdat2/user/hesy/spatialmeta/SpatialMETA/ccRCC/adata_SM_X49_T_raw.h5ad")

root_dir = project_root / "SpaDTA_718" / "runs" / "sm_downstream" / "fig2i"
experiment_dir = root_dir / "sm_to_st_generation_spatial_top_third"
superres_dir = root_dir / "superres_eval_fixed_train_mean_libsize"
figure_dir = root_dir / "figures"

device = "cuda"
split_seed = 42
max_epoch = 256
n_per_batch = 128
proj_dim = 256
token_dim = 128
n_latent = 10
num_prototypes = 8
dropout_rate = 0.03
reconstruction_st_weight = 0.5
reconstruction_sm_weight = 0.5
dec_weight = 1.0
homo_weight = 0.05
homo_warmup_epochs = 0
kl_weight = 0.0
n_epochs_kl_warmup = 0
lr = 5e-4
weight_decay = 1e-6
reconstruction_method_st = "zinb"
reconstruction_method_sm = "g"
standardize_inputs = False
use_standardized_reconstruction = False
reconstruction_reduction = "mean"
spatial_hidden_dim = 128
spatial_context_hidden_dim = 128
spatial_context_k = 12
spatial_encoder_mode = "local_context"
spatial_token_scale = 0.5
spatial_token_dropout = 0.15
spatial_consistency_weight = 0.0
spatial_consistency_warmup_epochs = 16
spatial_contrastive_weight = 0.0
spatial_contrastive_warmup_epochs = 16
spatial_contrastive_pos_k = 4
spatial_contrastive_neg_k = 16
spatial_contrastive_temperature = 0.2
spatial_contrastive_neg_strategy = "mid"

knn_k = 8
target_scope = "eval_only"

layer = "generated_st_log1p_from_sm_celllevel"
title = f"{gene} cell level"
point_size = 5.0
background_point_size = 4.2
dpi = 240
cmap = "viridis"
clip_quantile = 0.99
title_fontsize = 18
colorbar_label_fontsize = 18
colorbar_tick_fontsize = 14
region_axis = "y"
region_side = "top"
region_fraction = 1.0 / 3.0
background_color = "#b8b8b8"
background_alpha = 0.88

excluded_transfer_keys = (
    "spatial_coords",
    "spatial_coord_mean",
    "spatial_coord_std",
    "spatial_coords_standardized",
    "spatial_neighbor_idx",
    "spatial_neighbor_rel",
    "spatial_neighbor_dist",
)


def log_stage(message: str) -> None:
    print(f"[fig2i] {message}", flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_joint_adata(path: Path) -> AnnData:
    adata = sc.read_h5ad(path)
    adata.obs_names = adata.obs_names.astype(str)
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
        normalize_total_joint_adata_sm_st(adata, target_sum_SM=1e3, target_sum_ST=None)
    elif "normalized" in adata.layers:
        adata.X = adata.layers["normalized"].copy()
    return adata


def prepare_model_adata(adata: AnnData) -> AnnData:
    return prepare_spadta_model_input(
        adata,
        modality="sm",
        expression_graph_k=spatial_contrastive_pos_k,
        spatial_context_k=spatial_context_k,
    )


def feature_key_series(adata: AnnData) -> pd.Series:
    if "name" in adata.var.columns:
        return adata.var["name"].astype(str)
    if "m/z" in adata.var.columns:
        return adata.var["m/z"].astype(str)
    return pd.Series(adata.var_names.astype(str), index=adata.var_names)


def build_model_config() -> dict[str, object]:
    return {
        "device": device,
        "proj_dim": proj_dim,
        "token_dim": token_dim,
        "n_latent": n_latent,
        "num_prototypes": num_prototypes,
        "dropout_rate": dropout_rate,
        "reconstruction_method_st": reconstruction_method_st,
        "reconstruction_method_sm": reconstruction_method_sm,
        "standardize_inputs": standardize_inputs,
        "use_standardized_reconstruction": use_standardized_reconstruction,
        "spatial_hidden_dim": spatial_hidden_dim,
        "spatial_context_hidden_dim": spatial_context_hidden_dim,
        "spatial_context_k": spatial_context_k,
        "spatial_encoder_mode": spatial_encoder_mode,
        "spatial_token_scale": spatial_token_scale,
        "spatial_token_dropout": spatial_token_dropout,
        "spatial_contrastive_pos_k": spatial_contrastive_pos_k,
        "spatial_contrastive_neg_k": spatial_contrastive_neg_k,
        "spatial_contrastive_temperature": spatial_contrastive_temperature,
        "spatial_contrastive_neg_strategy": spatial_contrastive_neg_strategy,
    }


def init_model(adata: AnnData, model_config: dict[str, object]) -> DecAlignSpatialMetaLinear:
    return DecAlignSpatialMetaLinear(
        adata,
        proj_dim=int(model_config["proj_dim"]),
        token_dim=int(model_config["token_dim"]),
        n_latent=int(model_config["n_latent"]),
        num_prototypes=int(model_config["num_prototypes"]),
        dropout_rate=float(model_config["dropout_rate"]),
        device=str(model_config["device"]),
        reconstruction_method_st=str(model_config["reconstruction_method_st"]),
        reconstruction_method_sm=str(model_config["reconstruction_method_sm"]),
        standardize_inputs=bool(model_config["standardize_inputs"]),
        use_standardized_reconstruction=bool(model_config["use_standardized_reconstruction"]),
        spatial_hidden_dim=int(model_config["spatial_hidden_dim"]),
        spatial_context_hidden_dim=int(model_config["spatial_context_hidden_dim"]),
        spatial_context_k=int(model_config["spatial_context_k"]),
        spatial_encoder_mode=str(model_config["spatial_encoder_mode"]),
        spatial_token_scale=float(model_config["spatial_token_scale"]),
        spatial_token_dropout=float(model_config["spatial_token_dropout"]),
        spatial_contrastive_pos_k=int(model_config["spatial_contrastive_pos_k"]),
        spatial_contrastive_neg_k=int(model_config["spatial_contrastive_neg_k"]),
        spatial_contrastive_temperature=float(model_config["spatial_contrastive_temperature"]),
        spatial_contrastive_neg_strategy=str(model_config["spatial_contrastive_neg_strategy"]),
    )


def filtered_transfer_state_dict(
    state_dict: dict[str, torch.Tensor],
    target_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in state_dict.items()
        if not any(key.startswith(prefix) for prefix in excluded_transfer_keys)
        and key in target_state_dict
        and value.shape == target_state_dict[key].shape
    }


def get_spatial_coords(adata: AnnData) -> np.ndarray:
    if "spatial" in adata.obsm:
        return np.asarray(adata.obsm["spatial"], dtype=np.float32)
    if {"x_coord", "y_coord"}.issubset(adata.obs.columns):
        return adata.obs[["x_coord", "y_coord"]].to_numpy(dtype=np.float32)
    raise ValueError("adata needs obsm['spatial'] or obs[['x_coord', 'y_coord']]")


def split_adata_spatial_top_third(adata: AnnData) -> tuple[AnnData, AnnData, np.ndarray, np.ndarray, dict[str, object]]:
    coords = get_spatial_coords(adata)
    y_coords = coords[:, 1].astype(np.float32, copy=False)
    all_idx = np.arange(adata.n_obs, dtype=int)

    y_min = float(np.min(y_coords))
    y_max = float(np.max(y_coords))
    y_range = y_max - y_min
    threshold = y_min + y_range / 3.0
    eval_mask = y_coords <= threshold

    fallback = None
    if eval_mask.sum() <= 0 or eval_mask.sum() >= adata.n_obs:
        order = np.argsort(y_coords, kind="mergesort")
        n_eval = max(1, min(adata.n_obs - 1, int(np.ceil(adata.n_obs / 3.0))))
        eval_idx = np.sort(order[:n_eval])
        fallback = "count_based_top_third_by_smallest_y"
    else:
        eval_idx = np.flatnonzero(eval_mask)

    train_idx = np.setdiff1d(all_idx, eval_idx, assume_unique=True)
    split_summary = {
        "split_strategy": "spatial_top_third_on_slice",
        "spatial_axis": "y",
        "spatial_side": "min",
        "visual_region": "top",
        "y_min": y_min,
        "y_max": y_max,
        "y_range": y_range,
        "y_threshold": float(threshold),
        "eval_fraction_actual": float(eval_idx.size / adata.n_obs),
        "train_fraction_actual": float(train_idx.size / adata.n_obs),
        "eval_spot_count": int(eval_idx.size),
        "train_spot_count": int(train_idx.size),
        "fallback": fallback,
    }
    return adata[train_idx].copy(), adata[eval_idx].copy(), train_idx, eval_idx, split_summary


def mean_train_st_library_size(adata: AnnData) -> float:
    st_mask = adata.var["type"].astype(str).eq("ST").to_numpy()
    X_st = adata.X[:, st_mask]
    if sparse.issparse(X_st):
        lib = np.asarray(X_st.sum(axis=1)).reshape(-1)
    else:
        lib = np.asarray(X_st, dtype=np.float32).sum(axis=1)
    return float(np.mean(lib))


@torch.no_grad()
def generate_st_from_sm_fixed_libsize(
    model: DecAlignSpatialMetaLinear,
    adata: AnnData,
    batch_size: int,
    fixed_st_libsize: float,
) -> dict[str, np.ndarray]:
    X_all = adata.X.toarray().astype(np.float32, copy=False) if sparse.issparse(adata.X) else np.asarray(adata.X, dtype=np.float32)
    st_mask = np.asarray(model.st_mask, dtype=bool)
    sm_mask = np.asarray(model.sm_mask, dtype=bool)
    actual_st = X_all[:, st_mask].copy()
    masked_X = X_all.copy()
    masked_X[:, st_mask] = 0.0

    preds_raw = []
    preds_log = []
    truths_log = []

    model.eval()
    for start in range(0, adata.n_obs, batch_size):
        end = min(start + batch_size, adata.n_obs)
        batch_idx = np.arange(start, end)
        X_batch = torch.tensor(masked_X[batch_idx], dtype=torch.float32, device=model.device)
        latent = model.encode_with_indices(X_batch, indices=batch_idx)
        lib_size = torch.full((end - start,), float(fixed_st_libsize), dtype=torch.float32, device=model.device)
        output_dict = model.decode(latent, lib_size)
        pred_raw = output_dict["px_rna_scale"].detach().cpu().numpy()
        pred_log = model._transform_st_features(output_dict["px_rna_scale"]).detach().cpu().numpy()
        truth_log = model._transform_st_features(
            torch.tensor(actual_st[batch_idx], dtype=torch.float32, device=model.device)
        ).detach().cpu().numpy()
        preds_raw.append(pred_raw)
        preds_log.append(pred_log)
        truths_log.append(truth_log)

    return {
        "pred_st_raw": np.vstack(preds_raw),
        "pred_st_log1p": np.vstack(preds_log),
        "true_st_raw": actual_st,
        "true_st_log1p": np.vstack(truths_log),
        "sm_input_raw": X_all[:, sm_mask].copy(),
    }


def average_feature_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    values = []
    for idx in range(y_true.shape[1]):
        a = y_true[:, idx]
        b = y_pred[:, idx]
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            continue
        corr = np.corrcoef(a, b)[0, 1]
        if np.isfinite(corr):
            values.append(float(corr))
    return float(np.mean(values)) if values else float("nan")


def average_spot_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    values = []
    for idx in range(y_true.shape[0]):
        a = y_true[idx]
        b = y_pred[idx]
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            continue
        corr = np.corrcoef(a, b)[0, 1]
        if np.isfinite(corr):
            values.append(float(corr))
    return float(np.mean(values)) if values else float("nan")


def evaluate_generation(result: dict[str, np.ndarray], feature_names: np.ndarray) -> tuple[pd.DataFrame, dict[str, object]]:
    y_true = result["true_st_log1p"]
    y_pred = result["pred_st_log1p"]

    overall = {
        "rmse_log1p": float(np.sqrt(mean_squared_error(y_true.ravel(), y_pred.ravel()))),
        "mse_log1p": float(mean_squared_error(y_true.ravel(), y_pred.ravel())),
        "r2_log1p": float(r2_score(y_true, y_pred, multioutput="variance_weighted")),
        "feature_corr_mean": average_feature_correlation(y_true, y_pred),
        "spot_corr_mean": average_spot_correlation(y_true, y_pred),
        "n_eval_spots": int(y_true.shape[0]),
        "n_st_features": int(y_true.shape[1]),
    }

    feature_rows = []
    for idx, name in enumerate(feature_names):
        a = y_true[:, idx]
        b = y_pred[:, idx]
        corr = np.nan
        if np.std(a) >= 1e-8 and np.std(b) >= 1e-8:
            corr = float(np.corrcoef(a, b)[0, 1])
        feature_rows.append(
            {
                "feature": str(name),
                "rmse_log1p": float(np.sqrt(mean_squared_error(a, b))),
                "mse_log1p": float(mean_squared_error(a, b)),
                "corr": corr,
                "true_mean_log1p": float(np.mean(a)),
                "pred_mean_log1p": float(np.mean(b)),
            }
        )
    feature_df = pd.DataFrame(feature_rows).sort_values(["corr", "true_mean_log1p"], ascending=[False, False]).reset_index(drop=True)
    overall["top_features"] = feature_df.head(20).to_dict(orient="records")
    return feature_df, overall


def save_generation_h5ad(eval_adata: AnnData, result: dict[str, np.ndarray], output_path: Path) -> None:
    st_mask = eval_adata.var["type"].astype(str).eq("ST").to_numpy()
    out = eval_adata[:, st_mask].copy()
    out.X = result["true_st_raw"].copy()
    out.layers["generated_st_raw_sm_only"] = result["pred_st_raw"]
    out.layers["generated_st_log1p_sm_only"] = result["pred_st_log1p"]
    out.layers["true_st_raw_for_eval"] = result["true_st_raw"]
    out.layers["true_st_log1p_for_eval"] = result["true_st_log1p"]
    out.write_h5ad(output_path)


def run_sm_to_st_generation_spatial_top_third() -> dict[str, object]:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(split_seed)
    log_stage(f"training top-third model for {sample_name}")

    base_adata = load_joint_adata(train_input_h5ad)
    train_data, eval_data, train_idx, eval_idx, split_summary = split_adata_spatial_top_third(base_adata)
    train_data = prepare_model_adata(train_data)
    eval_data = prepare_model_adata(eval_data)
    fixed_st_libsize = mean_train_st_library_size(train_data)
    log_stage(f"mean train ST library size={fixed_st_libsize:.6f}")

    model_config = build_model_config()
    model = init_model(train_data, model_config)
    checkpoint_path = experiment_dir / "model_checkpoint_full.pth"
    transfer_checkpoint_path = experiment_dir / "model_checkpoint_transfer_filtered.pth"
    if checkpoint_path.exists():
        log_stage(f"reusing completed training checkpoint {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        history = {}
    else:
        history = model.fit(
            max_epoch=max_epoch,
            n_per_batch=n_per_batch,
            reconstruction_reduction=reconstruction_reduction,
            reconstruction_st_weight=reconstruction_st_weight,
            reconstruction_sm_weight=reconstruction_sm_weight,
            dec_weight=dec_weight,
            homo_weight=homo_weight,
            homo_warmup_epochs=homo_warmup_epochs,
            kl_weight=kl_weight,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
            lr=lr,
            weight_decay=weight_decay,
            random_seed=split_seed,
            balance_start_epoch=16,
            balance_ema=0.8,
            task_weight_floor=0.05,
            spatial_consistency_weight=spatial_consistency_weight,
            spatial_consistency_warmup_epochs=spatial_consistency_warmup_epochs,
            spatial_contrastive_weight=spatial_contrastive_weight,
            spatial_contrastive_warmup_epochs=spatial_contrastive_warmup_epochs,
        )
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": model_config,
                "args": model_config,
                "train_input_h5ad": str(train_input_h5ad),
                "eval_input_h5ad": str(train_input_h5ad),
                "split_strategy": split_summary,
            },
            checkpoint_path,
        )
    eval_model = init_model(eval_data, model_config)
    filtered_state = filtered_transfer_state_dict(model.state_dict(), eval_model.state_dict())
    torch.save({"model_state_dict": filtered_state, "excluded_prefixes": excluded_transfer_keys}, transfer_checkpoint_path)
    missing, unexpected = eval_model.load_state_dict(filtered_state, strict=False)
    generation = generate_st_from_sm_fixed_libsize(
        eval_model,
        eval_data,
        batch_size=n_per_batch,
        fixed_st_libsize=fixed_st_libsize,
    )
    st_feature_names = feature_key_series(eval_data)[eval_data.var["type"].astype(str).eq("ST").to_numpy()].to_numpy()
    feature_df, overall = evaluate_generation(generation, st_feature_names)

    history_path = experiment_dir / "train_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    feature_metrics_path = experiment_dir / "st_feature_metrics.csv"
    feature_df.to_csv(feature_metrics_path, index=False)
    split_path = experiment_dir / "split_metadata.json"
    split_payload = {
        "train_idx": train_idx.tolist(),
        "eval_idx": eval_idx.tolist(),
        "train_obs_names": train_data.obs_names.astype(str).tolist(),
        "eval_obs_names": eval_data.obs_names.astype(str).tolist(),
        "spatial_split": split_summary,
        "split_seed_recorded_only": int(split_seed),
    }
    split_path.write_text(json.dumps(split_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    generated_h5ad_path = experiment_dir / "eval_generated_st_from_sm_only.h5ad"
    save_generation_h5ad(eval_data, generation, generated_h5ad_path)

    summary = {
        "train_input_h5ad": str(train_input_h5ad),
        "eval_input_h5ad": str(train_input_h5ad),
        "same_dataset_split": True,
        "split_train_frac_arg": 0.8,
        "split_seed": int(split_seed),
        "split_strategy": split_summary,
        "inference_st_library_size_mode": "fixed_train_mean",
        "inference_st_library_size_value": float(fixed_st_libsize),
        "feature_alignment": None,
        "device": device,
        "checkpoint_full": str(checkpoint_path),
        "checkpoint_transfer_filtered": str(transfer_checkpoint_path),
        "history_csv": str(history_path),
        "feature_metrics_csv": str(feature_metrics_path),
        "split_metadata_json": str(split_path),
        "generated_eval_h5ad": str(generated_h5ad_path),
        "missing_keys_when_loading_eval": missing,
        "unexpected_keys_when_loading_eval": unexpected,
        "excluded_transfer_prefixes": excluded_transfer_keys,
        "metrics": overall,
    }
    summary_path = experiment_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def build_highres_joint_adata(train_adata: AnnData, raw_sm_adata: AnnData) -> tuple[AnnData, dict[str, object]]:
    train_feature_keys = feature_key_series(train_adata)
    train_types = train_adata.var["type"].astype(str).to_numpy()

    raw_feature_keys = feature_key_series(raw_sm_adata)
    raw_key_to_idx = {str(key): idx for idx, key in enumerate(raw_feature_keys.tolist())}

    X_raw_sm = raw_sm_adata.X.toarray().astype(np.float32, copy=False) if sparse.issparse(raw_sm_adata.X) else np.asarray(raw_sm_adata.X, dtype=np.float32)
    X_joint = np.zeros((raw_sm_adata.n_obs, train_adata.n_vars), dtype=np.float32)

    common_sm = 0
    missing_sm = []
    for col_idx, (feature_key, feature_type) in enumerate(zip(train_feature_keys.tolist(), train_types)):
        if str(feature_type) != "SM":
            continue
        raw_idx = raw_key_to_idx.get(str(feature_key))
        if raw_idx is None:
            missing_sm.append(str(feature_key))
            continue
        X_joint[:, col_idx] = X_raw_sm[:, raw_idx]
        common_sm += 1

    obs = raw_sm_adata.obs.copy()
    if "spot_name" not in obs.columns:
        obs["spot_name"] = raw_sm_adata.obs_names.astype(str)
    if "x_coord" not in obs.columns or "y_coord" not in obs.columns:
        spatial = np.asarray(raw_sm_adata.obsm["spatial"], dtype=np.float32)
        obs["x_coord"] = spatial[:, 0]
        obs["y_coord"] = spatial[:, 1]

    eval_adata = AnnData(X=X_joint, obs=obs, var=train_adata.var.copy())
    eval_adata.obs_names = raw_sm_adata.obs_names.astype(str)
    eval_adata.var_names = train_adata.var_names.astype(str)
    eval_adata.obsm["spatial"] = np.asarray(raw_sm_adata.obsm["spatial"], dtype=np.float32)
    eval_adata.layers["counts"] = sparse.csr_matrix(X_joint)
    eval_adata = prepare_model_adata(eval_adata)

    summary = {
        "raw_sm_obs": int(raw_sm_adata.n_obs),
        "raw_sm_var": int(raw_sm_adata.n_vars),
        "train_joint_var": int(train_adata.n_vars),
        "common_sm_feature_count": int(common_sm),
        "missing_sm_feature_count": int(len(missing_sm)),
        "missing_sm_feature_examples": missing_sm[:20],
    }
    return eval_adata, summary


def aggregate_highres_to_target(
    pred_highres_raw: np.ndarray,
    highres_coords: np.ndarray,
    target_coords: np.ndarray,
    knn_value: int,
) -> tuple[np.ndarray, dict[str, object]]:
    k = max(1, min(int(knn_value), highres_coords.shape[0]))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(highres_coords)
    dists, idx = nn.kneighbors(target_coords)
    weights = 1.0 / np.maximum(dists, 1e-8)
    weights = weights / weights.sum(axis=1, keepdims=True)
    pred_target = np.sum(pred_highres_raw[idx] * weights[:, :, None], axis=1)
    return pred_target, {
        "aggregation_knn_k": k,
        "neighbor_distance_mean": float(np.mean(dists)),
        "neighbor_distance_median": float(np.median(dists)),
        "neighbor_distance_max": float(np.max(dists)),
    }


def evaluate_predictions(
    y_true_raw: np.ndarray,
    y_pred_raw: np.ndarray,
    feature_names: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, object], np.ndarray, np.ndarray]:
    y_true_log = np.log1p(np.clip(y_true_raw, a_min=0.0, a_max=None))
    y_pred_log = np.log1p(np.clip(y_pred_raw, a_min=0.0, a_max=None))

    overall = {
        "rmse_log1p": float(np.sqrt(mean_squared_error(y_true_log.ravel(), y_pred_log.ravel()))),
        "mse_log1p": float(mean_squared_error(y_true_log.ravel(), y_pred_log.ravel())),
        "r2_log1p": float(r2_score(y_true_log, y_pred_log, multioutput="variance_weighted")),
        "feature_corr_mean": average_feature_correlation(y_true_log, y_pred_log),
        "spot_corr_mean": average_spot_correlation(y_true_log, y_pred_log),
        "n_eval_spots": int(y_true_log.shape[0]),
        "n_st_features": int(y_true_log.shape[1]),
    }

    feature_rows = []
    for idx, name in enumerate(feature_names):
        a = y_true_log[:, idx]
        b = y_pred_log[:, idx]
        corr = np.nan
        if np.std(a) >= 1e-8 and np.std(b) >= 1e-8:
            corr = float(np.corrcoef(a, b)[0, 1])
        feature_rows.append(
            {
                "feature": str(name),
                "rmse_log1p": float(np.sqrt(mean_squared_error(a, b))),
                "mse_log1p": float(mean_squared_error(a, b)),
                "corr": corr,
                "true_mean_log1p": float(np.mean(a)),
                "pred_mean_log1p": float(np.mean(b)),
            }
        )
    feature_df = pd.DataFrame(feature_rows).sort_values(["corr", "true_mean_log1p"], ascending=[False, False]).reset_index(drop=True)
    overall["top_features"] = feature_df.head(20).to_dict(orient="records")
    return feature_df, overall, y_true_log, y_pred_log


def save_target_h5ad(
    target_st_adata: AnnData,
    pred_raw: np.ndarray,
    pred_log: np.ndarray,
    true_raw: np.ndarray,
    true_log: np.ndarray,
    output_path: Path,
) -> None:
    out = target_st_adata.copy()
    out.X = true_raw.copy()
    out.layers["generated_st_raw_from_sm_superres"] = pred_raw
    out.layers["generated_st_log1p_from_sm_superres"] = pred_log
    out.layers["true_st_raw_for_eval"] = true_raw
    out.layers["true_st_log1p_for_eval"] = true_log
    out.write_h5ad(output_path)


def save_celllevel_h5ad(
    raw_sm_adata: AnnData,
    st_feature_names: np.ndarray,
    pred_raw: np.ndarray,
    pred_log: np.ndarray,
    output_path: Path,
) -> None:
    out = AnnData(
        X=np.zeros((raw_sm_adata.n_obs, len(st_feature_names)), dtype=np.float32),
        obs=raw_sm_adata.obs.copy(),
        var=pd.DataFrame(index=st_feature_names.astype(str)),
    )
    out.obs_names = raw_sm_adata.obs_names.astype(str)
    if "spatial" in raw_sm_adata.obsm:
        out.obsm["spatial"] = np.asarray(raw_sm_adata.obsm["spatial"], dtype=np.float32)
    out.layers["generated_st_raw_from_sm_celllevel"] = pred_raw
    out.layers["generated_st_log1p_from_sm_celllevel"] = pred_log
    out.write_h5ad(output_path)


def evaluate_sm_to_st_superres_fixed_train_mean_libsize(experiment_summary: dict[str, object]) -> dict[str, object]:
    superres_dir.mkdir(parents=True, exist_ok=True)
    train_adata = load_joint_adata(Path(experiment_summary["train_input_h5ad"]))
    target_adata = load_joint_adata(train_input_h5ad)
    split_payload = json.loads(Path(experiment_summary["split_metadata_json"]).read_text(encoding="utf-8"))
    eval_obs_names = split_payload.get("eval_obs_names")
    if target_scope == "eval_only" and eval_obs_names:
        target_adata = target_adata[eval_obs_names].copy()

    checkpoint_payload = torch.load(experiment_summary["checkpoint_transfer_filtered"], map_location="cpu")
    checkpoint_full = torch.load(experiment_summary["checkpoint_full"], map_location="cpu")
    model_config = checkpoint_full.get("model_config", checkpoint_full.get("args"))

    raw_sm_adata = sc.read_h5ad(highres_sm_h5ad)
    raw_sm_adata.obs_names = raw_sm_adata.obs_names.astype(str)
    highres_joint_adata, highres_summary = build_highres_joint_adata(train_adata, raw_sm_adata)

    fixed_st_libsize = mean_train_st_library_size(train_adata)
    eval_model = init_model(highres_joint_adata, model_config)
    missing, unexpected = eval_model.load_state_dict(checkpoint_payload["model_state_dict"], strict=False)
    generation = generate_st_from_sm_fixed_libsize(
        eval_model,
        highres_joint_adata,
        batch_size=int(model_config.get("n_per_batch", n_per_batch)),
        fixed_st_libsize=fixed_st_libsize,
    )

    st_mask = target_adata.var["type"].astype(str).eq("ST").to_numpy()
    st_feature_names = feature_key_series(target_adata)[st_mask].to_numpy()

    celllevel_h5ad_path = superres_dir / "celllevel_generated_st_from_sm.h5ad"
    save_celllevel_h5ad(
        raw_sm_adata,
        st_feature_names,
        generation["pred_st_raw"],
        generation["pred_st_log1p"],
        celllevel_h5ad_path,
    )

    true_st_raw = target_adata.X[:, st_mask].toarray().astype(np.float32, copy=False) if sparse.issparse(target_adata.X) else np.asarray(target_adata.X[:, st_mask], dtype=np.float32)
    pred_target_raw, agg_summary = aggregate_highres_to_target(
        generation["pred_st_raw"],
        np.asarray(highres_joint_adata.obsm["spatial"], dtype=np.float32),
        np.asarray(target_adata.obsm["spatial"], dtype=np.float32),
        knn_k,
    )
    feature_df, metrics, true_log, pred_log = evaluate_predictions(true_st_raw, pred_target_raw, st_feature_names)

    feature_metrics_path = superres_dir / "st_feature_metrics_superres_aggregated.csv"
    feature_df.to_csv(feature_metrics_path, index=False)
    target_st_adata = target_adata[:, st_mask].copy()
    eval_h5ad_path = superres_dir / "aggregated_superres_st_eval.h5ad"
    save_target_h5ad(target_st_adata, pred_target_raw, pred_log, true_st_raw, true_log, eval_h5ad_path)

    summary = {
        "experiment_summary": str((experiment_dir / "summary.json").resolve()),
        "highres_sm_h5ad": str(highres_sm_h5ad.resolve()),
        "target_input_h5ad": str(train_input_h5ad.resolve()),
        "target_scope": target_scope,
        "inference_st_library_size_mode": "fixed_train_mean",
        "inference_st_library_size_value": float(fixed_st_libsize),
        "celllevel_generated_h5ad": str(celllevel_h5ad_path),
        "feature_metrics_csv": str(feature_metrics_path),
        "aggregated_eval_h5ad": str(eval_h5ad_path),
        "total_target_spot_count": int(load_joint_adata(train_input_h5ad).n_obs),
        "aggregated_target_spot_count": int(target_adata.n_obs),
        "split_eval_spot_count": int(len(eval_obs_names)) if eval_obs_names else None,
        "highres_joint_summary": highres_summary,
        "aggregation_summary": agg_summary,
        "missing_keys_when_loading_eval": missing,
        "unexpected_keys_when_loading_eval": unexpected,
        "metrics": metrics,
    }
    summary_path = superres_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def region_mask(coords: np.ndarray, axis: str, side: str, threshold: float) -> np.ndarray:
    axis_idx = 0 if axis == "x" else 1
    values = coords[:, axis_idx]
    if side in {"top", "left"}:
        return values <= threshold
    return values >= threshold


def compute_region_threshold(coords: np.ndarray, axis: str, side: str, fraction: float) -> float:
    axis_idx = 0 if axis == "x" else 1
    values = coords[:, axis_idx].astype(np.float32, copy=False)
    value_min = float(np.min(values))
    value_max = float(np.max(values))
    value_range = value_max - value_min
    if side in {"top", "left"}:
        return value_min + value_range * float(fraction)
    return value_max - value_range * float(fraction)


def plot_layer_gene_on_spatial_region(input_h5ad: Path) -> dict[str, object]:
    figure_dir.mkdir(parents=True, exist_ok=True)
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
    threshold = compute_region_threshold(coords, axis=region_axis, side=region_side, fraction=region_fraction)
    mask = region_mask(coords, axis=region_axis, side=region_side, threshold=threshold)
    region_values = values[mask]
    region_coords = coords[mask]

    plot_vmin = float(np.nanmin(region_values))
    plot_vmax = float(np.nanquantile(region_values, clip_quantile))
    plot_vmax = max(plot_vmax, plot_vmin + 1e-8)

    fig, ax = plt.subplots(figsize=(8.3, 6.8))
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
    ax.set_title(title, fontsize=title_fontsize)
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Intensity", rotation=90, fontsize=colorbar_label_fontsize)
    colorbar.ax.tick_params(labelsize=colorbar_tick_fontsize)
    fig.tight_layout()

    region_tag = f"{region_side}_{region_axis}_{str(threshold).replace('.', 'p')}"
    stem = f"{gene}_{layer}_{region_tag}"
    png_path = figure_dir / f"{stem}.png"
    pdf_path = figure_dir / f"{stem}.pdf"
    svg_path = figure_dir / f"{stem}.svg"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight", format="svg")
    plt.close(fig)

    summary = {
        "input_h5ad": str(input_h5ad.resolve()),
        "layer": layer,
        "gene": gene,
        "region_axis": region_axis,
        "region_side": region_side,
        "region_fraction": float(region_fraction),
        "threshold": float(threshold),
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
        "svg": str(svg_path),
    }
    (figure_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def generation_summary_matches(path: Path) -> bool:
    if not path.exists():
        return False
    summary = load_json(path)
    return str(summary.get("train_input_h5ad")) == str(train_input_h5ad)


def superres_summary_matches(path: Path) -> bool:
    if not path.exists():
        return False
    summary = load_json(path)
    if str(summary.get("highres_sm_h5ad")) != str(highres_sm_h5ad):
        return False
    highres_summary = summary.get("highres_joint_summary", {})
    raw_sm_obs = int(highres_summary.get("raw_sm_obs", 0))
    total_target_spot_count = int(summary.get("total_target_spot_count", 0))
    return raw_sm_obs > total_target_spot_count


def run_generation_if_needed() -> dict[str, object]:
    summary_path = experiment_dir / "summary.json"
    if generation_summary_matches(summary_path):
        return load_json(summary_path)
    return run_sm_to_st_generation_spatial_top_third()


def run_superres_if_needed(experiment_summary: dict[str, object]) -> dict[str, object]:
    summary_path = superres_dir / "summary.json"
    if superres_summary_matches(summary_path):
        return load_json(summary_path)
    return evaluate_sm_to_st_superres_fixed_train_mean_libsize(experiment_summary)


def run_fig2i() -> dict[str, object]:
    experiment_summary = run_generation_if_needed()
    superres_summary = run_superres_if_needed(experiment_summary)
    plot_summary = plot_layer_gene_on_spatial_region(Path(superres_summary["celllevel_generated_h5ad"]))

    summary = {
        "sample_name": sample_name,
        "gene": gene,
        "train_input_h5ad": str(train_input_h5ad),
        "highres_sm_h5ad": str(highres_sm_h5ad),
        "experiment_summary_json": str((experiment_dir / "summary.json").resolve()),
        "superres_summary_json": str((superres_dir / "summary.json").resolve()),
        "celllevel_generated_h5ad": str(Path(superres_summary["celllevel_generated_h5ad"]).resolve()),
        "plot_summary": plot_summary,
    }
    summary_path = figure_dir / f"{sample_name}_{gene}_fig2i_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    run_fig2i()
