from __future__ import annotations

import json
import random
import sys
from pathlib import Path

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

package_root = Path(__file__).resolve().parents[2]
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))

from SpaDTA_718.model.model import DecAlignSpatialMetaLinear
from SpaDTA_718.model.preprocess import normalize_total_joint_adata_sm_st, prepare_spadta_model_input


project_root = Path("/data/user/hesy/projects/SpatialMETA")
processed_root = Path("/bigdat2/user/hesy/spatialmeta/SpatialMETA/processed")

sample_name = "X49_T"
train_input_h5ad = processed_root / f"{sample_name}.h5ad"

root_dir = project_root / "SpaDTA_718" / "runs" / "sm_downstream" / "fig2j"
experiment_dir = root_dir / "st_to_sm_generation_spatial_top_third"
figure_dir = root_dir / "figures"
table_dir = root_dir / "tables"

device = "cuda"
split_seed = 42
max_epoch = 128
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

background_point_size = 16.0
foreground_point_size = 36.0
dpi = 260
background_color = "#b8b8b8"
background_alpha = 0.78
foreground_edge_color = "white"
foreground_edge_width = 0.25
cmap = "viridis"
vmin_pcc = -0.2
vmax_pcc = 0.8

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
    print(f"[fig2j] {message}", flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def to_dense_float32(values) -> np.ndarray:
    if sparse.issparse(values):
        return values.toarray().astype(np.float32, copy=False)
    return np.asarray(values, dtype=np.float32)


def load_joint_adata(path: Path) -> AnnData:
    adata = sc.read_h5ad(path)
    adata.obs_names = adata.obs_names.astype(str)
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
        normalize_total_joint_adata_sm_st(
            adata,
            target_sum_SM=1e3,
            target_sum_ST=None,
        )
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


def get_spatial_coords(adata: AnnData) -> np.ndarray:
    if "spatial" in adata.obsm:
        return np.asarray(adata.obsm["spatial"], dtype=np.float32)
    if {"x_coord", "y_coord"}.issubset(adata.obs.columns):
        return adata.obs[["x_coord", "y_coord"]].to_numpy(dtype=np.float32)
    raise ValueError("adata needs obsm['spatial'] or obs[['x_coord', 'y_coord']].")


def split_adata_spatial_top_third(adata: AnnData) -> tuple[AnnData, AnnData, np.ndarray, np.ndarray, dict[str, object]]:
    log_stage("computing spatial top-third split")
    coords = get_spatial_coords(adata)
    y_coords = coords[:, 1].astype(np.float32, copy=False)
    all_idx = np.arange(adata.n_obs, dtype=int)

    y_min = float(np.min(y_coords))
    y_max = float(np.max(y_coords))
    y_range = y_max - y_min
    threshold = y_min + (y_range / 3.0)
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
    if train_idx.size == 0:
        raise ValueError("spatial top-third split produced an empty training set.")

    split_meta = {
        "split_strategy": "spatial_top_third_on_slice",
        "spatial_axis": "y",
        "spatial_side": "min",
        "visual_region": "top",
        "coordinate_system_note": "Assumes image-like spatial coordinates; smaller raw y is visually higher after invert_yaxis().",
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
    return adata[train_idx].copy(), adata[eval_idx].copy(), train_idx, eval_idx, split_meta


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


def average_feature_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    values: list[float] = []
    for idx in range(y_true.shape[1]):
        current_true = y_true[:, idx]
        current_pred = y_pred[:, idx]
        if np.std(current_true) < 1e-8 or np.std(current_pred) < 1e-8:
            continue
        corr = np.corrcoef(current_true, current_pred)[0, 1]
        if np.isfinite(corr):
            values.append(float(corr))
    return float(np.mean(values)) if values else float("nan")


def average_spot_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    values: list[float] = []
    for idx in range(y_true.shape[0]):
        current_true = y_true[idx]
        current_pred = y_pred[idx]
        if np.std(current_true) < 1e-8 or np.std(current_pred) < 1e-8:
            continue
        corr = np.corrcoef(current_true, current_pred)[0, 1]
        if np.isfinite(corr):
            values.append(float(corr))
    return float(np.mean(values)) if values else float("nan")


@torch.no_grad()
def generate_sm_from_st(model: DecAlignSpatialMetaLinear, adata: AnnData, batch_size: int) -> dict[str, np.ndarray]:
    X_all = to_dense_float32(adata.X)
    st_mask = np.asarray(model.st_mask, dtype=bool)
    sm_mask = np.asarray(model.sm_mask, dtype=bool)
    actual_sm = X_all[:, sm_mask].copy()
    masked_X = X_all.copy()
    masked_X[:, sm_mask] = 0.0

    preds_raw: list[np.ndarray] = []
    preds_log: list[np.ndarray] = []
    truths_log: list[np.ndarray] = []

    model.eval()
    for start in range(0, adata.n_obs, batch_size):
        end = min(start + batch_size, adata.n_obs)
        batch_idx = np.arange(start, end)
        X_batch = torch.tensor(masked_X[batch_idx], dtype=torch.float32, device=model.device)
        _, output_dict, _ = model.forward_with_indices(X_batch, indices=batch_idx, reduction="sum")
        pred_raw = output_dict["px_sm_scale"].detach().cpu().numpy()
        pred_log = model._transform_sm_prediction(output_dict["px_sm_scale"]).detach().cpu().numpy()
        truth_log = model._transform_sm_features(
            torch.tensor(actual_sm[batch_idx], dtype=torch.float32, device=model.device)
        ).detach().cpu().numpy()
        preds_raw.append(pred_raw)
        preds_log.append(pred_log)
        truths_log.append(truth_log)

    return {
        "pred_sm_raw": np.vstack(preds_raw),
        "pred_sm_log1p": np.vstack(preds_log),
        "true_sm_raw": actual_sm,
        "true_sm_log1p": np.vstack(truths_log),
        "st_input_raw": X_all[:, st_mask].copy(),
    }


def evaluate_generation(result: dict[str, np.ndarray], feature_names: np.ndarray) -> tuple[pd.DataFrame, dict[str, object]]:
    y_true = result["true_sm_log1p"]
    y_pred = result["pred_sm_log1p"]

    overall = {
        "rmse_log1p": float(np.sqrt(mean_squared_error(y_true.ravel(), y_pred.ravel()))),
        "mse_log1p": float(mean_squared_error(y_true.ravel(), y_pred.ravel())),
        "r2_log1p": float(r2_score(y_true, y_pred, multioutput="variance_weighted")),
        "feature_corr_mean": average_feature_correlation(y_true, y_pred),
        "spot_corr_mean": average_spot_correlation(y_true, y_pred),
        "n_eval_spots": int(y_true.shape[0]),
        "n_sm_features": int(y_true.shape[1]),
    }

    rows: list[dict[str, object]] = []
    for idx, name in enumerate(feature_names):
        current_true = y_true[:, idx]
        current_pred = y_pred[:, idx]
        corr = np.nan
        if np.std(current_true) >= 1e-8 and np.std(current_pred) >= 1e-8:
            corr = float(np.corrcoef(current_true, current_pred)[0, 1])
        rows.append(
            {
                "feature": str(name),
                "rmse_log1p": float(np.sqrt(mean_squared_error(current_true, current_pred))),
                "mse_log1p": float(mean_squared_error(current_true, current_pred)),
                "corr": corr,
                "true_mean_log1p": float(np.mean(current_true)),
                "pred_mean_log1p": float(np.mean(current_pred)),
            }
        )

    feature_df = pd.DataFrame(rows).sort_values(["corr", "true_mean_log1p"], ascending=[False, False]).reset_index(drop=True)
    overall["top_features"] = feature_df.head(20).to_dict(orient="records")
    return feature_df, overall


def save_generation_h5ad(eval_adata: AnnData, result: dict[str, np.ndarray], output_path: Path) -> None:
    sm_mask = eval_adata.var["type"].astype(str).eq("SM").to_numpy()
    out = eval_adata[:, sm_mask].copy()
    out.X = result["true_sm_raw"].copy()
    out.layers["generated_sm_raw_st_only"] = result["pred_sm_raw"]
    out.layers["generated_sm_log1p_st_only"] = result["pred_sm_log1p"]
    out.layers["true_sm_raw_for_eval"] = result["true_sm_raw"]
    out.layers["true_sm_log1p_for_eval"] = result["true_sm_log1p"]
    out.write_h5ad(output_path)


def per_spot_pcc(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
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


def plot_spot_pcc_on_top_third(
    full_adata: AnnData,
    eval_h5ad_path: Path,
    split_meta: dict[str, object],
) -> dict[str, object]:
    log_stage("plotting spot-level PCC on held-out top third")
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    eval_adata = sc.read_h5ad(eval_h5ad_path)
    eval_adata.obs_names = eval_adata.obs_names.astype(str)
    full_adata.obs_names = full_adata.obs_names.astype(str)

    if "spatial" not in eval_adata.obsm or "spatial" not in full_adata.obsm:
        raise ValueError("Both full and eval adata need obsm['spatial'].")

    y_true = to_dense_float32(eval_adata.layers["true_sm_log1p_for_eval"])
    y_pred = to_dense_float32(eval_adata.layers["generated_sm_log1p_st_only"])
    spot_pcc = per_spot_pcc(y_true, y_pred)
    eval_coords = np.asarray(eval_adata.obsm["spatial"], dtype=np.float32)
    bg_coords = np.asarray(full_adata.obsm["spatial"], dtype=np.float32)

    pcc_table = pd.DataFrame(
        {
            "obs_name": eval_adata.obs_names.astype(str),
            "x_coord": eval_coords[:, 0],
            "y_coord": eval_coords[:, 1],
            "spot_pcc": spot_pcc,
        }
    )
    pcc_csv = table_dir / f"{sample_name}_top_third_spot_pcc.csv"
    pcc_table.to_csv(pcc_csv, index=False)

    fig, ax = plt.subplots(figsize=(7.4, 7.8))
    ax.scatter(
        bg_coords[:, 0],
        bg_coords[:, 1],
        s=background_point_size,
        c=background_color,
        alpha=background_alpha,
        linewidths=0.0,
        zorder=1,
    )
    scatter = ax.scatter(
        eval_coords[:, 0],
        eval_coords[:, 1],
        c=spot_pcc,
        s=foreground_point_size,
        cmap=cmap,
        vmin=vmin_pcc,
        vmax=vmax_pcc,
        linewidths=foreground_edge_width,
        edgecolors=foreground_edge_color,
        zorder=2,
    )
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("PCC", fontsize=14, pad=10)

    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.025)
    colorbar.ax.tick_params(labelsize=9)

    mean_pcc = float(np.nanmean(spot_pcc))
    median_pcc = float(np.nanmedian(spot_pcc))

    fig.tight_layout()

    figure_base = f"{sample_name}_st_to_sm_spot_pcc_top_third"
    png_path = figure_dir / f"{figure_base}.png"
    pdf_path = figure_dir / f"{figure_base}.pdf"
    svg_path = figure_dir / f"{figure_base}.svg"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "sample_name": sample_name,
        "eval_h5ad": str(eval_h5ad_path.resolve()),
        "spot_pcc_csv": str(pcc_csv.resolve()),
        "figure_png": str(png_path.resolve()),
        "figure_pdf": str(pdf_path.resolve()),
        "figure_svg": str(svg_path.resolve()),
        "n_background_spots": int(full_adata.n_obs),
        "n_eval_spots": int(eval_adata.n_obs),
        "spot_pcc_mean": mean_pcc,
        "spot_pcc_median": median_pcc,
        "spot_pcc_min": float(np.nanmin(spot_pcc)),
        "spot_pcc_max": float(np.nanmax(spot_pcc)),
        "split_meta": split_meta,
    }
    summary_path = figure_dir / f"{sample_name}_fig2j_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def run_st_to_sm_generation_spatial_top_third() -> dict[str, object]:
    seed_everything(split_seed)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    log_stage(f"loading {train_input_h5ad}")
    full_adata = load_joint_adata(train_input_h5ad)
    train_data, eval_data, train_idx, eval_idx, split_meta = split_adata_spatial_top_third(full_adata)
    train_data = prepare_model_adata(train_data)
    eval_data = prepare_model_adata(eval_data)

    split_payload = {
        "train_idx": train_idx.tolist(),
        "eval_idx": eval_idx.tolist(),
        "train_obs_names": train_data.obs_names.astype(str).tolist(),
        "eval_obs_names": eval_data.obs_names.astype(str).tolist(),
        "spatial_split": split_meta,
        "split_seed_recorded_only": int(split_seed),
    }

    model_config = build_model_config()
    log_stage("initializing train model")
    model = init_model(train_data, model_config)
    checkpoint_path = experiment_dir / "model_checkpoint_full.pth"
    transfer_checkpoint_path = experiment_dir / "model_checkpoint_transfer_filtered.pth"
    if checkpoint_path.exists():
        log_stage(f"reusing completed training checkpoint {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        history = {}
    else:
        log_stage("fitting model")
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
            },
            checkpoint_path,
        )

    log_stage("initializing eval model")
    eval_model = init_model(eval_data, model_config)
    filtered_state = filtered_transfer_state_dict(model.state_dict(), eval_model.state_dict())
    torch.save(
        {
            "model_state_dict": filtered_state,
            "excluded_prefixes": excluded_transfer_keys,
        },
        transfer_checkpoint_path,
    )

    missing_keys, unexpected_keys = eval_model.load_state_dict(filtered_state, strict=False)
    log_stage("generating metabolites from held-out ST")
    generation = generate_sm_from_st(eval_model, eval_data, batch_size=n_per_batch)
    sm_feature_names = feature_key_series(eval_data)[eval_data.var["type"].astype(str).eq("SM").to_numpy()].to_numpy()
    feature_df, metrics = evaluate_generation(generation, sm_feature_names)

    history_path = experiment_dir / "train_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    feature_metrics_path = experiment_dir / "sm_feature_metrics.csv"
    feature_df.to_csv(feature_metrics_path, index=False)
    split_path = experiment_dir / "split_metadata.json"
    split_path.write_text(json.dumps(split_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    eval_h5ad_path = experiment_dir / "eval_generated_sm_from_st_only.h5ad"
    save_generation_h5ad(eval_data, generation, eval_h5ad_path)

    figure_summary = plot_spot_pcc_on_top_third(full_adata=full_adata, eval_h5ad_path=eval_h5ad_path, split_meta=split_meta)

    summary = {
        "sample_name": sample_name,
        "train_input_h5ad": str(train_input_h5ad.resolve()),
        "device": device,
        "checkpoint_full": str(checkpoint_path.resolve()),
        "checkpoint_transfer_filtered": str(transfer_checkpoint_path.resolve()),
        "history_csv": str(history_path.resolve()),
        "feature_metrics_csv": str(feature_metrics_path.resolve()),
        "split_metadata_json": str(split_path.resolve()),
        "generated_eval_h5ad": str(eval_h5ad_path.resolve()),
        "missing_keys_when_loading_eval": list(missing_keys),
        "unexpected_keys_when_loading_eval": list(unexpected_keys),
        "excluded_transfer_prefixes": excluded_transfer_keys,
        "metrics": metrics,
        "figure_summary": figure_summary,
    }
    summary_path = root_dir / "summary.json"
    root_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    run_st_to_sm_generation_spatial_top_third()
