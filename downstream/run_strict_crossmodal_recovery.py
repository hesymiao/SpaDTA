from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from SpaDTA_718.downstream import fig2j, fig2k
from SpaDTA_718.model.preprocess import prepare_spadta_model_input
from SpaDTA_718.model.model_generate import DecAlignSpatialMetaLinear
from SpaDTA_718.model.workflow import seed_everything


project_root_path = Path("/data/user/hesy/projects/SpatialMETA")
run_root = project_root_path / "SpaDTA_718" / "runs"
downstream_root = run_root / "sm_downstream"


def load_current_config(sample_name: str, device: str) -> tuple[dict[str, object], Path, Path]:
    config_path = run_root / "SM" / sample_name / "config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    train_kwargs = dict(payload["train_kwargs"])
    for key in ("spatial_fourier_scales", "save_embedding_epochs"):
        value = train_kwargs.get(key)
        if isinstance(value, str):
            train_kwargs[key] = json.loads(value)
    train_kwargs["device"] = device
    return train_kwargs, Path(payload["input_h5ad_path"]), config_path


def fit_with_current_config(model, train_kwargs: dict[str, object]):
    fit_signature = inspect.signature(model.fit)
    fit_kwargs = {
        key: train_kwargs[key]
        for key in fit_signature.parameters
        if key in train_kwargs and key != "epoch_end_callback"
    }
    fit_kwargs["task_weight_floor"] = train_kwargs["balance_weight_floor"]
    return model.fit(**fit_kwargs)


def build_generation_model(adata: sc.AnnData, train_kwargs: dict[str, object], shared_graph_mode: str) -> DecAlignSpatialMetaLinear:
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
        shared_graph_mode=shared_graph_mode,
    )


def mask_eval_encoder_input(eval_data: sc.AnnData, direction: str) -> None:
    encoder_input = eval_data.layers["spadta_encoder_input"].copy()
    target_mask = eval_data.var["type"].astype(str).eq("SM" if direction == "st_to_sm" else "ST").to_numpy()
    if hasattr(encoder_input, "tolil"):
        encoder_input = encoder_input.tolil(copy=True)
        encoder_input[:, target_mask] = 0.0
        eval_data.layers["spadta_encoder_input"] = encoder_input.tocsr()
    else:
        encoder_input[:, target_mask] = 0.0
        eval_data.layers["spadta_encoder_input"] = encoder_input


def generate_source_only(model, adata: sc.AnnData, direction: str, batch_size: int, fixed_st_libsize: float | None = None):
    X_all = np.asarray(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X, dtype=np.float32)
    st_mask = np.asarray(model.st_mask, dtype=bool)
    sm_mask = np.asarray(model.sm_mask, dtype=bool)
    source_mask = st_mask if direction == "st_to_sm" else sm_mask
    target_mask = sm_mask if direction == "st_to_sm" else st_mask
    target_name = "sm" if direction == "st_to_sm" else "st"
    target_raw = X_all[:, target_mask].copy()
    preds_raw = []
    preds_log = []
    truths_log = []
    model.eval()
    for start in range(0, adata.n_obs, batch_size):
        end = min(start + batch_size, adata.n_obs)
        batch_idx = np.arange(start, end)
        X_batch = torch.tensor(X_all[batch_idx], dtype=torch.float32, device=model.device)
        if direction == "st_to_sm":
            lib_size = torch.ones(end - start, dtype=torch.float32, device=model.device)
        else:
            lib_size = torch.full((end - start,), float(fixed_st_libsize), dtype=torch.float32, device=model.device)
        output = model.generate_source_only_with_indices(
            X_batch, batch_idx, "st" if direction == "st_to_sm" else "sm", lib_size
        )
        prediction = output["px_sm_scale" if target_name == "sm" else "px_rna_scale"]
        transform_prediction = model._transform_sm_prediction if target_name == "sm" else model._transform_st_features
        transform_truth = model._transform_sm_features if target_name == "sm" else model._transform_st_features
        preds_raw.append(prediction.cpu().numpy())
        preds_log.append(transform_prediction(prediction).cpu().numpy())
        truths_log.append(transform_truth(torch.tensor(target_raw[batch_idx], dtype=torch.float32, device=model.device)).cpu().numpy())
    result = {
        "pred_" + target_name + "_raw": np.vstack(preds_raw),
        "pred_" + target_name + "_log1p": np.vstack(preds_log),
        "true_" + target_name + "_raw": target_raw,
        "true_" + target_name + "_log1p": np.vstack(truths_log),
        ("st_input_raw" if target_name == "sm" else "sm_input_raw"): X_all[:, source_mask].copy(),
    }
    return result


def prepare_split(input_h5ad: Path, train_kwargs: dict[str, object]):
    full_adata = sc.read_h5ad(input_h5ad)
    full_adata.obs_names = full_adata.obs_names.astype(str)
    train_data, eval_data, train_idx, eval_idx, split_meta = fig2j.split_adata_spatial_top_third(full_adata)
    prepare_kwargs = {
        "modality": "sm",
        "expression_graph_k": int(train_kwargs["spatial_contrastive_pos_k"]),
        "spatial_context_k": int(train_kwargs["spatial_context_k"]),
    }
    train_data = prepare_spadta_model_input(train_data, **prepare_kwargs)
    eval_data = prepare_spadta_model_input(eval_data, **prepare_kwargs)
    split_payload = {
        "train_idx": train_idx.tolist(),
        "eval_idx": eval_idx.tolist(),
        "train_obs_names": train_data.obs_names.astype(str).tolist(),
        "eval_obs_names": eval_data.obs_names.astype(str).tolist(),
        "spatial_split": split_meta,
        "split_seed_recorded_only": int(train_kwargs["random_seed"]),
    }
    return full_adata, train_data, eval_data, split_meta, split_payload


def save_common(
    *,
    direction: str,
    module,
    output_root: Path,
    experiment_dir: Path,
    input_h5ad: Path,
    config_path: Path,
    sample_name: str,
    train_kwargs: dict[str, object],
    full_adata,
    train_data,
    eval_data,
    split_meta: dict[str, object],
    split_payload: dict[str, object],
) -> dict[str, object]:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    model = build_generation_model(
        train_data,
        train_kwargs,
        shared_graph_mode=str(train_kwargs.get("shared_graph_mode", "praga_fused")),
    )
    history = fit_with_current_config(model, train_kwargs)

    mask_eval_encoder_input(eval_data, direction)
    eval_model = build_generation_model(eval_data, train_kwargs, shared_graph_mode="spatial_only")
    filtered_state = module.filtered_transfer_state_dict(model.state_dict(), eval_model.state_dict())
    missing, unexpected = eval_model.load_state_dict(filtered_state, strict=False)

    if direction == "st_to_sm":
        generation = generate_source_only(eval_model, eval_data, direction, int(train_kwargs["n_per_batch"]))
        feature_names = module.feature_key_series(eval_data)[
            eval_data.var["type"].astype(str).eq("SM").to_numpy()
        ].to_numpy()
        feature_df, metrics = module.evaluate_generation(generation, feature_names)
        eval_path = experiment_dir / "eval_generated_sm_from_st_only.h5ad"
        metrics_path = experiment_dir / "sm_feature_metrics.csv"
        module.save_generation_h5ad(eval_data, generation, eval_path)
    else:
        fixed_st_libsize = module.mean_train_st_library_size(train_data)
        generation = generate_source_only(
            eval_model, eval_data, direction, int(train_kwargs["n_per_batch"]), fixed_st_libsize
        )
        feature_names = module.feature_key_series(eval_data)[
            eval_data.var["type"].astype(str).eq("ST").to_numpy()
        ].to_numpy()
        feature_df, metrics = module.evaluate_generation(generation, feature_names)
        metrics["inference_st_library_size_mode"] = "fixed_train_mean"
        metrics["inference_st_library_size_value"] = float(fixed_st_libsize)
        eval_path = experiment_dir / "eval_generated_st_from_sm_only.h5ad"
        metrics_path = experiment_dir / "st_feature_metrics.csv"
        module.save_generation_h5ad(eval_data, generation, eval_path)

    checkpoint_path = experiment_dir / "model_checkpoint_full.pth"
    transfer_path = experiment_dir / "model_checkpoint_transfer_filtered.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "train_kwargs": train_kwargs,
            "source_config": str(config_path),
            "train_input_h5ad": str(input_h5ad),
            "split_strategy": split_meta,
        },
        checkpoint_path,
    )
    torch.save(
        {
            "model_state_dict": filtered_state,
            "excluded_prefixes": module.excluded_transfer_keys,
            "source_config": str(config_path),
        },
        transfer_path,
    )
    pd.DataFrame(history).to_csv(experiment_dir / "train_history.csv", index=False)
    feature_df.to_csv(metrics_path, index=False)
    (experiment_dir / "split_metadata.json").write_text(
        json.dumps(split_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    module.sample_name = sample_name
    module.figure_dir = output_root / "figures"
    module.table_dir = output_root / "tables"
    figure_summary = module.plot_spot_pcc_on_top_third(
        full_adata=full_adata,
        eval_h5ad_path=eval_path,
        split_meta=split_meta,
    )
    summary = {
        "sample_name": sample_name,
        "direction": direction,
        "source_config": str(config_path),
        "train_input_h5ad": str(input_h5ad),
        "effective_train_kwargs": train_kwargs,
        "checkpoint_full": str(checkpoint_path),
        "checkpoint_transfer_filtered": str(transfer_path),
        "feature_metrics_csv": str(metrics_path),
        "generated_eval_h5ad": str(eval_path),
        "missing_keys_when_loading_eval": missing,
        "unexpected_keys_when_loading_eval": unexpected,
        "metrics": metrics,
        "figure_summary": figure_summary,
        "eval_target_encoder_input_masked": True,
        "eval_shared_graph_mode": "spatial_only",
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def run_direction(direction: str, device: str, sample_name: str = "m1_FMP") -> dict[str, object]:
    train_kwargs, input_h5ad, config_path = load_current_config(sample_name, device)
    seed_everything(
        int(train_kwargs["random_seed"]),
        bool(train_kwargs["deterministic"]),
        bool(train_kwargs["deterministic_warn_only"]),
    )
    full_adata, train_data, eval_data, split_meta, split_payload = prepare_split(input_h5ad, train_kwargs)
    if direction == "st_to_sm":
        output_root = downstream_root / f"fig2j_{sample_name}"
        experiment_dir = output_root / "st_to_sm_generation_spatial_top_third"
        module = fig2j
    else:
        output_root = downstream_root / f"fig2k_{sample_name}"
        experiment_dir = output_root / "sm_to_st_generation_spatial_top_third"
        module = fig2k
    return save_common(
        direction=direction,
        module=module,
        output_root=output_root,
        experiment_dir=experiment_dir,
        input_h5ad=input_h5ad,
        config_path=config_path,
        sample_name=sample_name,
        train_kwargs=train_kwargs,
        full_adata=full_adata,
        train_data=train_data,
        eval_data=eval_data,
        split_meta=split_meta,
        split_payload=split_payload,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("direction", choices=["st_to_sm", "sm_to_st"])
    parser.add_argument("--sample-name", default="m1_FMP")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    summary = run_direction(args.direction, args.device, args.sample_name)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
