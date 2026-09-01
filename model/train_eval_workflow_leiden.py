from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

from .cluster_eval_utils import compute_metrics
from .train_eval_workflow import (
    build_branch_scaled_full,
    build_model,
    build_train_kwargs,
    capture_rng_state,
    current_warmup_weight,
    normalize_train_overrides,
    pca_project_local,
    resolve_gt_labels_for_template,
    restore_rng_state,
    save_checkpoint,
    save_embedding_snapshot,
    select_labeled_embedding,
    seed_everything,
    serialize_table_value,
    summarize_branch_variance,
    update_global_metrics,
)
from .preprocess import validate_spadta_model_input


def find_resolution(
    features: np.ndarray,
    n_clusters: int,
    random_seed: int,
    n_neighbors: int,
    metric: str,
    tolerance: int = 0,
) -> float:
    values = np.asarray(features, dtype=np.float64)
    adata = sc.AnnData(values, dtype=values.dtype)
    use_neighbors = min(int(n_neighbors), max(1, adata.n_obs - 1))
    sc.pp.neighbors(adata, n_neighbors=use_neighbors, use_rep="X", metric=str(metric))

    obtained_clusters = -1
    iteration = 0
    lower = 0.0
    upper = 1000.0
    current_res = 0.0

    while obtained_clusters != int(n_clusters) and iteration < 100:
        current_res = (lower + upper) / 2.0
        clustered = sc.tl.leiden(
            adata,
            resolution=float(current_res),
            random_state=int(random_seed),
            copy=True,
        )
        labels = clustered.obs["leiden"].astype(str)
        obtained_clusters = int(labels.nunique())

        if int(n_clusters) - obtained_clusters > int(tolerance):
            lower = current_res
        elif obtained_clusters - int(n_clusters) > int(tolerance):
            upper = current_res

        iteration += 1

    return float(current_res)


def run_leiden_fixed_k(
    features: np.ndarray,
    n_clusters: int,
    random_seed: int,
    n_neighbors: int,
    metric: str = "euclidean",
) -> tuple[np.ndarray, float]:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"expected a 2D embedding matrix, got shape={values.shape}")
    if values.shape[0] < int(n_clusters):
        raise ValueError("number of clusters exceeds number of observations")

    resolution = find_resolution(
        values,
        n_clusters=int(n_clusters),
        random_seed=int(random_seed),
        n_neighbors=int(n_neighbors),
        metric=str(metric),
        tolerance=0,
    )
    adata = sc.AnnData(values, dtype=values.dtype)
    use_neighbors = min(int(n_neighbors), max(1, adata.n_obs - 1))
    sc.pp.neighbors(adata, n_neighbors=use_neighbors, use_rep="X", metric=str(metric))
    clustered = sc.tl.leiden(
        adata,
        resolution=float(resolution),
        random_state=int(random_seed),
        copy=True,
    )
    labels = clustered.obs["leiden"].astype(str).to_numpy()
    if labels.shape[0] != values.shape[0]:
        raise RuntimeError("leiden returned a different number of labels than input rows")
    return labels, float(resolution)


def evaluate_embedding_fast(
    *,
    q_mu_shared: np.ndarray,
    q_mu_st: np.ndarray,
    q_mu_sm: np.ndarray,
    labels_true: np.ndarray,
    target_n_clusters: int,
    pca_components: int,
    random_seed: int,
    n_neighbors: int,
    distance_metric: str,
    branch_scaled_shared_weight: float = 1.0,
    branch_scaled_st_weight: float = 1.0,
    branch_scaled_sm_weight: float = 1.0,
) -> dict[str, float]:
    branch_scaled_full = build_branch_scaled_full(
        q_mu_shared=q_mu_shared,
        q_mu_st=q_mu_st,
        q_mu_sm=q_mu_sm,
        shared_weight=float(branch_scaled_shared_weight),
        st_weight=float(branch_scaled_st_weight),
        sm_weight=float(branch_scaled_sm_weight),
    )
    branch_scaled_full, labels_true, label_audit = select_labeled_embedding(
        branch_scaled_full,
        labels_true,
        target_n_clusters,
    )
    projected = pca_project_local(branch_scaled_full, pca_components)
    labels_pred, resolution = run_leiden_fixed_k(
        projected,
        n_clusters=target_n_clusters,
        random_seed=random_seed,
        n_neighbors=n_neighbors,
        metric=distance_metric,
    )
    metrics = compute_metrics(labels_true, labels_pred)
    metrics["observed_pred_clusters"] = float(pd.Series(labels_pred).nunique())
    metrics.update(label_audit)
    metrics["cluster_resolution_used"] = float(resolution)
    metrics["cluster_n_neighbors_used"] = float(n_neighbors)
    return {key: float(value) for key, value in metrics.items()}


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
    default_leiden_random_state: int,
    default_eval_every: int,
    default_max_epoch: int,
    default_n_per_batch: int,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Spatial multi-omic train/eval entry with periodic fixed-k Leiden evaluation.")
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
    parser.add_argument("--leiden-random-state", type=int, default=default_leiden_random_state)
    parser.add_argument("--gt-h5ad", type=Path, default=default_gt_h5ad)
    parser.add_argument("--annotation-csv", type=Path, default=default_annotation_csv)
    parser.add_argument("--gt-key", default=None)
    parser.add_argument("--train-overrides-json", default=None)
    parser.add_argument("--save-checkpoint-epochs", type=int, nargs="*", default=None)
    return parser


def run_train_eval_workflow_leiden(
    *,
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
        "cluster_eval_method": "leiden_fixed_k",
        "leiden_random_state": int(args.leiden_random_state),
        "cluster_n_neighbors": int(train_kwargs["cluster_n_neighbors"]),
        "cluster_distance_metric": "euclidean",
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
        raise ValueError("max_cells is preprocessing and is not supported during training")
    adata_train = sc.read_h5ad(input_h5ad_path)
    validate_spadta_model_input(
        adata_train,
        expression_graph_k=int(train_kwargs["spatial_contrastive_pos_k"]),
        spatial_context_k=int(train_kwargs["spatial_context_k"]),
    )
    adata_template = adata_train.copy()
    model = build_model(adata_train, train_kwargs)
    cached_gt_labels = resolve_gt_labels_for_template(
        adata_template=adata_template,
        gt_h5ad=gt_h5ad,
        annotation_csv=annotation_csv,
        gt_key=args.gt_key,
    )

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
                "leiden_ari": metrics["ARI"],
                "leiden_nmi": metrics["NMI"],
                "cluster_eval_method": "leiden_fixed_k",
                "cluster_resolution_used": metrics["cluster_resolution_used"],
                "cluster_n_neighbors_used": metrics["cluster_n_neighbors_used"],
                "cluster_distance_metric": "euclidean",
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
        current_contrastive_weight = current_warmup_weight(
            target=float(train_kwargs["spatial_contrastive_weight"]),
            warmup_epochs=int(train_kwargs["spatial_contrastive_warmup_epochs"]),
            epoch_num=int(epoch_num),
        )
        training_row = {
            "config_name": config_name,
            "sample_name": args.sample_name,
            "epoch": int(epoch_num),
            "reconstruction_loss_st": float(stats["reconstruction_loss_st"]),
            "reconstruction_loss_sm": float(stats["reconstruction_loss_sm"]),
            "kldiv_loss": float(stats["kldiv_loss"]),
            "dec_loss": float(stats["dec_loss"]),
            "hete_loss": float(stats["hete_loss"]),
            "homo_loss": float(stats["homo_loss"]),
            "task_loss_shared": float(stats["task_loss_shared"]),
            "task_loss_reconstruction_st": float(stats["task_loss_reconstruction_st"]),
            "task_loss_reconstruction_sm": float(stats["task_loss_reconstruction_sm"]),
            "spatial_consistency_loss": float(stats["spatial_consistency_loss"]),
            "spatial_contrastive_loss": float(stats["spatial_contrastive_loss"]),
            "spatial_negative_margin_loss": float(stats["spatial_negative_margin_loss"]),
            "private_latent_ceiling_loss": float(stats["private_latent_ceiling_loss"]),
            "weighted_private_latent_ceiling_term": float(stats["weighted_private_latent_ceiling_term"]),
            "current_private_latent_ceiling_weight": float(stats["current_private_latent_ceiling_weight"]),
            "private_latent_shared_std_reference": float(stats["private_latent_shared_std_reference"]),
            "private_st_latent_std_mean": float(stats["private_st_latent_std_mean"]),
            "private_sm_latent_std_mean": float(stats["private_sm_latent_std_mean"]),
            "private_st_latent_excess_fraction": float(stats["private_st_latent_excess_fraction"]),
            "private_sm_latent_excess_fraction": float(stats["private_sm_latent_excess_fraction"]),
            "total_loss": float(stats["total_loss"]),
            "task_weight_shared": float(stats["task_weight_shared"]),
            "task_weight_reconstruction_st": float(stats["task_weight_reconstruction_st"]),
            "task_weight_reconstruction_sm": float(stats["task_weight_reconstruction_sm"]),
            "spatial_contrastive_weight": float(current_contrastive_weight),
            "weighted_contrastive_term": float(current_contrastive_weight * float(stats["spatial_contrastive_loss"])),
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
            "decoder_private_mask_probability_current": float(stats["decoder_private_mask_probability_current"]),
            "decoder_st_private_actual_mask_fraction": float(stats["decoder_st_private_actual_mask_fraction"]),
            "decoder_sm_private_actual_mask_fraction": float(stats["decoder_sm_private_actual_mask_fraction"]),
            "decoder_st_private_masked_dimensions_mean": float(stats["decoder_st_private_masked_dimensions_mean"]),
            "decoder_sm_private_masked_dimensions_mean": float(stats["decoder_sm_private_masked_dimensions_mean"]),
            "decoder_st_private_kept_dimensions_mean": float(stats["decoder_st_private_kept_dimensions_mean"]),
            "decoder_sm_private_kept_dimensions_mean": float(stats["decoder_sm_private_kept_dimensions_mean"]),
            "spatial_contrastive_early_stop_recent_slope": float(stats["spatial_contrastive_early_stop_recent_slope"]),
            "spatial_contrastive_early_stop_recent_abs_slope": float(
                stats["spatial_contrastive_early_stop_recent_abs_slope"]
            ),
            "spatial_contrastive_early_stop_recent_mean": float(stats["spatial_contrastive_early_stop_recent_mean"]),
            "spatial_contrastive_early_stop_consecutive_hits": int(
                stats["spatial_contrastive_early_stop_consecutive_hits"]
            ),
            "spatial_contrastive_early_stop_triggered": int(stats["spatial_contrastive_early_stop_triggered"]),
        }
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
        if not should_eval:
            return

        current_kl_weight = float(train_kwargs["kl_weight"])
        row = {
            "config_name": config_name,
            "sample_name": args.sample_name,
            "epoch": int(epoch_num),
            "n_per_batch": int(train_kwargs["n_per_batch"]),
            "max_epoch": int(train_kwargs["max_epoch"]),
            "leiden_random_state": int(args.leiden_random_state),
            "reconstruction_loss_st": stats["reconstruction_loss_st"],
            "reconstruction_loss_sm": stats["reconstruction_loss_sm"],
            "kldiv_loss": stats["kldiv_loss"],
            "kl_weight": current_kl_weight,
            "weighted_kl_term": float(current_kl_weight * float(stats["kldiv_loss"])),
            "task_loss_shared": stats["task_loss_shared"],
            "total_loss": stats["total_loss"],
            "task_weight_shared": stats["task_weight_shared"],
            "task_weight_reconstruction_st": stats["task_weight_reconstruction_st"],
            "task_weight_reconstruction_sm": stats["task_weight_reconstruction_sm"],
            "spatial_contrastive_loss": stats["spatial_contrastive_loss"],
            "spatial_negative_margin_loss": stats["spatial_negative_margin_loss"],
            "private_latent_ceiling_loss": float(stats["private_latent_ceiling_loss"]),
            "weighted_private_latent_ceiling_term": float(stats["weighted_private_latent_ceiling_term"]),
            "current_private_latent_ceiling_weight": float(stats["current_private_latent_ceiling_weight"]),
            "private_latent_shared_std_reference": float(stats["private_latent_shared_std_reference"]),
            "private_st_latent_std_mean": float(stats["private_st_latent_std_mean"]),
            "private_sm_latent_std_mean": float(stats["private_sm_latent_std_mean"]),
            "private_st_latent_excess_fraction": float(stats["private_st_latent_excess_fraction"]),
            "private_sm_latent_excess_fraction": float(stats["private_sm_latent_excess_fraction"]),
            "spatial_contrastive_weight": float(current_contrastive_weight),
            "weighted_contrastive_term": float(current_contrastive_weight * float(stats["spatial_contrastive_loss"])),
            "spatial_contrastive_mode": str(train_kwargs["spatial_contrastive_mode"]),
            "spatial_negative_margin": float(train_kwargs["spatial_negative_margin"]),
            "spatial_positive_weighting": str(train_kwargs["spatial_positive_weighting"]),
            "spatial_positive_aggregation": str(train_kwargs["spatial_positive_aggregation"]),
            "spatial_positive_weight_temperature": float(train_kwargs["spatial_positive_weight_temperature"]),
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
            "decoder_private_mask_probability_current": float(stats["decoder_private_mask_probability_current"]),
            "decoder_st_private_actual_mask_fraction": float(stats["decoder_st_private_actual_mask_fraction"]),
            "decoder_sm_private_actual_mask_fraction": float(stats["decoder_sm_private_actual_mask_fraction"]),
            "decoder_st_private_masked_dimensions_mean": float(stats["decoder_st_private_masked_dimensions_mean"]),
            "decoder_sm_private_masked_dimensions_mean": float(stats["decoder_sm_private_masked_dimensions_mean"]),
            "decoder_st_private_kept_dimensions_mean": float(stats["decoder_st_private_kept_dimensions_mean"]),
            "decoder_sm_private_kept_dimensions_mean": float(stats["decoder_sm_private_kept_dimensions_mean"]),
            "spatial_contrastive_early_stop_recent_slope": float(stats["spatial_contrastive_early_stop_recent_slope"]),
            "spatial_contrastive_early_stop_recent_abs_slope": float(
                stats["spatial_contrastive_early_stop_recent_abs_slope"]
            ),
            "spatial_contrastive_early_stop_recent_mean": float(stats["spatial_contrastive_early_stop_recent_mean"]),
            "spatial_contrastive_early_stop_window_epochs": int(stats["spatial_contrastive_early_stop_window_epochs"]),
            "spatial_contrastive_early_stop_slope_threshold": float(
                stats["spatial_contrastive_early_stop_slope_threshold"]
            ),
            "spatial_contrastive_early_stop_min_epoch": int(stats["spatial_contrastive_early_stop_min_epoch"]),
            "spatial_contrastive_early_stop_patience": int(stats["spatial_contrastive_early_stop_patience"]),
            "spatial_contrastive_early_stop_consecutive_hits": int(
                stats["spatial_contrastive_early_stop_consecutive_hits"]
            ),
            "spatial_contrastive_early_stop_triggered": int(stats["spatial_contrastive_early_stop_triggered"]),
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
            branch_scaled_full = build_branch_scaled_full(
                q_mu_shared=q_mu_shared,
                q_mu_st=q_mu_st,
                q_mu_sm=q_mu_sm,
                shared_weight=float(train_kwargs["branch_scaled_shared_weight"]),
                st_weight=float(train_kwargs["branch_scaled_st_weight"]),
                sm_weight=float(train_kwargs["branch_scaled_sm_weight"]),
            )
            metrics = evaluate_embedding_fast(
                q_mu_shared=q_mu_shared,
                q_mu_st=q_mu_st,
                q_mu_sm=q_mu_sm,
                labels_true=cached_gt_labels,
                target_n_clusters=args.target_n_clusters,
                pca_components=args.pca_components,
                random_seed=args.leiden_random_state,
                n_neighbors=int(train_kwargs["cluster_n_neighbors"]),
                distance_metric="euclidean",
                branch_scaled_shared_weight=float(train_kwargs["branch_scaled_shared_weight"]),
                branch_scaled_st_weight=float(train_kwargs["branch_scaled_st_weight"]),
                branch_scaled_sm_weight=float(train_kwargs["branch_scaled_sm_weight"]),
            )

            shared_stats = summarize_branch_variance(q_mu_shared)
            st_stats = summarize_branch_variance(q_mu_st)
            sm_stats = summarize_branch_variance(q_mu_sm)
            full_total_variance = (
                shared_stats["total_variance"] + st_stats["total_variance"] + sm_stats["total_variance"]
            )
            row.update(
                {
                    "embedding_source": "branch_scaled_full",
                    "pca_dimension": int(args.pca_components),
                    "n_neighbors": int(train_kwargs["cluster_n_neighbors"]),
                    "distance_metric": "euclidean",
                    "number_of_clusters": int(args.target_n_clusters),
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

            if int(epoch_num) in save_embedding_epochs:
                save_embedding_snapshot(
                    output_root=args.output_root,
                    epoch=int(epoch_num),
                    spot_ids=adata_template.obs_names,
                    q_mu_shared=q_mu_shared,
                    q_mu_st=q_mu_st,
                    q_mu_sm=q_mu_sm,
                    branch_scaled_full=branch_scaled_full,
                    metrics=metrics,
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
        spatial_contrastive_use_all_latent=bool(train_kwargs["spatial_contrastive_use_all_latent"]),
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


def run_train_eval_leiden_cli(
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
    default_leiden_random_state: int,
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
        default_leiden_random_state=default_leiden_random_state,
        default_eval_every=default_eval_every,
        default_max_epoch=int(base_train_kwargs["max_epoch"]),
        default_n_per_batch=int(base_train_kwargs["n_per_batch"]),
    )
    args = parser.parse_args()
    run_train_eval_workflow_leiden(
        args=args,
        base_train_kwargs=base_train_kwargs,
    )
