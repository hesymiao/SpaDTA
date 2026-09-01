from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from SpaDTA_718.model.train_eval_workflow import run_train_eval_cli


package_root = Path(__file__).resolve().parents[1]
processed_root = Path("/bigdat2/user/hesy/spatialmeta/SpatialMETA/SpaDTA_718_model_input/ATAC")
smart_data_root = Path("/bigdat2/user/hesy/spatialmeta/SpatialMETA/smart/SMART_data")
sample = "Mouse_Brain_E18_S1"
target_clusters_by_sample = {
    "Mouse_Brain_E11_S1": 5,
    "Mouse_Brain_E13_S1": 7,
    "Mouse_Brain_E15_S1": 11,
    "Mouse_Brain_E18_S1": 10,
}


# Accepted deterministic confirm-B configuration.
base_train_kwargs = {
    "max_epoch": 300,
    "n_per_batch": 256,
    "proj_dim": 256,
    "token_dim": 128,
    "n_latent": 10,
    "num_prototypes": 8,
    "max_cells": 0,
    "random_seed": 42,
    "cluster_random_seed": 0,
    "dropout_rate": 0.03,
    "cluster_n_neighbors": 15,
    "reconstruction_st_weight": 0.75,
    "reconstruction_sm_weight": 0.25,
    "dec_weight": 1.0,
    "hete_weight": 0.0,
    "homo_weight": 0.01,
    "hete_warmup_epochs": 0,
    "homo_warmup_epochs": 0,
    "kl_weight": 0.0,
    "n_epochs_kl_warmup": 0,
    "shared_kl_weight_scale": 1.0,
    "private_kl_weight_scale": 1.0,
    "late_kl_start_epoch": 0,
    "late_kl_ramp_epochs": 0,
    "late_shared_kl_weight_scale": 1.0,
    "late_private_kl_weight_scale": 1.0,
    "lr": 5e-4,
    "weight_decay": 1e-6,
    "reconstruction_reduction": "mean",
    "reconstruction_method_st": "zinb",
    "reconstruction_method_sm": "mse",
    "balance_start_epoch": 16,
    "balance_ema": 0.8,
    "balance_weight_floor": 0.05,
    "spatial_coord_hidden_dim": 128,
    "spatial_context_hidden_dim": 128,
    "spatial_context_k": 12,
    "spatial_encoder_mode": "local_context",
    "spatial_fourier_scales": (1.0, 2.0, 4.0),
    "spatial_token_scale": 0.5,
    "spatial_token_dropout": 0.15,
    "spatial_consistency_weight": 0.0,
    "spatial_consistency_warmup_epochs": 16,
    "spatial_contrastive_weight": 0.005,
    "spatial_contrastive_warmup_epochs": 16,
    "spatial_contrastive_pos_k": 4,
    "spatial_contrastive_neg_k": 16,
    "spatial_contrastive_temperature": 0.2,
    "spatial_contrastive_neg_strategy": "mid",
    "spatial_contrastive_mode": "positive_negative",
    "spatial_negative_margin": 0.2,
    "spatial_positive_weighting": "uniform",
    "spatial_positive_aggregation": "shared_numerator",
    "spatial_positive_weight_temperature": 1.0,
    "spatial_negative_margin_weight": 0.0,
    "spatial_negative_margin_warmup_epochs": 16,
    "spatial_negative_margin_stop_epoch": 0,
    "standardize_inputs": False,
    "standardized_reconstruction": False,
    "feature_input_mode": False,
    "deterministic": True,
    "deterministic_warn_only": False,
    "spatial_consistency_use_all_latent": False,
    "spatial_contrastive_use_all_latent": True,
    "spatial_contrastive_latent_mode": "auto",
    "decoder_hidden_dim": 384,
    "decoder_num_layers": 1,
    "decoder_private_feature_masking": True,
    "decoder_private_mask_probability": 0.3,
    "decoder_private_mask_warmup_start": 0,
    "decoder_private_mask_warmup_end": 0,
    "private_encoder_num_layers": 1,
    "private_encoder_activation": "none",
    "embedding_eval_mode": "branch_scaled_full",
    "save_embedding_epochs": [100, 200, 300],
    "spatial_contrastive_early_stop_enabled": False,
    "spatial_contrastive_early_stop_window_epochs": 70,
    "spatial_contrastive_early_stop_slope_threshold": 1.0e-4,
    "spatial_contrastive_early_stop_min_epoch": 400,
    "spatial_contrastive_early_stop_patience": 20,
}


def requested_sample() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sample-name", default=sample)
    args, _ = parser.parse_known_args()
    return str(args.sample_name)


if __name__ == "__main__":
    selected_sample = requested_sample()
    if selected_sample not in target_clusters_by_sample:
        raise ValueError(f"unknown ATAC sample: {selected_sample}")
    run_train_eval_cli(
        package_root=package_root,
        default_sample=sample,
        default_processed_root=processed_root,
        default_output_root=package_root / "runs" / "atac_final" / selected_sample,
        default_gt_h5ad=smart_data_root / selected_sample / "unused_gt.h5ad",
        default_annotation_csv=package_root / "data" / "annotations" / f"{selected_sample}_manual_anno.csv",
        default_device="cuda:4",
        default_cluster_resolution=1.0,
        default_target_n_clusters=target_clusters_by_sample[selected_sample],
        default_pca_components=20,
        default_cluster_random_state=2020,
        default_eval_every=300,
        base_train_kwargs=base_train_kwargs,
    )
