from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from SpaDTA_718.model.train_eval_workflow import run_train_eval_cli


package_root = Path(__file__).resolve().parents[1]
processed_root = Path(
    "/bigdat2/user/hesy/spatialmeta/SpatialMETA/SpaDTA_718_model_input_preselect800_20260719/SM"
)
sample = "248_T"
gt_root = Path(
    "/bigdat2/user/hesy/spatialmeta/SpatialMETA/06_spatialmeta_groundtruth/06_spatialmeta_groundtruth"
)
target_clusters_by_sample = {
    "248_T": 18,
    "R114_T": 9,
    "S15_T": 14,
    "X49_T": 10,
    "Y27_T": 10,
    "Y7_T": 15,
    "m1_FMP": 14,
    "m3_FMP": 12,
    "m4_FMP": 14,
}

device = "cuda:7"
cluster_resolution = 1.0
pca_components = 20
cluster_random_state = 0
eval_every = 300
base_train_kwargs = {
    "max_epoch": 300,
    "n_per_batch": 512,
    "proj_dim": 256,
    "token_dim": 128,
    "n_latent": 32,
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
    "n_epochs_kl_warmup": 120,
    "shared_kl_weight_scale": 1.0,
    "private_kl_weight_scale": 2.5,
    "late_kl_start_epoch": 0,
    "late_kl_ramp_epochs": 0,
    "late_shared_kl_weight_scale": 1.0,
    "late_private_kl_weight_scale": 2.5,
    "lr": 1e-4,
    "weight_decay": 1e-6,
    "reconstruction_reduction": "mean",
    "reconstruction_method_st": "zinb",
    "reconstruction_method_sm": "g",
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
    "spatial_contrastive_weight": 0.02,
    "spatial_contrastive_warmup_epochs": 16,
    "spatial_contrastive_pos_k": 3,
    "spatial_contrastive_neg_k": 8,
    "spatial_contrastive_temperature": 0.2,
    "spatial_contrastive_neg_strategy": "mid",
    "spatial_contrastive_mode": "positive_negative",
    "spatial_negative_margin": 0.2,
    "spatial_negative_margin_weight": 0.0,
    "spatial_negative_margin_warmup_epochs": 40,
    "spatial_negative_margin_stop_epoch": 60,
    "spatial_positive_weighting": "uniform",
    "spatial_positive_aggregation": "shared_numerator",
    "spatial_positive_weight_temperature": 1.0,
    "standardize_inputs": False,
    "standardized_reconstruction": False,
    "feature_input_mode": False,
    "deterministic": False,
    "deterministic_warn_only": False,
    "spatial_consistency_use_all_latent": False,
    "spatial_contrastive_use_all_latent": False,
    "shared_latent_std_weight": 0.0,
    "shared_latent_cov_weight": 0.0,
    "shared_latent_geometry_warmup_epochs": 16,
    "shared_latent_std_target": 1.0,
    "private_latent_ceiling_weight": 0.0,
    "private_latent_ceiling_ratio": 0.9,
    "private_latent_ceiling_start_epoch": 0,
    "private_latent_ceiling_ramp_epochs": 0,
    "decoder_hidden_dim": 2048,
    "decoder_num_layers": 1,
    "embedding_eval_mode": "branch_scaled_full",
    "decoder_private_feature_masking": False,
    "decoder_private_mask_probability": 0.3,
    "decoder_private_mask_warmup_start": 0,
    "decoder_private_mask_warmup_end": 0,
    "save_embedding_epochs": [120, 180, 240, 300],
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
        raise ValueError(f"unknown SM sample: {selected_sample}")
    run_train_eval_cli(
        package_root=package_root,
        default_sample=sample,
        default_processed_root=processed_root,
        default_output_root=package_root / "runs" / "sm_final" / selected_sample,
        default_gt_h5ad=gt_root / f"adata_joint_{selected_sample}_hvf2800.h5ad",
        default_annotation_csv=package_root / "data" / "unused_annotation.csv",
        default_device=device,
        default_cluster_resolution=cluster_resolution,
        default_target_n_clusters=target_clusters_by_sample[selected_sample],
        default_pca_components=pca_components,
        default_cluster_random_state=cluster_random_state,
        default_eval_every=eval_every,
        base_train_kwargs=base_train_kwargs,
    )
