from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from spaDTA.model.workflow import run_single_sample

sample = '248_T'
package_root = Path(__file__).resolve().parents[1]
processed_root = Path("/bigdat2/user/hesy/spatialmeta/SpatialMETA/processed")
gt_root = Path("/bigdat2/user/hesy/spatialmeta/SpatialMETA/06_spatialmeta_groundtruth/06_spatialmeta_groundtruth")
output_root = package_root / "runs" / 'first'
config_name = sample
sample_name = sample
output_prefix_name = f"{sample}_output"
device = "cuda:5"
cluster_resolution = 0.41
train_kwargs = {
    "device": device,
    "max_epoch": 128,
    "n_per_batch": 128,
    "proj_dim": 256,
    "token_dim": 128,
    "n_latent": 10,
    "num_prototypes": 8,
    "max_cells": 0,
    "random_seed": 42,
    "cluster_random_seed": 0,
    "dropout_rate": 0.03,
    "cluster_n_neighbors": 15,
    "cluster_resolution": cluster_resolution,
    "reconstruction_st_weight": 0.75,
    "reconstruction_sm_weight": 0.25,
    "dec_weight": 1.0,
    "hete_weight": 0.0,
    "homo_weight": 0.01,
    "hete_warmup_epochs": 0,
    "homo_warmup_epochs": 0,
    "kl_weight": 0.0,
    "n_epochs_kl_warmup": 0,
    "lr": 5e-4,
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
    "spatial_contrastive_pos_k": 4,
    "spatial_contrastive_neg_k": 16,
    "spatial_contrastive_temperature": 0.2,
    "spatial_contrastive_neg_strategy": "mid",
    "standardize_inputs": False,
    "standardized_reconstruction": False,
    "deterministic": False,
    "deterministic_warn_only": False,
    "spatial_consistency_use_all_latent": False,
    "spatial_contrastive_use_all_latent": False,
}

print(f"[entry] sample={sample_name}", flush=True)
print(f"[entry] device={device}", flush=True)
print(f"[entry] output_root={output_root}", flush=True)
print(f"[entry] output_prefix_name={output_prefix_name}", flush=True)

run_single_sample(
    sample_name=sample_name,
    processed_root=processed_root,
    gt_root=gt_root,
    output_root=output_root,
    config_name=config_name,
    train_kwargs=train_kwargs,
    output_prefix_name=output_prefix_name,
)
