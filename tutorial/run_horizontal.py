from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from SpaDTA_718.model.horizontal_workflow import run_horizontal_samples

package_root = Path(__file__).resolve().parents[1]

input_h5ad_path = None
processed_root = Path("/bigdat2/user/hesy/spatialmeta/SpatialMETA/SpaDTA_718_model_input/SM")
output_root = package_root / "runs" / "horizontal"
config_name = "ccrcc"
output_prefix_name = "ccrcc"

sample_names = [
    "R114_T",
    "S15_T",
    "X49_T",
    "Y7_T",
    "Y27_T",
]
sample_count = 5

device = "cuda:7"
batch_key = "sample"
train_kwargs = {
    "batch_key": batch_key,
    "device": device,
    "max_epoch": 256,
    "n_per_batch": 256,
    "proj_dim": 256,
    "token_dim": 128,
    "n_latent": 10,
    "num_prototypes": 8,
    "max_cells_per_sample": 0,
    "random_seed": 42,
    "cluster_random_seed": 0,
    "dropout_rate": 0.03,
    "cluster_n_neighbors": 15,
    "cluster_resolution": 1.0,
    "reconstruction_st_weight": 0.75,
    "reconstruction_sm_weight": 0.25,
    "dec_weight": 1.0,
    "hete_weight": 0.0,
    "homo_weight": 0.01,
    "horizontal_weight": 1.0,
    "hete_warmup_epochs": 0,
    "homo_warmup_epochs": 0,
    "horizontal_warmup_epochs": 16,
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
    "batch_embedding": "embedding",
    "batch_hidden_dim": 8,
    "posthoc_batch_method": "center",
}

if input_h5ad_path is None and len(sample_names) != sample_count:
    raise ValueError(
        f"sample_count={sample_count}, but sample_names has {len(sample_names)} entries: {sample_names}"
    )

print(f"[entry-horizontal] config_name={config_name}", flush=True)
print(f"[entry-horizontal] device={device}", flush=True)
print(f"[entry-horizontal] output_root={output_root}", flush=True)
print(f"[entry-horizontal] output_prefix_name={output_prefix_name}", flush=True)
print(f"[entry-horizontal] sample_count={sample_count}", flush=True)
if input_h5ad_path is not None:
    print(f"[entry-horizontal] input_h5ad={input_h5ad_path}", flush=True)
else:
    print(f"[entry-horizontal] processed_root={processed_root}", flush=True)
    print(f"[entry-horizontal] sample_names={sample_names}", flush=True)

run_horizontal_samples(
    sample_count=sample_count,
    input_h5ad_path=input_h5ad_path,
    processed_root=processed_root,
    sample_names=sample_names,
    output_root=output_root,
    config_name=config_name,
    output_prefix_name=output_prefix_name,
    train_kwargs=train_kwargs,
)
