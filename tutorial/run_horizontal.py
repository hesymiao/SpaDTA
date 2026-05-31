from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from spaDTA.model.horizontal_workflow import run_horizontal_samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run spaDTA horizontal integration with explicit sample inputs.",
    )
    parser.add_argument("--processed-root", type=Path, default=None)
    parser.add_argument("--input-h5ad", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config-name", type=str, required=True)
    parser.add_argument("--output-prefix-name", type=str, required=True)
    parser.add_argument("--sample-count", type=int, required=True)
    parser.add_argument("--sample-names", nargs="*", default=None)
    parser.add_argument("--batch-key", type=str, default="sample")
    parser.add_argument("--device", type=str, required=True)

    parser.add_argument("--max-epoch", type=int, default=128)
    parser.add_argument("--n-per-batch", type=int, default=128)
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--token-dim", type=int, default=128)
    parser.add_argument("--n-latent", type=int, default=10)
    parser.add_argument("--num-prototypes", type=int, default=8)
    parser.add_argument("--max-cells-per-sample", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--cluster-random-seed", type=int, default=0)
    parser.add_argument("--dropout-rate", type=float, default=0.03)
    parser.add_argument("--cluster-n-neighbors", type=int, default=15)
    parser.add_argument("--cluster-resolution", type=float, default=1.0)

    parser.add_argument("--reconstruction-st-weight", type=float, default=0.75)
    parser.add_argument("--reconstruction-sm-weight", type=float, default=0.25)
    parser.add_argument("--dec-weight", type=float, default=1.0)
    parser.add_argument("--hete-weight", type=float, default=0.0)
    parser.add_argument("--homo-weight", type=float, default=0.01)
    parser.add_argument("--horizontal-weight", type=float, default=1.0)
    parser.add_argument("--hete-warmup-epochs", type=int, default=0)
    parser.add_argument("--homo-warmup-epochs", type=int, default=0)
    parser.add_argument("--horizontal-warmup-epochs", type=int, default=16)
    parser.add_argument("--kl-weight", type=float, default=0.0)
    parser.add_argument("--n-epochs-kl-warmup", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--reconstruction-reduction", type=str, default="mean")
    parser.add_argument("--reconstruction-method-st", type=str, default="zinb")
    parser.add_argument("--reconstruction-method-sm", type=str, default="g")
    parser.add_argument("--balance-start-epoch", type=int, default=16)
    parser.add_argument("--balance-ema", type=float, default=0.8)
    parser.add_argument("--balance-weight-floor", type=float, default=0.05)

    parser.add_argument("--spatial-coord-hidden-dim", type=int, default=128)
    parser.add_argument("--spatial-context-hidden-dim", type=int, default=128)
    parser.add_argument("--spatial-context-k", type=int, default=12)
    parser.add_argument("--spatial-encoder-mode", type=str, default="local_context")
    parser.add_argument("--spatial-fourier-scales", nargs="+", type=float, default=[1.0, 2.0, 4.0])
    parser.add_argument("--spatial-token-scale", type=float, default=0.5)
    parser.add_argument("--spatial-token-dropout", type=float, default=0.15)
    parser.add_argument("--spatial-consistency-weight", type=float, default=0.0)
    parser.add_argument("--spatial-consistency-warmup-epochs", type=int, default=16)
    parser.add_argument("--spatial-contrastive-weight", type=float, default=0.02)
    parser.add_argument("--spatial-contrastive-warmup-epochs", type=int, default=16)
    parser.add_argument("--spatial-contrastive-pos-k", type=int, default=4)
    parser.add_argument("--spatial-contrastive-neg-k", type=int, default=16)
    parser.add_argument("--spatial-contrastive-temperature", type=float, default=0.2)
    parser.add_argument("--spatial-contrastive-neg-strategy", type=str, default="mid")

    parser.add_argument("--batch-embedding", type=str, default="embedding", choices=["embedding", "onehot"])
    parser.add_argument("--batch-hidden-dim", type=int, default=8)
    parser.add_argument(
        "--posthoc-batch-method",
        type=str,
        default="center",
        choices=["none", "combat", "center", "zscore"],
    )

    parser.add_argument("--standardize-inputs", action="store_true")
    parser.add_argument("--standardized-reconstruction", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--deterministic-warn-only", action="store_true")
    parser.add_argument("--spatial-consistency-use-all-latent", action="store_true")
    parser.add_argument("--spatial-contrastive-use-all-latent", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.input_h5ad is None and args.processed_root is None:
        raise ValueError("Either --input-h5ad or --processed-root must be provided.")
    if args.input_h5ad is None and not args.sample_names:
        raise ValueError("--sample-names is required when --input-h5ad is not provided.")
    if args.sample_names is not None and len(args.sample_names) != args.sample_count:
        raise ValueError(
            f"--sample-count={args.sample_count}, but received {len(args.sample_names)} values in --sample-names."
        )

    train_kwargs = {
        "device": args.device,
        "max_epoch": args.max_epoch,
        "n_per_batch": args.n_per_batch,
        "proj_dim": args.proj_dim,
        "token_dim": args.token_dim,
        "n_latent": args.n_latent,
        "num_prototypes": args.num_prototypes,
        "max_cells_per_sample": args.max_cells_per_sample,
        "random_seed": args.random_seed,
        "cluster_random_seed": args.cluster_random_seed,
        "dropout_rate": args.dropout_rate,
        "cluster_n_neighbors": args.cluster_n_neighbors,
        "cluster_resolution": args.cluster_resolution,
        "reconstruction_st_weight": args.reconstruction_st_weight,
        "reconstruction_sm_weight": args.reconstruction_sm_weight,
        "dec_weight": args.dec_weight,
        "hete_weight": args.hete_weight,
        "homo_weight": args.homo_weight,
        "horizontal_weight": args.horizontal_weight,
        "hete_warmup_epochs": args.hete_warmup_epochs,
        "homo_warmup_epochs": args.homo_warmup_epochs,
        "horizontal_warmup_epochs": args.horizontal_warmup_epochs,
        "kl_weight": args.kl_weight,
        "n_epochs_kl_warmup": args.n_epochs_kl_warmup,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "reconstruction_reduction": args.reconstruction_reduction,
        "reconstruction_method_st": args.reconstruction_method_st,
        "reconstruction_method_sm": args.reconstruction_method_sm,
        "balance_start_epoch": args.balance_start_epoch,
        "balance_ema": args.balance_ema,
        "balance_weight_floor": args.balance_weight_floor,
        "spatial_coord_hidden_dim": args.spatial_coord_hidden_dim,
        "spatial_context_hidden_dim": args.spatial_context_hidden_dim,
        "spatial_context_k": args.spatial_context_k,
        "spatial_encoder_mode": args.spatial_encoder_mode,
        "spatial_fourier_scales": tuple(float(scale) for scale in args.spatial_fourier_scales),
        "spatial_token_scale": args.spatial_token_scale,
        "spatial_token_dropout": args.spatial_token_dropout,
        "spatial_consistency_weight": args.spatial_consistency_weight,
        "spatial_consistency_warmup_epochs": args.spatial_consistency_warmup_epochs,
        "spatial_contrastive_weight": args.spatial_contrastive_weight,
        "spatial_contrastive_warmup_epochs": args.spatial_contrastive_warmup_epochs,
        "spatial_contrastive_pos_k": args.spatial_contrastive_pos_k,
        "spatial_contrastive_neg_k": args.spatial_contrastive_neg_k,
        "spatial_contrastive_temperature": args.spatial_contrastive_temperature,
        "spatial_contrastive_neg_strategy": args.spatial_contrastive_neg_strategy,
        "standardize_inputs": args.standardize_inputs,
        "standardized_reconstruction": args.standardized_reconstruction,
        "deterministic": args.deterministic,
        "deterministic_warn_only": args.deterministic_warn_only,
        "spatial_consistency_use_all_latent": args.spatial_consistency_use_all_latent,
        "spatial_contrastive_use_all_latent": args.spatial_contrastive_use_all_latent,
        "batch_embedding": args.batch_embedding,
        "batch_hidden_dim": args.batch_hidden_dim,
        "posthoc_batch_method": args.posthoc_batch_method,
    }

    print(f"[entry-horizontal] output_root={args.output_root}", flush=True)
    print(f"[entry-horizontal] config_name={args.config_name}", flush=True)
    print(f"[entry-horizontal] output_prefix_name={args.output_prefix_name}", flush=True)
    print(f"[entry-horizontal] sample_count={args.sample_count}", flush=True)
    print(f"[entry-horizontal] device={args.device}", flush=True)
    if args.sample_names:
        print(f"[entry-horizontal] sample_names={args.sample_names}", flush=True)
    if args.input_h5ad is not None:
        print(f"[entry-horizontal] input_h5ad={args.input_h5ad}", flush=True)
    if args.processed_root is not None:
        print(f"[entry-horizontal] processed_root={args.processed_root}", flush=True)

    run_horizontal_samples(
        output_root=args.output_root,
        config_name=args.config_name,
        output_prefix_name=args.output_prefix_name,
        sample_count=args.sample_count,
        batch_key=args.batch_key,
        train_kwargs=train_kwargs,
        input_h5ad_path=args.input_h5ad,
        processed_root=args.processed_root,
        sample_names=args.sample_names,
    )


if __name__ == "__main__":
    main()
