from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import scanpy as sc
import torch

PROJECT_ROOT = Path("/data/user/hesy/projects/SpatialMETA")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from SpaDTA_718.model.train_eval_workflow import (
    build_model,
    pca_project_local,
    run_mclust_fixed_k,
)


PACKAGE_ROOT = PROJECT_ROOT / "SpaDTA_718"
MODEL_RUN_ROOT = PACKAGE_ROOT / "runs" / "SM"
OUTPUT_ROOT = PACKAGE_ROOT / "runs" / "sm_downstream" / "inputs"
CACHE_ROOT = PACKAGE_ROOT / "runs" / "_prepared_sm_cache"
DEFAULT_SAMPLES = ("Y7_T", "248_T", "m3_FMP", "X49_T")
RSCRIPT = Path("/data/user/hesy/miniconda3/envs/renv/bin/Rscript")


def load_checkpoint(path: Path, device: str) -> dict[str, object]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def export_sample(sample: str, device: str, overwrite: bool, embedding_only: bool = False) -> Path:
    run_dir = MODEL_RUN_ROOT / sample
    checkpoint_path = run_dir / "checkpoint_best_ari.pt"
    config_path = run_dir / "config.json"
    snapshot_dir = run_dir / "saved_best_embeddings" / "best_ari"
    output_dir = OUTPUT_ROOT / sample
    output_path = output_dir / f"{sample}_output.h5ad"
    if output_path.exists() and not overwrite:
        print(f"[prepare-downstream] reuse {output_path}", flush=True)
        return output_path

    config = json.loads(config_path.read_text(encoding="utf-8"))
    checkpoint = load_checkpoint(checkpoint_path, device)
    train_kwargs = dict(checkpoint["train_kwargs"])
    train_kwargs["device"] = device
    cache_path = CACHE_ROOT / f"{sample}_max{int(train_kwargs['max_cells'])}_countsnorm.h5ad"
    input_path = cache_path if cache_path.exists() else Path(config["input_h5ad_path"])
    print(f"[prepare-downstream] load {input_path}", flush=True)
    adata = sc.read_h5ad(input_path)
    branch_scaled_full = np.load(snapshot_dir / "branch_scaled_full.npy")
    spot_ids = pd.read_csv(snapshot_dir / "spot_ids.csv")["spot_id"].astype(str)
    if list(spot_ids) != list(adata.obs_names.astype(str)):
        raise ValueError(f"{sample}: saved embedding spot IDs do not match the prepared AnnData")

    projected = pca_project_local(branch_scaled_full, int(config["pca_components"]))
    clusters = run_mclust_fixed_k(
        projected,
        n_clusters=int(config["target_n_clusters"]),
        random_seed=int(config["mclust_random_seed"]),
        rscript=RSCRIPT,
        work_dir=output_dir / "_mclust_tmp",
        model_name=str(config["mclust_model_name"]),
    )

    adata.obsm["X_emb_decalign_linear"] = np.asarray(branch_scaled_full, dtype=np.float32)
    if not embedding_only:
        model = build_model(adata, train_kwargs)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        print(f"[prepare-downstream] infer {sample}", flush=True)
        _, recon_st, recon_sm, contribution_st, details = model._iterate_full(
            n_per_batch=int(train_kwargs["n_per_batch"]), latent_key="q_mu"
        )
        reconstruction = np.zeros((adata.n_obs, adata.n_vars), dtype=np.float32)
        reconstruction[:, np.asarray(model.st_mask, dtype=bool)] = recon_st
        reconstruction[:, np.asarray(model.sm_mask, dtype=bool)] = recon_sm
        adata.layers["reconstruction_decalign_linear"] = reconstruction
        adata.obsm["X_q_mu_shared_decalign_linear"] = np.asarray(details["q_mu_shared"], dtype=np.float32)
        for source_key, target_key in {
            "homo_st_embedding": "X_emb_homo_st_decalign_linear",
            "homo_sm_embedding": "X_emb_homo_sm_decalign_linear",
            "homo_joint_embedding": "X_emb_homo_joint_decalign_linear",
        }.items():
            if source_key in details:
                adata.obsm[target_key] = np.asarray(details[source_key], dtype=np.float32)
        contribution_st = np.asarray(contribution_st, dtype=np.float32)
        contribution_sm = np.asarray(details.get("contribution_sm", 1.0 - contribution_st), dtype=np.float32)
        adata.obs["contribution_st_decalign_linear"] = contribution_st
        adata.obs["contribution_sm_decalign_linear"] = contribution_sm
    cluster_labels = pd.Categorical(clusters.astype(str))
    adata.obs["decalign_linear_clusters"] = cluster_labels
    adata.obs["cluster_target_count"] = cluster_labels
    adata.uns["sm_downstream_provenance"] = {
        "model_run": str(run_dir.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "embedding": "saved_best_embeddings/best_ari/branch_scaled_full.npy",
        "cluster_method": "mclust EEE fixed-k on PCA20 of branch_scaled_full",
        "cluster_seed": int(config["mclust_random_seed"]),
        "target_clusters": int(config["target_n_clusters"]),
        "embedding_only": bool(embedding_only),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp.h5ad")
    adata.write_h5ad(tmp_path, compression="gzip")
    tmp_path.replace(output_path)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "sample": sample,
                "output_h5ad": str(output_path.resolve()),
                "n_obs": int(adata.n_obs),
                "n_vars": int(adata.n_vars),
                **adata.uns["sm_downstream_provenance"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[prepare-downstream] wrote {output_path}", flush=True)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export current SpaDTA SM checkpoints for downstream figures.")
    parser.add_argument("--samples", nargs="+", default=list(DEFAULT_SAMPLES))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--embedding-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for sample in args.samples:
        export_sample(sample, args.device, args.overwrite, args.embedding_only)


if __name__ == "__main__":
    main()
