from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from SpaDTA_718.model.cluster_eval_utils import (
    compute_metrics,
    load_gt_from_h5ad,
    pca_project,
    resolve_rscript,
    run_mclust_fixed_k,
)


package_root = project_root / "SpaDTA_718"
default_run_dir = package_root / "runs" / "second" / "S15_T"
default_gt_h5ad = (
    Path("/bigdat2/user/hesy/spatialmeta/SpatialMETA/06_spatialmeta_groundtruth/06_spatialmeta_groundtruth")
    / "adata_joint_S15_T_hvf2800.h5ad"
)
default_embedding_key = "X_emb_decalign_linear"
default_gt_obs_key = "pathological_annotation"
default_pca_components = 20
default_random_seed = 0
default_cluster_key = "cluster_target_count_mclust"
default_mclust_model = "EEE"


def infer_target_n_clusters(gt_h5ad: Path, gt_obs_key: str) -> int:
    gt_adata = ad.read_h5ad(gt_h5ad)
    labels = gt_adata.obs[gt_obs_key].astype(str)
    return int(pd.unique(labels).size)


def discover_input_files(run_dir: Path) -> list[Path]:
    return sorted(path for path in run_dir.glob("*.h5ad") if not path.name.endswith("_mclust.h5ad"))


def choose_embedding_key(adata: ad.AnnData, requested_key: str) -> str:
    if requested_key in adata.obsm:
        return requested_key
    for key in ("X_q_mu_decalign_linear", "X_umap"):
        if key in adata.obsm:
            return key
    raise KeyError(f"embedding key {requested_key!r} not found; obsm keys: {list(adata.obsm.keys())}")


def process_single_file(
    *,
    input_h5ad: Path,
    gt_h5ad: Path,
    gt_obs_key: str,
    cluster_key: str,
    embedding_key: str,
    pca_components: int,
    random_seed: int,
    rscript: Path,
    work_dir: Path,
    target_n_clusters: int,
    model_name: str,
) -> dict[str, object]:
    adata = ad.read_h5ad(input_h5ad)
    effective_embedding_key = choose_embedding_key(adata, embedding_key)
    gt_labels = load_gt_from_h5ad(
        gt_h5ad,
        adata.obs_names,
        np.asarray(adata.obsm["spatial"], dtype=np.float64)[:, :2],
        gt_obs_key,
    )
    valid_mask = gt_labels.notna().to_numpy()
    reduced = pca_project(np.asarray(adata.obsm[effective_embedding_key], dtype=np.float64), pca_components)
    labels = run_mclust_fixed_k(
        reduced,
        n_clusters=target_n_clusters,
        random_seed=random_seed,
        rscript=rscript,
        work_dir=work_dir,
        model_name=model_name,
    ).astype(str)
    metrics = compute_metrics(gt_labels.to_numpy()[valid_mask], labels[valid_mask])

    categories = sorted(pd.unique(labels).tolist())
    adata.obs[cluster_key] = pd.Categorical(labels, categories=categories, ordered=True)
    adata.obs[f"{cluster_key}_gt"] = pd.Categorical(gt_labels.astype(str))

    output_h5ad = input_h5ad.with_name(f"{input_h5ad.stem}_mclust.h5ad")
    metrics_csv = input_h5ad.with_name(f"{input_h5ad.stem}_mclust_metrics.csv")
    summary_json = input_h5ad.with_name(f"{input_h5ad.stem}_mclust_summary.json")

    adata.uns[f"{cluster_key}_meta"] = {
        "method": "mclust",
        "model_name": model_name,
        "target_n_clusters": int(target_n_clusters),
        "random_seed": int(random_seed),
        "embedding_key": effective_embedding_key,
        "pca_components": int(pca_components),
        "gt_h5ad": str(gt_h5ad),
        "gt_obs_key": gt_obs_key,
    }
    adata.write_h5ad(output_h5ad)

    row = {
        "input_h5ad": str(input_h5ad),
        "output_h5ad": str(output_h5ad),
        "metrics_csv": str(metrics_csv),
        "summary_json": str(summary_json),
        "embedding_key": effective_embedding_key,
        "cluster_key": cluster_key,
        "target_n_clusters": int(target_n_clusters),
        "observed_pred_clusters": int(pd.Series(labels).nunique()),
        "matched_spots": int(valid_mask.sum()),
        "random_seed": int(random_seed),
        "pca_components": int(pca_components),
        **metrics,
    }
    pd.DataFrame([row]).to_csv(metrics_csv, index=False)
    summary_json.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed-k mclust on existing SM or ATAC result h5ad files.")
    parser.add_argument("--run-dir", type=Path, default=default_run_dir)
    parser.add_argument("--input-h5ad", type=Path, nargs="*", default=None)
    parser.add_argument("--gt-h5ad", type=Path, default=default_gt_h5ad)
    parser.add_argument("--gt-obs-key", default=default_gt_obs_key)
    parser.add_argument("--embedding-key", default=default_embedding_key)
    parser.add_argument("--cluster-key", default=default_cluster_key)
    parser.add_argument("--pca-components", type=int, default=default_pca_components)
    parser.add_argument("--random-seed", type=int, default=default_random_seed)
    parser.add_argument("--mclust-model", default=default_mclust_model)
    parser.add_argument("--rscript", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_n_clusters = infer_target_n_clusters(args.gt_h5ad, args.gt_obs_key)
    rscript = resolve_rscript(args.rscript)
    if args.input_h5ad:
        input_files = [path.expanduser().resolve() for path in args.input_h5ad]
    else:
        input_files = discover_input_files(args.run_dir.expanduser().resolve())

    rows: list[dict[str, object]] = []
    work_dir = args.run_dir.expanduser().resolve() / "_mclust_tmp"
    work_dir.mkdir(parents=True, exist_ok=True)
    for input_h5ad in input_files:
        row = process_single_file(
            input_h5ad=input_h5ad,
            gt_h5ad=args.gt_h5ad,
            gt_obs_key=args.gt_obs_key,
            cluster_key=args.cluster_key,
            embedding_key=args.embedding_key,
            pca_components=args.pca_components,
            random_seed=args.random_seed,
            rscript=rscript,
            work_dir=work_dir,
            target_n_clusters=target_n_clusters,
            model_name=args.mclust_model,
        )
        rows.append(row)
        print(
            f"[mclust] {input_h5ad.name}: matched={row['matched_spots']} "
            f"ARI={row['ARI']:.4f} NMI={row['NMI']:.4f}",
            flush=True,
        )

    summary_csv = args.run_dir.expanduser().resolve() / "mclust_batch_summary.csv"
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    print(f"[mclust] summary_csv={summary_csv}", flush=True)


if __name__ == "__main__":
    main()
