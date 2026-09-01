from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA


project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from SpaDTA_718.model.cluster_eval_utils import (
    compute_metrics,
    load_gt_from_annotation_csv,
    load_gt_from_h5ad,
    pca_project,
    resolve_rscript,
    run_mclust_fixed_k,
)


package_root = project_root / "SpaDTA_718"
sm_processed_root = Path("/bigdat2/user/hesy/spatialmeta/SpatialMETA/processed")
sm_gt_root = Path(
    "/bigdat2/user/hesy/spatialmeta/SpatialMETA/06_spatialmeta_groundtruth/06_spatialmeta_groundtruth"
)
atac_annotation_root = package_root / "data" / "annotations"

target_clusters = {
    "sm": {
        "248_T": 18,
        "R114_T": 9,
        "S15_T": 14,
        "X49_T": 10,
        "Y27_T": 10,
        "Y7_T": 15,
        "m1_FMP": 14,
        "m3_FMP": 12,
        "m4_FMP": 14,
    },
    "atac": {
        "Mouse_Brain_E11_S1": 5,
        "Mouse_Brain_E13_S1": 7,
        "Mouse_Brain_E15_S1": 11,
        "Mouse_Brain_E18_S1": 10,
    },
}
default_cluster_seeds = {"sm": 0, "atac": 2020}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one saved SpaDTA SM or ATAC embedding with the final comparison protocol."
    )
    parser.add_argument("--modality", choices=("sm", "atac"), required=True)
    parser.add_argument("--sample-name", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--embedding-name", default="branch_scaled_full")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--rscript", type=Path, default=None)
    return parser.parse_args()


def load_snapshot(run_dir: Path, epoch: int, embedding_name: str) -> tuple[np.ndarray, pd.Index, Path]:
    snapshot_dir = run_dir / "saved_epoch_embeddings" / f"epoch_{epoch:04d}"
    embedding_path = snapshot_dir / f"{embedding_name}.npy"
    spot_ids_path = snapshot_dir / "spot_ids.csv"
    if not embedding_path.is_file():
        raise FileNotFoundError(embedding_path)
    if not spot_ids_path.is_file():
        raise FileNotFoundError(spot_ids_path)

    embedding = np.asarray(np.load(embedding_path), dtype=np.float64)
    spot_ids = pd.Index(pd.read_csv(spot_ids_path)["spot_id"].astype(str))
    if embedding.ndim != 2 or embedding.shape[0] != len(spot_ids):
        raise ValueError(
            f"embedding/spot mismatch: embedding={embedding.shape}, spot_ids={len(spot_ids)}"
        )
    return embedding, spot_ids, snapshot_dir


def load_labels(modality: str, sample_name: str, spot_ids: pd.Index) -> pd.Series:
    if modality == "atac":
        annotation_csv = atac_annotation_root / f"{sample_name}_manual_anno.csv"
        return load_gt_from_annotation_csv(annotation_csv, spot_ids)

    processed_h5ad = sm_processed_root / f"{sample_name}.h5ad"
    gt_h5ad = sm_gt_root / f"adata_joint_{sample_name}_hvf2800.h5ad"
    adata = sc.read_h5ad(processed_h5ad)
    adata.obs_names = adata.obs_names.astype(str)
    missing = spot_ids.difference(adata.obs_names)
    if len(missing):
        raise KeyError(f"processed SM data is missing spot IDs, examples: {missing[:5].tolist()}")
    coords = np.asarray(adata[spot_ids].obsm["spatial"], dtype=np.float64)[:, :2]
    return load_gt_from_h5ad(gt_h5ad, spot_ids, coords, None)


def project_embedding(modality: str, embedding: np.ndarray, n_components: int = 20) -> np.ndarray:
    if modality == "sm":
        return pca_project(embedding, n_components)

    keep = embedding.std(axis=0) > 1.0e-8
    used = embedding[:, keep] if int(keep.sum()) >= 2 else embedding
    effective_components = min(n_components, used.shape[0] - 1, used.shape[1])
    return PCA(n_components=effective_components).fit_transform(used)


def main() -> None:
    args = parse_args()
    modality = str(args.modality)
    sample_name = str(args.sample_name)
    if sample_name not in target_clusters[modality]:
        known = ", ".join(target_clusters[modality])
        raise ValueError(f"unknown {modality} sample {sample_name!r}; expected one of: {known}")

    run_dir = args.run_dir.expanduser().resolve()
    embedding, spot_ids, snapshot_dir = load_snapshot(run_dir, args.epoch, args.embedding_name)
    labels = load_labels(modality, sample_name, spot_ids)
    valid = labels.notna().to_numpy()
    if int(valid.sum()) < 2:
        raise RuntimeError(f"too few matched labels: {int(valid.sum())}")

    valid_embedding = embedding[valid]
    valid_spot_ids = spot_ids[valid]
    labels_true = labels.iloc[np.flatnonzero(valid)].astype(str).to_numpy()
    projected = project_embedding(modality, valid_embedding)

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_dir / "final_protocol" / f"epoch_{args.epoch:04d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = default_cluster_seeds[modality]
    target = target_clusters[modality][sample_name]
    labels_pred = run_mclust_fixed_k(
        projected,
        n_clusters=target,
        random_seed=seed,
        rscript=resolve_rscript(args.rscript),
        work_dir=output_dir / "_mclust_tmp",
        model_name="EEE",
    ).astype(str)
    metrics = compute_metrics(labels_true, labels_pred)

    result = {
        "modality": modality,
        "sample": sample_name,
        "run_dir": str(run_dir),
        "snapshot_dir": str(snapshot_dir),
        "epoch": int(args.epoch),
        "embedding_name": str(args.embedding_name),
        "total_spots": int(len(spot_ids)),
        "matched_spots": int(valid.sum()),
        "target_clusters": int(target),
        "observed_clusters": int(pd.Series(labels_pred).nunique()),
        "pca_components": int(projected.shape[1]),
        "clusterer": f"PCA20+mclust_EEE_seed{seed}",
        **{key: float(value) for key, value in metrics.items()},
    }
    np.save(output_dir / "mclust_labels.npy", labels_pred)
    pd.DataFrame(
        {
            "spot_id": valid_spot_ids,
            "ground_truth": labels_true,
            "mclust_label": labels_pred,
        }
    ).to_csv(output_dir / "spot_labels.csv", index=False)
    pd.DataFrame([result]).to_csv(output_dir / "metrics.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
