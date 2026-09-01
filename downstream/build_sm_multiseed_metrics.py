from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree, distance_matrix
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from SpaDTA_718.model.cluster_eval_utils import pca_project


default_gt_root = Path(
    "/bigdat2/user/hesy/spatialmeta/SpatialMETA/06_spatialmeta_groundtruth/06_spatialmeta_groundtruth"
)
default_model_input_root = Path(
    "/bigdat2/user/hesy/spatialmeta/SpatialMETA/"
    "SpaDTA_718_model_input_preselect800_20260719/SM"
)
default_rscript = Path("/data/user/hesy/miniconda3/envs/renv/bin/Rscript")


def compute_chaos(labels: np.ndarray, location: np.ndarray) -> float:
    location = np.asarray(location, dtype=np.float64)[:, :2]
    matched_location = StandardScaler().fit_transform(location)
    values = []
    total_points = 0
    for cluster in np.unique(labels):
        points = matched_location[labels == cluster]
        if len(points) <= 2:
            continue
        distances = distance_matrix(points, points)
        np.fill_diagonal(distances, np.inf)
        values.extend(np.min(distances, axis=1).tolist())
        total_points += len(points)
    return float(np.sum(values) / total_points) if total_points else float("nan")


def compute_pas(labels: np.ndarray, location: np.ndarray) -> float:
    location = np.asarray(location, dtype=np.float64)[:, :2]
    distances = distance_matrix(location, location)
    np.fill_diagonal(distances, np.inf)
    neighbours = np.argsort(distances, axis=1)[:, :10]
    disagree = (labels[neighbours] != labels[:, None]).sum(axis=1) > 5
    return float(np.sum(disagree) / len(labels)) if len(labels) else float("nan")


def run_mclust(points: np.ndarray, target_clusters: int, seed: int, rscript: Path) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="sm_multiseed_mclust_") as tmp:
        tmp_path = Path(tmp)
        input_path = tmp_path / "embedding.csv"
        output_path = tmp_path / "labels.csv"
        script_path = tmp_path / "run.R"
        pd.DataFrame(points).to_csv(input_path, index=False)
        script_path.write_text(
            "suppressPackageStartupMessages(library(mclust))\n"
            "args <- commandArgs(trailingOnly=TRUE)\n"
            "set.seed(as.integer(args[4]))\n"
            "x <- read.csv(args[1], check.names=FALSE)\n"
            "fit <- Mclust(x, G=as.integer(args[3]), modelNames='EEE')\n"
            "write.csv(data.frame(cluster=as.integer(fit$classification)), args[2], row.names=FALSE, quote=FALSE)\n",
            encoding="ascii",
        )
        subprocess.run(
            [str(rscript), str(script_path), str(input_path), str(output_path), str(target_clusters), str(seed)],
            check=True,
            capture_output=True,
            text=True,
        )
        return pd.read_csv(output_path)["cluster"].to_numpy(dtype=str)


def load_ground_truth(config: dict[str, object], spot_ids: pd.Index, coords: np.ndarray) -> pd.Series:
    import scanpy as sc

    gt = sc.read_h5ad(Path(str(config["gt_h5ad"])))
    gt.obs_names = gt.obs_names.astype(str)
    gt_key = "pathological_annotation" if "pathological_annotation" in gt.obs.columns else "annotation"
    labels = gt.obs[gt_key].astype(object)
    if set(spot_ids) == set(gt.obs_names):
        return labels.loc[spot_ids].set_axis(spot_ids)
    tree = cKDTree(np.asarray(gt.obsm["spatial"], dtype=np.float64)[:, :2])
    distances, indices = tree.query(coords, k=1)
    result = pd.Series(pd.NA, index=spot_ids, dtype="object")
    matched = distances < 5.0
    result.iloc[np.flatnonzero(matched)] = labels.iloc[indices[matched]].to_numpy()
    return result


def evaluate_run(run_dir: Path, output_name: str, model_input_root: Path) -> Path:
    import scanpy as sc

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    snapshot = run_dir / "saved_epoch_embeddings" / "epoch_0300"
    embedding = np.asarray(np.load(snapshot / "branch_scaled_full.npy"), dtype=np.float64)
    spot_ids = pd.Index(pd.read_csv(snapshot / "spot_ids.csv")["spot_id"].astype(str))
    input_data = sc.read_h5ad(model_input_root / f"{config['sample_name']}.h5ad")
    input_data.obs_names = input_data.obs_names.astype(str)
    coords = np.asarray(input_data[spot_ids].obsm["spatial"], dtype=np.float64)[:, :2]
    labels = load_ground_truth(config, spot_ids, coords)
    valid = labels.notna().to_numpy()
    labels_true = labels.iloc[np.flatnonzero(valid)].astype(str).to_numpy()
    projected = pca_project(embedding[valid], 20)
    target_clusters = int(config["target_n_clusters"])
    cluster_seed = int(config.get("mclust_random_seed", 0))
    labels_pred = run_mclust(projected, target_clusters, cluster_seed, Path(str(config.get("rscript", default_rscript))))
    valid_coords = coords[valid]
    result = {
        "method": "SpaDTA",
        "sample": str(config["sample_name"]),
        "training_seed": int(config["train_kwargs"]["random_seed"]),
        "mclust_seed": cluster_seed,
        "matched_spots": int(valid.sum()),
        "target_clusters": target_clusters,
        "predicted_clusters": int(pd.Series(labels_pred).nunique()),
        "ARI": float(adjusted_rand_score(labels_true, labels_pred)),
        "NMI": float(normalized_mutual_info_score(labels_true, labels_pred)),
        "CHAOS": compute_chaos(labels_pred, valid_coords),
        "PAS": compute_pas(labels_pred, valid_coords),
        "embedding_name": "branch_scaled_full",
        "pca_components": int(projected.shape[1]),
        "clusterer": "PCA20+mclust_EEE",
    }
    output_path = run_dir / output_name
    pd.DataFrame([result]).to_csv(output_path, index=False, float_format="%.9f")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one SM multiseed run directory.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-name", default="sm_metrics.csv")
    parser.add_argument("--model-input-root", type=Path, default=default_model_input_root)
    args = parser.parse_args()
    output_path = evaluate_run(
        args.run_dir.expanduser().resolve(),
        args.output_name,
        args.model_input_root.expanduser().resolve(),
    )
    print(output_path, flush=True)


if __name__ == "__main__":
    main()
