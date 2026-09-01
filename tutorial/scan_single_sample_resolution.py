from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial import cKDTree
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


project_root = Path("/data/user/hesy/projects/SpatialMETA")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

sample_name = "248_T"
embedding_key = "X_emb_decalign_linear"
cluster_key = "cluster_target_count"
neighbor_count = 15
cluster_random_seed = 0
spatial_match_threshold = 5.0
cpu_core_limit = 30
coarse_low_resolution_min = 0.01
coarse_low_resolution_max = 1.00
coarse_low_resolution_step = 0.05
coarse_high_resolution_min = 1.05
coarse_high_resolution_max = 1.20
coarse_high_resolution_step = 0.10
bridge_resolution_step = 0.001
local_resolution_window = 0.02
local_resolution_step = 0.0005
local_resolution_candidate_count = 8
resolution_preference_center = 1.0

pred_path = (
    project_root
    / "SpaDTA_718"
    / "runs"
    / "first"
    / sample_name
    / f"{sample_name}_output.h5ad"
)
gt_path = (
    Path("/bigdat2/user/hesy/spatialmeta/SpatialMETA/06_spatialmeta_groundtruth/06_spatialmeta_groundtruth")
    / f"adata_joint_{sample_name}_hvf2800.h5ad"
)
output_dir = pred_path.parent
trace_csv_path = output_dir / f"{sample_name}_resolution_scan_trace.csv"
metrics_csv_path = output_dir / f"{sample_name}_targetcount_metrics.csv"
target_h5ad_path = output_dir / f"{sample_name}_targetcount.h5ad"


def limit_cpu_cores():
    available_cores = sorted(os.sched_getaffinity(0))
    if len(available_cores) <= cpu_core_limit:
        selected_cores = available_cores
    else:
        selected_cores = available_cores[:cpu_core_limit]
    os.sched_setaffinity(0, selected_cores)
    return selected_cores


def build_resolution_grid():
    coarse_low = np.arange(
        coarse_low_resolution_min,
        coarse_low_resolution_max + coarse_low_resolution_step * 0.1,
        coarse_low_resolution_step,
    )
    coarse_high = np.arange(
        coarse_high_resolution_min,
        coarse_high_resolution_max + coarse_high_resolution_step * 0.1,
        coarse_high_resolution_step,
    )
    return np.unique(
        np.round(
            np.concatenate([coarse_low, coarse_high, np.array([resolution_preference_center])]),
            5,
        )
    )


def compute_metrics(y_true, y_pred):
    ari = adjusted_rand_score(y_true.astype(str), y_pred.astype(str))
    nmi = normalized_mutual_info_score(y_true.astype(str), y_pred.astype(str))
    return float(ari), float(nmi)


def prepare_eval_table(adata_pred, adata_gt):
    gt_key = "pathological_annotation" if "pathological_annotation" in adata_gt.obs.columns else "annotation"
    gt_spatial = np.asarray(adata_gt.obsm["spatial"], dtype=np.float32)
    pred_spatial = np.asarray(adata_pred.obsm["spatial"], dtype=np.float32)

    tree = cKDTree(gt_spatial)
    distances, indices = tree.query(pred_spatial, k=1)
    matched_mask = distances < spatial_match_threshold

    if int(np.sum(matched_mask)) <= 10:
        raise RuntimeError("spatial matching failed: too few matched spots")

    matched_obs = adata_pred.obs.iloc[np.where(matched_mask)[0]].copy()
    matched_obs["gt_label"] = adata_gt.obs[gt_key].astype(str).to_numpy()[indices[matched_mask]]
    target_cluster_count = int(pd.Series(adata_gt.obs[gt_key].astype(str)).nunique())
    return matched_mask, matched_obs, target_cluster_count, gt_key


def evaluate_resolution(adata_pred, matched_mask, gt_labels, resolution):
    key_name = f"_scan_res_{str(float(resolution)).replace('.', '_')}"
    sc.tl.leiden(
        adata_pred,
        resolution=float(resolution),
        key_added=key_name,
        random_state=cluster_random_seed,
    )
    pred_labels = adata_pred.obs[key_name].astype(str)
    matched_pred_labels = pred_labels.iloc[np.where(matched_mask)[0]].copy()
    cluster_count = int(matched_pred_labels.nunique())
    ari, nmi = compute_metrics(gt_labels, matched_pred_labels)
    return key_name, cluster_count, ari, nmi, matched_pred_labels


def collect_bridge_resolutions(trace_df, target_cluster_count):
    candidate_values = []
    trace_sorted = trace_df.sort_values("resolution").reset_index(drop=True)
    for left_row, right_row in zip(trace_sorted.iloc[:-1].itertuples(), trace_sorted.iloc[1:].itertuples()):
        if int(left_row.pred_clusters) == int(right_row.pred_clusters):
            continue
        lower_count = min(int(left_row.pred_clusters), int(right_row.pred_clusters))
        upper_count = max(int(left_row.pred_clusters), int(right_row.pred_clusters))
        if not (lower_count <= target_cluster_count <= upper_count):
            continue
        candidate_values.extend(
            np.arange(
                float(left_row.resolution),
                float(right_row.resolution) + bridge_resolution_step * 0.1,
                bridge_resolution_step,
            ).tolist()
        )
    return np.unique(np.round(np.asarray(candidate_values, dtype=np.float64), 5))


def collect_local_resolutions(trace_df, target_cluster_count):
    nearest_rows = trace_df.iloc[
        (trace_df["pred_clusters"] - target_cluster_count).abs().argsort()[:local_resolution_candidate_count]
    ]
    candidate_values = []
    for resolution in nearest_rows["resolution"].astype(float).tolist():
        candidate_values.extend(
            np.arange(
                max(0.001, resolution - local_resolution_window),
                resolution + local_resolution_window + local_resolution_step * 0.1,
                local_resolution_step,
            ).tolist()
        )
    return np.unique(np.round(np.asarray(candidate_values, dtype=np.float64), 5))


def main():
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    if not gt_path.exists():
        raise FileNotFoundError(gt_path)

    selected_cores = limit_cpu_cores()

    print(f"[scan] sample={sample_name}", flush=True)
    print(f"[scan] pred_path={pred_path}", flush=True)
    print(f"[scan] gt_path={gt_path}", flush=True)
    print(f"[scan] cpu_core_limit={cpu_core_limit}", flush=True)
    print(f"[scan] cpu_core_count_in_use={len(selected_cores)}", flush=True)
    print(
        "[scan] resolution_grid="
        f"low[{coarse_low_resolution_min}, {coarse_low_resolution_max}, step={coarse_low_resolution_step}] "
        f"high[{coarse_high_resolution_min}, {coarse_high_resolution_max}, step={coarse_high_resolution_step}] "
        f"bridge_step={bridge_resolution_step} "
        f"local_window={local_resolution_window} "
        f"local_step={local_resolution_step} "
        f"local_candidates={local_resolution_candidate_count} "
        f"preference_center={resolution_preference_center}",
        flush=True,
    )

    print("[scan] reading prediction h5ad", flush=True)
    adata_pred = sc.read_h5ad(pred_path)
    print("[scan] reading ground-truth h5ad", flush=True)
    adata_gt = sc.read_h5ad(gt_path)

    if embedding_key not in adata_pred.obsm:
        raise KeyError(f"missing embedding key: {embedding_key}")
    if "spatial" not in adata_pred.obsm or "spatial" not in adata_gt.obsm:
        raise KeyError("missing spatial coordinates")

    print("[scan] building neighbor graph", flush=True)
    sc.pp.neighbors(
        adata_pred,
        use_rep=embedding_key,
        n_neighbors=neighbor_count,
        random_state=cluster_random_seed,
    )
    print("[scan] neighbor graph ready", flush=True)

    matched_mask, matched_obs, target_cluster_count, gt_key = prepare_eval_table(adata_pred, adata_gt)
    gt_labels = matched_obs["gt_label"].astype(str)
    matched_spot_count = int(matched_mask.sum())

    print(f"[scan] matched_spots={matched_spot_count}", flush=True)
    print(f"[scan] gt_key={gt_key}", flush=True)
    print(f"[scan] target_cluster_count={target_cluster_count}", flush=True)

    rows = []
    best_exact = None
    scanned_resolutions = set()

    def scan_resolutions(resolution_values):
        nonlocal best_exact
        for resolution in resolution_values:
            resolution = round(float(resolution), 5)
            if resolution in scanned_resolutions:
                continue
            key_name, cluster_count, ari, nmi, matched_pred_labels = evaluate_resolution(
                adata_pred=adata_pred,
                matched_mask=matched_mask,
                gt_labels=gt_labels,
                resolution=resolution,
            )
            rows.append(
                {
                    "sample": sample_name,
                    "resolution": resolution,
                    "pred_clusters": cluster_count,
                    "matched_spots": matched_spot_count,
                    "ARI": ari,
                    "NMI": nmi,
                    "is_target_count": int(cluster_count == target_cluster_count),
                    "cluster_key": key_name,
                }
            )
            scanned_resolutions.add(resolution)
            print(
                f"[scan] resolution={resolution:.5f} pred_clusters={cluster_count} ARI={ari:.6f} NMI={nmi:.6f}",
                flush=True,
            )
            if cluster_count == target_cluster_count:
                candidate = {
                    "resolution": resolution,
                    "cluster_key": key_name,
                    "pred_labels": matched_pred_labels.copy(),
                    "ari": ari,
                    "nmi": nmi,
                }
                if best_exact is None:
                    best_exact = candidate
                else:
                    current_gap = abs(best_exact["resolution"] - resolution_preference_center)
                    candidate_gap = abs(resolution - resolution_preference_center)
                    if candidate_gap < current_gap or (
                        candidate_gap == current_gap and resolution < best_exact["resolution"]
                    ):
                        best_exact = candidate

    scan_resolutions(build_resolution_grid())

    trace_df = pd.DataFrame(rows)
    if best_exact is None and not trace_df.empty:
        bridge_resolutions = collect_bridge_resolutions(trace_df, target_cluster_count)
        if bridge_resolutions.size > 0:
            scan_resolutions(bridge_resolutions)
            trace_df = pd.DataFrame(rows)

    if best_exact is None and not trace_df.empty:
        local_resolutions = collect_local_resolutions(trace_df, target_cluster_count)
        if local_resolutions.size > 0:
            scan_resolutions(local_resolutions)
            trace_df = pd.DataFrame(rows)

    trace_df = pd.DataFrame(rows).sort_values("resolution").reset_index(drop=True)
    trace_df.to_csv(trace_csv_path, index=False)
    print(f"[scan] trace_csv={trace_csv_path}", flush=True)

    if best_exact is None:
        raise RuntimeError(f"no resolution reached target cluster count {target_cluster_count}")

    adata_pred.obs[cluster_key] = adata_pred.obs[best_exact["cluster_key"]].astype(str).astype("category")
    adata_pred.write_h5ad(target_h5ad_path)

    metrics_df = pd.DataFrame(
        [
            {
                "sample": sample_name,
                "pred_file": str(target_h5ad_path),
                "gt_file": str(gt_path),
                "matched_spots": matched_spot_count,
                "matching_mode": "spatial",
                "pred_key": cluster_key,
                "pred_clusters": target_cluster_count,
                "ARI": best_exact["ari"],
                "NMI": best_exact["nmi"],
                "cluster_resolution": best_exact["resolution"],
            }
        ]
    )
    metrics_df.to_csv(metrics_csv_path, index=False)

    print(metrics_df.to_string(index=False))
    print(f"[scan] metrics_csv={metrics_csv_path}", flush=True)
    print(f"[scan] target_h5ad={target_h5ad_path}", flush=True)


if __name__ == "__main__":
    main()
