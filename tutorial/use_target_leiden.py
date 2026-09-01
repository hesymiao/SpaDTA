from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.spatial import cKDTree

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from SpaDTA_718.model.cluster_eval_utils import (
    compute_metrics,
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
default_cluster_key = "cluster_target_count_leiden"
default_neighbor_count = 15
default_random_seed = 0
default_spatial_match_threshold = 5.0
default_cpu_core_limit = 30
default_resolution_lower = 0.0
default_resolution_upper = 1000.0
default_search_max_iterations = 100
default_cluster_count_tolerance = 0
default_mclust_fallback = True
default_mclust_pca_components = 20
default_mclust_model = "EEE"


def limit_cpu_cores(cpu_core_limit: int) -> list[int]:
    available_cores = sorted(os.sched_getaffinity(0))
    selected_cores = available_cores[:cpu_core_limit] if len(available_cores) > cpu_core_limit else available_cores
    os.sched_setaffinity(0, selected_cores)
    return selected_cores


def infer_gt_key(adata_gt: sc.AnnData) -> str:
    if "pathological_annotation" in adata_gt.obs.columns:
        return "pathological_annotation"
    if "annotation" in adata_gt.obs.columns:
        return "annotation"
    raise KeyError("ground-truth h5ad is missing both 'pathological_annotation' and 'annotation'")


def discover_input_files(run_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in run_dir.glob("*.h5ad")
        if not path.name.endswith("_targetleiden.h5ad") and not path.name.endswith("_mclust.h5ad")
    )


def choose_embedding_key(adata_pred: sc.AnnData, requested_key: str) -> str:
    if requested_key in adata_pred.obsm:
        return requested_key
    for key in ("X_q_mu_decalign_linear", "X_umap"):
        if key in adata_pred.obsm:
            return key
    raise KeyError(f"embedding key {requested_key!r} not found; obsm keys: {list(adata_pred.obsm.keys())}")


def prepare_eval_table(
    adata_pred: sc.AnnData,
    adata_gt: sc.AnnData,
    spatial_match_threshold: float,
) -> tuple[np.ndarray, pd.Series, int, str]:
    gt_key = infer_gt_key(adata_gt)
    gt_spatial = np.asarray(adata_gt.obsm["spatial"], dtype=np.float32)
    pred_spatial = np.asarray(adata_pred.obsm["spatial"], dtype=np.float32)
    distances, indices = cKDTree(gt_spatial).query(pred_spatial, k=1)
    matched_mask = distances < spatial_match_threshold
    gt_labels = adata_gt.obs[gt_key].astype(str).to_numpy()[indices[matched_mask]]
    target_cluster_count = int(pd.Series(adata_gt.obs[gt_key].astype(str)).nunique())
    return matched_mask, pd.Series(gt_labels), target_cluster_count, gt_key


def build_search_adata(
    adata_pred: sc.AnnData,
    *,
    embedding_key: str,
    neighbor_count: int,
) -> AnnData:
    adata_search = AnnData(np.asarray(adata_pred.obsm[embedding_key], dtype=np.float32))
    sc.pp.neighbors(
        adata_search,
        n_neighbors=neighbor_count,
        use_rep="X",
    )
    return adata_search


def evaluate_louvain_resolution(
    *,
    adata_search: AnnData,
    matched_mask: np.ndarray,
    gt_labels: pd.Series,
    resolution: float,
    cluster_random_seed: int,
) -> dict[str, object]:
    adata_clustered = sc.tl.louvain(
        adata_search,
        resolution=float(resolution),
        random_state=cluster_random_seed,
        copy=True,
    )
    pred_labels = adata_clustered.obs["louvain"].astype(str)
    matched_pred_labels = pred_labels.iloc[np.where(matched_mask)[0]].copy()
    metrics = compute_metrics(gt_labels.to_numpy(), matched_pred_labels.to_numpy())
    return {
        "resolution": float(resolution),
        "full_cluster_count": int(pred_labels.nunique()),
        "matched_cluster_count": int(matched_pred_labels.nunique()),
        "ari": float(metrics["ARI"]),
        "nmi": float(metrics["NMI"]),
        "matched_pred_labels": matched_pred_labels,
        "all_pred_labels": pred_labels,
    }


def find_resolution(
    *,
    adata_search: AnnData,
    matched_mask: np.ndarray,
    gt_labels: pd.Series,
    target_cluster_count: int,
    cluster_random_seed: int,
    cluster_count_tolerance: int = default_cluster_count_tolerance,
    max_iterations: int = default_search_max_iterations,
    resolution_lower: float = default_resolution_lower,
    resolution_upper: float = default_resolution_upper,
) -> tuple[dict[str, object], pd.DataFrame]:
    obtained_clusters = -1
    iteration = 0
    resolutions = [float(resolution_lower), float(resolution_upper)]
    rows: list[dict[str, object]] = []
    latest_result: dict[str, object] | None = None

    while obtained_clusters != target_cluster_count and iteration < max_iterations:
        current_res = sum(resolutions) / 2.0
        if current_res <= resolutions[0] or current_res >= resolutions[1]:
            break
        latest_result = evaluate_louvain_resolution(
            adata_search=adata_search,
            matched_mask=matched_mask,
            gt_labels=gt_labels,
            resolution=current_res,
            cluster_random_seed=cluster_random_seed,
        )
        obtained_clusters = int(latest_result["full_cluster_count"])
        rows.append(
            {
                "iteration": int(iteration),
                "resolution": float(current_res),
                "pred_clusters_full": int(latest_result["full_cluster_count"]),
                "pred_clusters_matched": int(latest_result["matched_cluster_count"]),
                "matched_spots": int(matched_mask.sum()),
                "ARI": float(latest_result["ari"]),
                "NMI": float(latest_result["nmi"]),
                "is_target_count_full": int(obtained_clusters == target_cluster_count),
                "is_target_count_matched": int(int(latest_result["matched_cluster_count"]) == target_cluster_count),
                "search_low": float(resolutions[0]),
                "search_high": float(resolutions[1]),
            }
        )

        if target_cluster_count - obtained_clusters > cluster_count_tolerance:
            resolutions[0] = current_res
        elif obtained_clusters - target_cluster_count > cluster_count_tolerance:
            resolutions[1] = current_res

        iteration += 1

    if latest_result is None:
        raise RuntimeError("resolution search did not run")
    if iteration == max_iterations and obtained_clusters != target_cluster_count:
        print("!!! Hard !!!", flush=True)

    latest_result["iterations"] = int(iteration)
    latest_result["found_exact_target"] = bool(obtained_clusters == target_cluster_count)
    latest_result["target_cluster_count"] = int(target_cluster_count)
    return latest_result, pd.DataFrame(rows)


def process_single_file(
    *,
    input_h5ad: Path | None = None,
    adata_pred: AnnData | None = None,
    gt_h5ad: Path,
    embedding_key: str,
    cluster_key: str,
    neighbor_count: int,
    cluster_random_seed: int,
    spatial_match_threshold: float,
    output_prefix: Path | None = None,
    write_clustered_h5ad: bool = True,
    write_trace_csv: bool = True,
    write_metrics_csv: bool = True,
    mclust_fallback: bool = default_mclust_fallback,
    mclust_pca_components: int = default_mclust_pca_components,
    mclust_model: str = default_mclust_model,
    rscript: Path | None = None,
) -> dict[str, object]:
    if adata_pred is None:
        if input_h5ad is None:
            raise ValueError("either input_h5ad or adata_pred must be provided")
        adata_pred = sc.read_h5ad(input_h5ad)
    elif input_h5ad is not None:
        raise ValueError("provide either input_h5ad or adata_pred, not both")
    adata_gt = sc.read_h5ad(gt_h5ad)
    effective_embedding_key = choose_embedding_key(adata_pred, embedding_key)
    adata_search = build_search_adata(
        adata_pred,
        embedding_key=effective_embedding_key,
        neighbor_count=neighbor_count,
    )
    matched_mask, gt_labels, target_cluster_count, gt_key = prepare_eval_table(
        adata_pred,
        adata_gt,
        spatial_match_threshold=spatial_match_threshold,
    )
    matched_spot_count = int(matched_mask.sum())
    best_result, trace_df = find_resolution(
        adata_search=adata_search,
        matched_mask=matched_mask,
        gt_labels=gt_labels,
        target_cluster_count=target_cluster_count,
        cluster_random_seed=cluster_random_seed,
    )

    clustering_method = "target_count_louvain_binary_search"
    if not best_result["found_exact_target"] and mclust_fallback:
        points = np.asarray(adata_pred.obsm[effective_embedding_key], dtype=np.float64)
        reduced = pca_project(points, int(mclust_pca_components))
        labels = run_mclust_fixed_k(
            reduced,
            n_clusters=target_cluster_count,
            random_seed=cluster_random_seed,
            rscript=resolve_rscript(rscript),
            work_dir=output_prefix.parent / "_mclust_tmp" if output_prefix is not None else Path.cwd() / "_mclust_tmp",
            model_name=str(mclust_model),
        ).astype(str)
        all_pred_labels = pd.Series(labels, index=adata_pred.obs_names)
        matched_pred_labels = all_pred_labels.iloc[np.where(matched_mask)[0]].copy()
        best_result.update(
            {
                "matched_pred_labels": matched_pred_labels,
                "all_pred_labels": all_pred_labels,
                "matched_cluster_count": int(matched_pred_labels.nunique()),
                "full_cluster_count": int(all_pred_labels.nunique()),
            }
        )
        clustering_method = "mclust_fixed_k_fallback"

    final_labels = best_result["matched_pred_labels"].astype(str).to_numpy()
    adata_eval = adata_pred[matched_mask].copy()
    adata_eval.obs[cluster_key] = pd.Categorical(final_labels)
    adata_eval.obs["GT"] = gt_labels.to_numpy()
    metrics = compute_metrics(adata_eval.obs["GT"].to_numpy(), adata_eval.obs[cluster_key].to_numpy())
    ari = float(metrics["ARI"])
    nmi = float(metrics["NMI"])

    if output_prefix is None:
        if input_h5ad is None:
            raise ValueError("output_prefix is required when evaluating an in-memory AnnData")
        output_prefix = input_h5ad.with_suffix("")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_h5ad = output_prefix.with_name(f"{output_prefix.name}_targetleiden.h5ad")
    trace_csv = output_prefix.with_name(f"{output_prefix.name}_targetleiden_trace.csv")
    metrics_csv = output_prefix.with_name(f"{output_prefix.name}_targetleiden_metrics.csv")

    adata_eval.uns[f"{cluster_key}_meta"] = {
        "method": clustering_method,
        "gt_h5ad": str(gt_h5ad),
        "gt_obs_key": gt_key,
        "embedding_key": effective_embedding_key,
        "n_neighbors": int(neighbor_count),
        "cluster_random_seed": int(cluster_random_seed),
        "spatial_match_threshold": float(spatial_match_threshold),
        "target_n_clusters": int(target_cluster_count),
        "selected_resolution": float(best_result["resolution"]),
        "search_max_iterations": int(default_search_max_iterations),
        "resolution_lower": float(default_resolution_lower),
        "resolution_upper": float(default_resolution_upper),
        "found_exact_target": bool(best_result["found_exact_target"]),
        "mclust_fallback_enabled": bool(mclust_fallback),
        "mclust_fallback_used": bool(clustering_method == "mclust_fixed_k_fallback"),
        "mclust_model": str(mclust_model),
        "mclust_pca_components": int(mclust_pca_components),
    }
    if write_clustered_h5ad:
        adata_eval.write_h5ad(output_h5ad)
    if write_trace_csv:
        trace_df.sort_values("iteration").to_csv(trace_csv, index=False)

    row = {
        "input_h5ad": None if input_h5ad is None else str(input_h5ad),
        "output_h5ad": str(output_h5ad) if write_clustered_h5ad else "",
        "trace_csv": str(trace_csv) if write_trace_csv else "",
        "metrics_csv": str(metrics_csv) if write_metrics_csv else "",
        "embedding_key": effective_embedding_key,
        "cluster_key": cluster_key,
        "target_n_clusters": int(target_cluster_count),
        "observed_pred_clusters": int(best_result["matched_cluster_count"]),
        "observed_pred_clusters_full": int(best_result["full_cluster_count"]),
        "matched_spots": int(matched_spot_count),
        "random_seed": int(cluster_random_seed),
        "n_neighbors": int(neighbor_count),
        "selected_resolution": float(best_result["resolution"]),
        "resolution_iterations": int(best_result["iterations"]),
        "found_exact_target": bool(best_result["found_exact_target"]),
        "clustering_method": clustering_method,
        "mclust_fallback_used": bool(clustering_method == "mclust_fixed_k_fallback"),
        "ARI": ari,
        "NMI": nmi,
    }
    if write_metrics_csv:
        pd.DataFrame([row]).to_csv(metrics_csv, index=False)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run target-count clustering with pasted-text Louvain binary-search logic on existing result h5ad files."
    )
    parser.add_argument("--run-dir", type=Path, default=default_run_dir)
    parser.add_argument("--input-h5ad", type=Path, nargs="*", default=None)
    parser.add_argument("--gt-h5ad", type=Path, default=default_gt_h5ad)
    parser.add_argument("--embedding-key", default=default_embedding_key)
    parser.add_argument("--cluster-key", default=default_cluster_key)
    parser.add_argument("--n-neighbors", type=int, default=default_neighbor_count)
    parser.add_argument("--random-seed", type=int, default=default_random_seed)
    parser.add_argument("--spatial-match-threshold", type=float, default=default_spatial_match_threshold)
    parser.add_argument("--cpu-core-limit", type=int, default=default_cpu_core_limit)
    parser.add_argument("--no-mclust-fallback", action="store_true")
    parser.add_argument("--mclust-pca-components", type=int, default=default_mclust_pca_components)
    parser.add_argument("--mclust-model", default=default_mclust_model)
    parser.add_argument("--rscript", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_cores = limit_cpu_cores(int(args.cpu_core_limit))
    print(f"[targetleiden] cpu_cores={selected_cores}", flush=True)
    if args.input_h5ad:
        input_files = [path.expanduser().resolve() for path in args.input_h5ad]
    else:
        input_files = discover_input_files(args.run_dir.expanduser().resolve())

    rows: list[dict[str, object]] = []
    for input_h5ad in input_files:
        row = process_single_file(
            input_h5ad=input_h5ad,
            gt_h5ad=args.gt_h5ad.expanduser().resolve(),
            embedding_key=str(args.embedding_key),
            cluster_key=str(args.cluster_key),
            neighbor_count=int(args.n_neighbors),
            cluster_random_seed=int(args.random_seed),
            spatial_match_threshold=float(args.spatial_match_threshold),
            mclust_fallback=not bool(args.no_mclust_fallback),
            mclust_pca_components=int(args.mclust_pca_components),
            mclust_model=str(args.mclust_model),
            rscript=args.rscript,
        )
        rows.append(row)
        print(
            f"[targetleiden] {input_h5ad.name}: res={row['selected_resolution']:.5f} "
            f"ARI={row['ARI']:.4f} NMI={row['NMI']:.4f} "
            f"full_k={row['observed_pred_clusters_full']} matched_k={row['observed_pred_clusters']}",
            flush=True,
        )

    summary_csv = args.run_dir.expanduser().resolve() / "targetleiden_batch_summary.csv"
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    print(f"[targetleiden] summary_csv={summary_csv}", flush=True)


if __name__ == "__main__":
    main()
