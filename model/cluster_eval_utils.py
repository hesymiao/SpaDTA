from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial import cKDTree
from scipy.special import gammaln


default_rscript = Path("/data/user/hesy/miniconda3/envs/renv/bin/Rscript")
gt_spatial_match_threshold = 5.0
min_valid_gt_matches = 10


def pca_project(matrix: np.ndarray, n_components: int) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"expected a 2D embedding matrix, got shape={values.shape}")
    centered = values - values.mean(axis=0, keepdims=True)
    max_components = min(centered.shape[0], centered.shape[1])
    if max_components < 1:
        raise ValueError("embedding matrix is empty")
    use_components = max(1, min(int(n_components), max_components))
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:use_components].T


def normalize_branch_mean_variance(branch: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(branch, dtype=np.float64)
    centered = values - values.mean(axis=0, keepdims=True)
    per_dimension_variance = np.var(centered, axis=0, ddof=0)
    mean_dimension_variance = float(per_dimension_variance.mean())
    if mean_dimension_variance <= eps:
        raise ValueError(f"Branch mean dimension variance is too small: {mean_dimension_variance}")
    scale = float(np.sqrt(mean_dimension_variance))
    return centered / scale


def build_branch_scaled_full(
    q_mu_shared: np.ndarray,
    q_mu_st: np.ndarray,
    q_mu_sm: np.ndarray,
    *,
    shared_weight: float = 1.0,
    st_weight: float = 1.0,
    sm_weight: float = 1.0,
) -> np.ndarray:
    return np.concatenate(
        [
            float(shared_weight) * normalize_branch_mean_variance(q_mu_shared),
            float(st_weight) * normalize_branch_mean_variance(q_mu_st),
            float(sm_weight) * normalize_branch_mean_variance(q_mu_sm),
        ],
        axis=1,
    )


def summarize_branch_variance(branch: np.ndarray) -> dict[str, float]:
    values = np.asarray(branch, dtype=np.float64)
    centered = values - values.mean(axis=0, keepdims=True)
    per_dimension_variance = np.var(centered, axis=0, ddof=0)
    return {
        "total_variance": float(per_dimension_variance.sum()),
        "mean_dimension_variance": float(per_dimension_variance.mean()),
        "mean_vector_norm": float(np.linalg.norm(centered, axis=1).mean()),
    }


def load_gt_from_h5ad(
    gt_h5ad: Path,
    pred_obs_names: pd.Index,
    pred_coords: np.ndarray,
    gt_key: str | None,
) -> pd.Series:
    gt_adata = sc.read_h5ad(gt_h5ad)
    if gt_key is None:
        if "pathological_annotation" in gt_adata.obs.columns:
            gt_key = "pathological_annotation"
        elif "annotation" in gt_adata.obs.columns:
            gt_key = "annotation"
        else:
            raise KeyError("ground-truth h5ad is missing 'pathological_annotation' and 'annotation'")
    if gt_key not in gt_adata.obs.columns:
        raise KeyError(f"ground-truth key {gt_key!r} not found in {gt_h5ad}")

    gt_labels = gt_adata.obs[gt_key].astype(object)
    if set(pred_obs_names) == set(gt_adata.obs_names):
        aligned = gt_labels.loc[pred_obs_names].copy()
        aligned.index = pred_obs_names
        return aligned

    if "spatial" not in gt_adata.obsm:
        raise KeyError("ground-truth h5ad is missing obsm['spatial'] for coordinate matching")

    gt_coords = np.asarray(gt_adata.obsm["spatial"], dtype=np.float64)[:, :2]
    distances, indices = cKDTree(gt_coords).query(pred_coords, k=1)
    matched_mask = distances < gt_spatial_match_threshold
    matched_count = int(np.sum(matched_mask))
    if matched_count <= min_valid_gt_matches:
        raise RuntimeError(
            "prediction spots could not be sufficiently aligned to ground truth by coordinates; "
            f"matched_spots={matched_count}, threshold={gt_spatial_match_threshold:.6g}"
        )

    aligned = pd.Series(pd.NA, index=pred_obs_names, dtype="object")
    aligned.iloc[np.where(matched_mask)[0]] = gt_labels.iloc[indices[matched_mask]].to_numpy()
    return aligned


def load_gt_from_annotation_csv(annotation_csv: Path, pred_obs_names: pd.Index) -> pd.Series:
    table = pd.read_csv(annotation_csv, index_col=0)
    if "manual-anno" not in table.columns:
        raise KeyError(f"'manual-anno' column not found in {annotation_csv}")
    if not set(pred_obs_names).issubset(set(table.index)):
        missing = [name for name in pred_obs_names if name not in table.index][:5]
        raise KeyError(f"prediction barcodes are missing from annotation csv, examples: {missing}")
    aligned = table.loc[pred_obs_names, "manual-anno"].astype(object)
    aligned.index = pred_obs_names
    return aligned


def factorize_pair(labels_true: np.ndarray, labels_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    true_codes, true_uniques = pd.factorize(labels_true.astype(str), sort=True)
    pred_codes, pred_uniques = pd.factorize(labels_pred.astype(str), sort=True)
    contingency = np.zeros((true_uniques.shape[0], pred_uniques.shape[0]), dtype=np.int64)
    np.add.at(contingency, (true_codes, pred_codes), 1)
    return contingency, np.asarray(true_uniques), np.asarray(pred_uniques)


def comb2(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64, copy=False)
    return values * (values - 1.0) / 2.0


def entropy(counts: np.ndarray) -> float:
    counts = counts.astype(np.float64, copy=False)
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    probs = counts[counts > 0] / total
    return float(-np.sum(probs * np.log(probs)))


def mutual_information(contingency: np.ndarray) -> float:
    contingency = contingency.astype(np.float64, copy=False)
    total = contingency.sum()
    if total <= 0:
        return 0.0
    row_sums = contingency.sum(axis=1)
    col_sums = contingency.sum(axis=0)
    nz = contingency > 0
    nij = contingency[nz]
    rows, cols = np.nonzero(nz)
    terms = (nij / total) * np.log((total * nij) / (row_sums[rows] * col_sums[cols]))
    return float(np.sum(terms))


def expected_mutual_information(contingency: np.ndarray) -> float:
    contingency = contingency.astype(np.int64, copy=False)
    total = int(contingency.sum())
    if total <= 1:
        return 0.0

    row_sums = contingency.sum(axis=1).astype(np.int64, copy=False)
    col_sums = contingency.sum(axis=0).astype(np.int64, copy=False)
    log_total_choose = gammaln(total + 1.0)
    emi = 0.0

    for a_i in row_sums:
        if a_i == 0:
            continue
        log_a = gammaln(a_i + 1.0)
        log_total_minus_a = gammaln(total - a_i + 1.0)
        for b_j in col_sums:
            if b_j == 0:
                continue
            start = max(1, a_i + b_j - total)
            end = min(a_i, b_j)
            if end < start:
                continue
            log_choose_b = gammaln(b_j + 1.0) + gammaln(total - b_j + 1.0)
            for n_ij in range(start, end + 1):
                log_prob = (
                    log_a
                    - gammaln(n_ij + 1.0)
                    - gammaln(a_i - n_ij + 1.0)
                    + log_total_minus_a
                    - gammaln(b_j - n_ij + 1.0)
                    - gammaln(total - a_i - b_j + n_ij + 1.0)
                    - log_total_choose
                    + log_choose_b
                )
                prob = float(np.exp(log_prob))
                term = (n_ij / total) * np.log((total * n_ij) / (a_i * b_j))
                emi += prob * term
    return float(emi)


def adjusted_rand_index(contingency: np.ndarray) -> float:
    n = float(contingency.sum())
    if n <= 1:
        return 1.0
    row_sums = contingency.sum(axis=1)
    col_sums = contingency.sum(axis=0)
    sum_comb = float(comb2(contingency).sum())
    sum_rows = float(comb2(row_sums).sum())
    sum_cols = float(comb2(col_sums).sum())
    total_pairs = n * (n - 1.0) / 2.0
    expected = (sum_rows * sum_cols) / total_pairs if total_pairs > 0 else 0.0
    max_index = 0.5 * (sum_rows + sum_cols)
    denominator = max_index - expected
    if abs(denominator) < 1e-15:
        return 1.0
    return float((sum_comb - expected) / denominator)


def fowlkes_mallows_index(contingency: np.ndarray) -> float:
    row_sums = contingency.sum(axis=1)
    col_sums = contingency.sum(axis=0)
    tp = float(comb2(contingency).sum())
    fp = float(comb2(col_sums).sum()) - tp
    fn = float(comb2(row_sums).sum()) - tp
    denom = np.sqrt((tp + fp) * (tp + fn))
    if denom <= 0:
        return 0.0
    return float(tp / denom)


def compute_metrics(labels_true: np.ndarray, labels_pred: np.ndarray) -> dict[str, float]:
    contingency, _, _ = factorize_pair(labels_true, labels_pred)
    row_sums = contingency.sum(axis=1)
    col_sums = contingency.sum(axis=0)
    h_true = entropy(row_sums)
    h_pred = entropy(col_sums)
    mi = mutual_information(contingency)
    emi = expected_mutual_information(contingency)

    nmi_den = h_true + h_pred
    nmi = 0.0 if nmi_den <= 0 else float(2.0 * mi / nmi_den)
    homo = 1.0 if h_true <= 0 else float(mi / h_true)
    complete = 1.0 if h_pred <= 0 else float(mi / h_pred)
    v_measure = 0.0 if (homo + complete) <= 0 else float(2.0 * homo * complete / (homo + complete))
    ami_den = 0.5 * (h_true + h_pred) - emi
    ami = 1.0 if abs(ami_den) < 1e-15 else float((mi - emi) / ami_den)

    return {
        "ARI": adjusted_rand_index(contingency),
        "NMI": nmi,
        "AMI": ami,
        "Homo": homo,
        "V-Measure": v_measure,
        "FMI": fowlkes_mallows_index(contingency),
        "MI": float(mi),
    }


def resolve_rscript(rscript: Path | None) -> Path:
    candidates: list[Path] = []
    if rscript is not None:
        candidates.append(Path(rscript))
    candidates.extend(
        [
            default_rscript,
            Path("/data/user/hesy/miniconda3/envs/stabmap_official_r42/bin/Rscript"),
            Path("/data/user/hesy/miniconda3/envs/spanjy/bin/Rscript"),
            Path("/data/user/hesy/miniconda3/envs/unitcr/bin/Rscript"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Rscript not found; pass --rscript explicitly")


def run_mclust_fixed_k(
    points: np.ndarray,
    n_clusters: int,
    random_seed: int,
    rscript: Path,
    work_dir: Path,
    model_name: str,
) -> np.ndarray:
    if points.shape[0] < n_clusters:
        raise ValueError("number of clusters exceeds number of observations")

    work_dir.mkdir(parents=True, exist_ok=True)
    csv_handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".csv",
        prefix="mclust_input_",
        dir=work_dir,
        delete=False,
    )
    out_handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".csv",
        prefix="mclust_output_",
        dir=work_dir,
        delete=False,
    )
    script_handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".R",
        prefix="mclust_run_",
        dir=work_dir,
        delete=False,
    )
    csv_path = Path(csv_handle.name)
    out_path = Path(out_handle.name)
    script_path = Path(script_handle.name)
    csv_handle.close()
    out_handle.close()
    script_handle.close()

    try:
        pd.DataFrame(points).to_csv(csv_path, index=False)
        script_path.write_text(
            """
args <- commandArgs(trailingOnly=TRUE)
if (length(args) < 5) {
  stop("expected 5 trailing arguments for mclust invocation")
}
input_csv <- args[1]
output_csv <- args[2]
num_cluster <- as.integer(args[3])
model_name <- args[4]
seed_value <- as.integer(args[5])
suppressPackageStartupMessages(library(mclust))
set.seed(seed_value)
dat <- read.csv(input_csv, check.names=FALSE)
if (nrow(dat) < num_cluster) {
  stop("number of rows is smaller than requested clusters")
}
res <- Mclust(dat, G=num_cluster, modelNames=model_name)
if (is.null(res$classification) || length(res$classification) == 0) {
  stop("mclust returned empty classification")
}
out_df <- data.frame(cluster=as.integer(res$classification))
write.csv(out_df, file=output_csv, row.names=FALSE, quote=FALSE)
""",
            encoding="utf-8",
        )
        completed = subprocess.run(
            [
                os.fspath(rscript),
                os.fspath(script_path),
                os.fspath(csv_path),
                os.fspath(out_path),
                str(int(n_clusters)),
                str(model_name),
                str(int(random_seed)),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if not out_path.exists() or out_path.stat().st_size == 0:
            stderr = completed.stderr.strip() if completed.stderr else ""
            stdout = completed.stdout.strip() if completed.stdout else ""
            raise RuntimeError(
                "mclust finished without writing a usable output file; "
                f"stdout={stdout!r} stderr={stderr!r}"
            )
        result = pd.read_csv(out_path)
        if "cluster" not in result.columns:
            raise RuntimeError("mclust output is missing 'cluster' column")
        labels = result["cluster"].astype(int).to_numpy()
        if labels.shape[0] != points.shape[0]:
            raise RuntimeError("mclust returned a different number of labels than input rows")
        return labels
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ""
        stdout = exc.stdout.strip() if exc.stdout else ""
        raise RuntimeError(f"mclust failed via {rscript}: {stderr or stdout or exc}") from exc
    finally:
        for path in (csv_path, out_path, script_path):
            if path.exists():
                path.unlink()
