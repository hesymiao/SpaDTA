from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    homogeneity_score,
    mutual_info_score,
    normalized_mutual_info_score,
    v_measure_score,
)


PROJECT_ROOT = Path("/data/user/hesy/projects/SpatialMETA")
RESULT_ROOT = PROJECT_ROOT / "SpaDTA_718/runs/atac_result"
SPADTA_RUN_ROOT = PROJECT_ROOT / "SpaDTA_718/runs/ATAC"
DATA_ROOT = Path("/bigdat2/user/hesy/spatialmeta/SpatialMETA/smart/SMART_data")
OUTPUT_DIR = PROJECT_ROOT / "SpaDTA_718/runs/atac_downstream/fig4a_metrics"

SAMPLES = [
    "Mouse_Brain_E11_S1",
    "Mouse_Brain_E13_S1",
    "Mouse_Brain_E15_S1",
    "Mouse_Brain_E18_S1",
]
METHODS = [
    "SpaDTA",
    "PRESENT",
    "SMART",
    "WNN",
    "MOFA+",
    "SNF",
    "CellCharter",
    "SpatialGlue",
    "MEFISTO",
    "MultiVI",
    "COSMOS",
    "scMM",
    "MISO",
]
METRICS = ["ARI", "NMI", "AMI", "Homo", "V-Measure", "FMI", "MI"]
NEW_METRICS = ["AMI", "Homo", "V-Measure", "FMI", "MI"]

MCLUST_SOURCES = {
    "PRESENT": "present_seed2020",
    "SMART": "smart_existing_uniform_mclust",
    "MOFA+": "mofa",
    "CellCharter": "uniform_mclust_recheck/cellcharter",
    "SpatialGlue": "spatialglue",
    "MEFISTO": "mefisto",
    "MultiVI": "multivi",
    "COSMOS": "uniform_mclust_recheck/cosmos",
    "scMM": "scmm",
    "MISO": "uniform_mclust_recheck/miso",
}


def load_truth(sample: str) -> pd.Series:
    annotation = pd.read_csv(DATA_ROOT / sample / "anno.csv", dtype=str)
    truth = annotation.set_index("barcode")["cluster"]
    truth.index = truth.index.astype(str)
    return truth.dropna()


def load_prediction(method: str, sample: str) -> pd.Series:
    if method == "SpaDTA":
        embedding_dir = SPADTA_RUN_ROOT / sample / "saved_epoch_embeddings/epoch_0300"
        label_dir = SPADTA_RUN_ROOT / sample / "final_protocol/epoch_0300"
        spot_ids = pd.read_csv(embedding_dir / "spot_ids.csv")["spot_id"].astype(str)
        labels = pd.read_csv(label_dir / "spot_labels.csv")["mclust_label"].astype(str)
    elif method in MCLUST_SOURCES:
        method_dir = RESULT_ROOT / MCLUST_SOURCES[method] / sample
        mclust_input = np.load(method_dir / "mclust_input.npz", allow_pickle=True)
        spot_ids = pd.Series(mclust_input["obs_names"].astype(str))
        labels = pd.Series(
            np.load(method_dir / "mclust_labels.npy", allow_pickle=True).astype(str)
        )
    elif method in {"WNN", "SNF"}:
        method_key = method.lower()
        adata = ad.read_h5ad(
            RESULT_ROOT / method_key / sample / f"adata_{method_key}.h5ad",
            backed="r",
        )
        try:
            spot_ids = pd.Series(adata.obs_names.astype(str))
            labels = adata.obs["paper_cluster"].astype(str).reset_index(drop=True)
        finally:
            adata.file.close()
    else:
        raise ValueError(f"Unsupported method: {method}")

    if len(spot_ids) != len(labels):
        raise ValueError(
            f"{method} {sample}: {len(spot_ids)} spot IDs but {len(labels)} labels"
        )
    prediction = pd.Series(labels.to_numpy(), index=spot_ids.to_numpy(), name="prediction")
    if prediction.index.has_duplicates:
        prediction = prediction.loc[~prediction.index.duplicated(keep="first")]
    return prediction


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    # Match the seven supervised metrics and sklearn implementations used by SMART.
    return {
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
        "AMI": float(adjusted_mutual_info_score(y_true, y_pred)),
        "Homo": float(homogeneity_score(y_true, y_pred)),
        "V-Measure": float(v_measure_score(y_true, y_pred)),
        "FMI": float(fowlkes_mallows_score(y_true, y_pred)),
        "MI": float(mutual_info_score(y_true, y_pred)),
    }


def evaluate_all() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in METHODS:
        for sample in SAMPLES:
            truth = load_truth(sample)
            prediction = load_prediction(method, sample)
            common = truth.index.intersection(prediction.index, sort=False)
            if common.empty:
                raise ValueError(f"{method} {sample}: no spots overlap ground truth")
            y_true = truth.loc[common].astype(str).to_numpy()
            y_pred = prediction.loc[common].astype(str).to_numpy()
            rows.append(
                {
                    "method": method,
                    "sample": sample,
                    "matched_spots": len(common),
                    "ground_truth_classes": int(pd.Series(y_true).nunique()),
                    "predicted_clusters": int(pd.Series(y_pred).nunique()),
                    **compute_metrics(y_true, y_pred),
                }
            )
    return pd.DataFrame(rows)


def validate_existing_metrics(detail: pd.DataFrame, output_dir: Path) -> None:
    for metric in ["ARI", "NMI"]:
        expected = pd.read_csv(output_dir / f"{metric}.csv").set_index("method")
        actual = detail.pivot(index="method", columns="sample", values=metric).reindex(
            index=METHODS, columns=SAMPLES
        )
        delta = np.abs(
            actual.to_numpy(dtype=float) - expected.loc[METHODS, SAMPLES].to_numpy(dtype=float)
        )
        max_delta = float(np.nanmax(delta))
        if max_delta > 5e-6:
            raise RuntimeError(
                f"Recomputed {metric} disagrees with the existing table; max delta={max_delta:.8g}"
            )
        print(f"Validated {metric} against existing table (max delta={max_delta:.3g})")


def write_outputs(detail: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for metric in NEW_METRICS:
        wide = detail.pivot(index="method", columns="sample", values=metric).reindex(
            index=METHODS, columns=SAMPLES
        )
        wide["mean"] = wide.mean(axis=1)
        wide.index.name = "method"
        path = output_dir / f"{metric}.csv"
        wide.to_csv(path, float_format="%.6f")
        outputs.append(path)

    per_sample_path = output_dir / "seven_metrics_per_sample.csv"
    detail.to_csv(per_sample_path, index=False, float_format="%.6f")
    outputs.append(per_sample_path)

    means = detail.groupby("method", as_index=False)[METRICS].mean()
    means["method"] = pd.Categorical(means["method"], categories=METHODS, ordered=True)
    means = means.sort_values("method").reset_index(drop=True)
    means["method"] = means["method"].astype(str)
    means_path = output_dir / "seven_metrics_mean.csv"
    means.to_csv(means_path, index=False, float_format="%.6f")
    outputs.append(means_path)

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate SMART Fig. d-style supervised metrics for the ATAC benchmark."
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detail = evaluate_all()
    validate_existing_metrics(detail, args.output_dir)
    outputs = write_outputs(detail, args.output_dir)
    print("Generated files:")
    for path in outputs:
        print(path)
    print("\nSeven-metric means:")
    print(pd.read_csv(args.output_dir / "seven_metrics_mean.csv").to_string(index=False))


if __name__ == "__main__":
    main()
