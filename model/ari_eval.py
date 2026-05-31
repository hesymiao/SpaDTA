from __future__ import annotations

from pathlib import Path

from spaDTA.model.workflow import evaluate_clustering


package_root = Path(__file__).resolve().parents[1]
sample_name = "248_T"
gt_path = Path("/bigdat2/user/hesy/spatialmeta/SpatialMETA/06_spatialmeta_groundtruth/06_spatialmeta_groundtruth") / f"adata_joint_{sample_name}_hvf2800.h5ad"
pred_path = package_root / "runs" / "manual_example" / f"{sample_name}_manual_example.h5ad"
output_csv_path = pred_path.parent / f"{sample_name}_metrics_full.csv"

evaluate_clustering(
    sample_name=sample_name,
    gt_path=gt_path,
    pred_path=pred_path,
    output_csv_path=output_csv_path,
    pred_key="decalign_linear_clusters",
    light_eval=True,
)
