from __future__ import annotations

from pathlib import Path

from spaDTA.model.workflow import run_parallel_jobs


package_root = Path(__file__).resolve().parents[1]
processed_root = Path("/bigdat2/user/hesy/spatialmeta/SpatialMETA/processed")
gt_root = Path("/bigdat2/user/hesy/spatialmeta/SpatialMETA/06_spatialmeta_groundtruth/06_spatialmeta_groundtruth")
output_root = package_root / "runs" / "model_runs" / "current_localcontext_contrastive_w002_fixed_targetcount_20260427_localized"
config_name = "ablate_no_hete_homo_splitrecon_sharedhalf_balance"
jobs = [
    {"sample_name": "248_T", "device": "cuda:1", "cluster_resolution": 0.96},
    {"sample_name": "R114_T", "device": "cuda:3", "cluster_resolution": 0.36},
    {"sample_name": "S15_T", "device": "cuda:7", "cluster_resolution": 0.56},
    {"sample_name": "X49_T", "device": "cuda:1", "cluster_resolution": 0.41},
    {"sample_name": "Y27_T", "device": "cuda:3", "cluster_resolution": 0.37},
    {"sample_name": "Y7_T", "device": "cuda:7", "cluster_resolution": 0.798},
    {"sample_name": "m1_FMP", "device": "cuda:1", "cluster_resolution": 0.56},
    {"sample_name": "m3_FMP", "device": "cuda:3", "cluster_resolution": 0.41},
    {"sample_name": "m4_FMP", "device": "cuda:7", "cluster_resolution": 0.46},
]

run_parallel_jobs(
    jobs=jobs,
    processed_root=processed_root,
    gt_root=gt_root,
    output_root=output_root,
    config_name=config_name,
    max_workers=3,
)
