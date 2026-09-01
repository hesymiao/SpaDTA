from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from SpaDTA_718.downstream.workflow import run_downstream_for_samples


package_root = Path(__file__).resolve().parents[1]
run_root = package_root / "runs" / "model_runs" / "current_localcontext_contrastive_w002_fixed_targetcount_20260427_localized"
output_root = package_root / "runs" / "downstream_runs" / "current_localcontext_contrastive_w002_fixed_targetcount_20260427_localized"
gt_root = Path("/bigdat2/user/hesy/spatialmeta/SpatialMETA/06_spatialmeta_groundtruth/06_spatialmeta_groundtruth")
processed_root = Path("/bigdat2/user/hesy/spatialmeta/SpatialMETA/processed")
config_name = "ablate_no_hete_homo_splitrecon_sharedhalf_balance"
sample_names = [
    "248_T",
    "R114_T",
    "S15_T",
    "X49_T",
    "Y27_T",
    "Y7_T",
    "m1_FMP",
    "m3_FMP",
    "m4_FMP",
]
clean_output = False
worker_count = 3


def run_downstream_for_run_existing_clusters(
    sample_names: list[str],
    run_root: Path,
    output_root: Path,
    config_name: str,
    gt_root: Path,
    processed_root: Path,
    clean_output: bool = False,
    worker_count: int = 3,
):
    return run_downstream_for_samples(
        sample_names=sample_names,
        run_root=run_root,
        output_root=output_root,
        config_name=config_name,
        gt_root=gt_root,
        processed_root=processed_root,
        clean_output=clean_output,
        worker_count=worker_count,
    )


if __name__ == "__main__":
    run_downstream_for_run_existing_clusters(
        sample_names=sample_names,
        run_root=run_root,
        output_root=output_root,
        config_name=config_name,
        gt_root=gt_root,
        processed_root=processed_root,
        clean_output=clean_output,
        worker_count=worker_count,
    )
