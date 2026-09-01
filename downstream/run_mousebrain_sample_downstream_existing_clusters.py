from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from SpaDTA_718.downstream.workflow import run_downstream


package_root = Path(__file__).resolve().parents[1]
runs_dir = package_root / "runs"
sample_name = "Y27_T"
result_h5ad = runs_dir / sample_name / f"{sample_name}_ours_domains.h5ad"
gt_h5ad = Path(f"/bigdat2/user/hesy/spatialmeta/SpatialMETA/06_spatialmeta_groundtruth/06_spatialmeta_groundtruth/adata_joint_{sample_name}_hvf2800.h5ad")
output_dir = runs_dir / sample_name / "downstream_existing_clusters"
loss_csv = None
full_metrics_csv = None
default_metrics_csv = None
defaultcluster_h5ad = result_h5ad
cluster_key = "decalign_linear_clusters"
recluster_key = "decalign_linear_clusters"
embedding_key = "X_emb_decalign_linear"
reconstruction_layer = "reconstruction_decalign_linear"
normalized_layer = "normalized"
contribution_st_key = "contribution_st_decalign_linear"
contribution_sm_key = "contribution_sm_decalign_linear"
recluster_n_neighbors = 15
recluster_resolution = 1.0
recluster_random_seed = 0
spatial_match_threshold = 5.0
min_valid_spatial_matches = 10
title = f"{sample_name} downstream plots using model clusters"
clean_output = False


def run_mousebrain_sample_downstream_existing_clusters(
    sample_name: str,
    result_h5ad: Path,
    gt_h5ad: Path,
    output_dir: Path,
    loss_csv,
    full_metrics_csv,
    default_metrics_csv,
    defaultcluster_h5ad: Path,
    cluster_key: str,
    recluster_key: str,
    embedding_key: str,
    reconstruction_layer: str,
    normalized_layer: str,
    contribution_st_key: str,
    contribution_sm_key: str,
    recluster_n_neighbors: int,
    recluster_resolution: float,
    recluster_random_seed: int,
    spatial_match_threshold: float,
    min_valid_spatial_matches: int,
    title: str,
    clean_output: bool,
):
    return run_downstream(
        sample_name=sample_name,
        result_h5ad=result_h5ad,
        gt_h5ad=gt_h5ad,
        output_dir=output_dir,
        loss_csv=loss_csv,
        full_metrics_csv=full_metrics_csv,
        default_metrics_csv=default_metrics_csv,
        defaultcluster_h5ad=defaultcluster_h5ad,
        cluster_key=cluster_key,
        recluster_key=recluster_key,
        embedding_key=embedding_key,
        reconstruction_layer=reconstruction_layer,
        normalized_layer=normalized_layer,
        contribution_st_key=contribution_st_key,
        contribution_sm_key=contribution_sm_key,
        recluster_n_neighbors=recluster_n_neighbors,
        recluster_resolution=recluster_resolution,
        recluster_random_seed=recluster_random_seed,
        spatial_match_threshold=spatial_match_threshold,
        min_valid_spatial_matches=min_valid_spatial_matches,
        title=title,
        clean_output=clean_output,
    )


if __name__ == "__main__":
    run_mousebrain_sample_downstream_existing_clusters(
        sample_name=sample_name,
        result_h5ad=result_h5ad,
        gt_h5ad=gt_h5ad,
        output_dir=output_dir,
        loss_csv=loss_csv,
        full_metrics_csv=full_metrics_csv,
        default_metrics_csv=default_metrics_csv,
        defaultcluster_h5ad=defaultcluster_h5ad,
        cluster_key=cluster_key,
        recluster_key=recluster_key,
        embedding_key=embedding_key,
        reconstruction_layer=reconstruction_layer,
        normalized_layer=normalized_layer,
        contribution_st_key=contribution_st_key,
        contribution_sm_key=contribution_sm_key,
        recluster_n_neighbors=recluster_n_neighbors,
        recluster_resolution=recluster_resolution,
        recluster_random_seed=recluster_random_seed,
        spatial_match_threshold=spatial_match_threshold,
        min_valid_spatial_matches=min_valid_spatial_matches,
        title=title,
        clean_output=clean_output,
    )
