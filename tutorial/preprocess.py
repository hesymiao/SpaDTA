from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from spaDTA.model.preprocess import preprocess_sm_st_to_h5ad

sm_input_path = Path(
    "/bigdat2/user/hesy/spatialmeta/SpatialMETA/mouse_brain/sma/V11L12-109/V11L12-109_B1/output_data/V11L12-109_B1_MSI/V11L12-109_B1.Visium.FMP.220826_smamsi.csv"
)
st_input_path = Path(
    "/bigdat2/user/hesy/spatialmeta/SpatialMETA/mouse_brain/sma/V11L12-109/V11L12-109_B1/output_data/V11L12-109_B1_RNA/outs"
)
output_path = Path(
    "/data/user/hesy/projects/SpatialMETA/spaDTA/tutorial/preprocess_test_output/m3_FMP.h5ad"
)

device = "cuda:1"
random_seed = 42
min_total_intensity_raw = 0.0
min_total_intensity_reassign = 0.0
min_counts = 0
min_genes = 0
align_st_top = 2000
align_sm_top = 500
align_max_epoch = 128
align_n_latent = 10
rotation_degrees = -90.0
n_neighbors = 5
dist_fold = 1.5
joint_top_genes = 2000
joint_top_metabolites = 800

print(f"[entry] sm_input_path={sm_input_path}", flush=True)
print(f"[entry] st_input_path={st_input_path}", flush=True)
print(f"[entry] output_path={output_path}", flush=True)
print(f"[entry] device={device}", flush=True)

adata = preprocess_sm_st_to_h5ad(
    sm_input_path=sm_input_path,
    st_input_path=st_input_path,
    output_path=output_path,
    device=device,
    random_seed=random_seed,
    min_total_intensity_raw=min_total_intensity_raw,
    min_total_intensity_reassign=min_total_intensity_reassign,
    min_counts=min_counts,
    min_genes=min_genes,
    align_st_top=align_st_top,
    align_sm_top=align_sm_top,
    align_max_epoch=align_max_epoch,
    align_n_latent=align_n_latent,
    rotation_degrees=rotation_degrees,
    n_neighbors=n_neighbors,
    dist_fold=dist_fold,
    joint_top_genes=joint_top_genes,
    joint_top_metabolites=joint_top_metabolites,
)

print(f"[done] shape={adata.shape}", flush=True)
print(f"[done] output_path={output_path}", flush=True)
