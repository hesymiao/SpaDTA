import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from SpaDTA_718.model.preprocess import prepare_spadta_model_input, preprocess_sm_st_to_h5ad


data_root = Path("/bigdat2/user/hesy/spatialmeta/SpatialMETA")
output_root = data_root / "SpaDTA_718_model_input" / "SM"
sample = "m3_FMP"
device = "cuda:1"
random_seed = 42

sample_ids = {
    "m1_FMP": "V11L12-109_A1",
    "m3_FMP": "V11L12-109_B1",
    "m4_FMP": "V11L12-109_C1",
}


def resolve_job(sample_name, target_root):
    if sample_name not in sample_ids:
        raise ValueError(f"Unknown FMP sample: {sample_name}. Expected one of {sorted(sample_ids)}")
    sample_id = sample_ids[sample_name]
    sample_root = data_root / "mouse_brain" / "sma" / "V11L12-109" / sample_id / "output_data"
    sm_path = sample_root / f"{sample_id}_MSI" / f"{sample_id}.Visium.FMP.220826_smamsi.csv"
    st_path = sample_root / f"{sample_id}_RNA" / "outs"
    intermediate_path = target_root / "_intermediate" / sample_name / f"{sample_name}_aligned_selected.h5ad"
    output_path = target_root / f"{sample_name}.h5ad"
    return sm_path, st_path, intermediate_path, output_path


def run_sample(sample_name, target_root=output_root, target_device=device):
    sm_path, st_path, intermediate_path, output_path = resolve_job(sample_name, target_root)
    intermediate_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[job] sample={sample_name} sm_path={sm_path} st_path={st_path} "
        f"output_path={output_path} device={target_device}",
        flush=True,
    )
    joint_adata = preprocess_sm_st_to_h5ad(
        sm_input_path=sm_path,
        st_input_path=st_path,
        output_path=intermediate_path,
        device=target_device,
        random_seed=random_seed,
        min_total_intensity_raw=0.0,
        min_total_intensity_reassign=0.0,
        min_counts=0,
        min_genes=0,
        align_max_epoch=128,
        align_n_latent=10,
        rotation_degrees=-90.0,
        n_neighbors=5,
        dist_fold=1.5,
        joint_top_genes=2000,
        joint_top_metabolites=800,
    )
    model_input = prepare_spadta_model_input(
        joint_adata,
        modality="sm",
        expression_graph_k=3,
        spatial_context_k=12,
    )
    model_input.uns["spadta_preprocessing_source"] = {
        "kind": "fmp_raw_csv_visium",
        "sm": str(sm_path),
        "st": str(st_path),
    }
    model_input.write_h5ad(output_path)
    print(f"[done] sample={sample_name} shape={model_input.shape} output_path={output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Preprocess one FMP RNA+SM sample for SpaDTA.")
    parser.add_argument("--sample-name", default=sample, choices=sorted(sample_ids))
    parser.add_argument("--output-root", type=Path, default=output_root)
    parser.add_argument("--device", default=device)
    args = parser.parse_args()
    run_sample(args.sample_name, args.output_root, args.device)


if __name__ == "__main__":
    main()
