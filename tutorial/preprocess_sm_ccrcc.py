import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import scanpy as sc

from SpaDTA_718.model.preprocess import prepare_spadta_model_input, preprocess_aligned_joint_adata


data_root = Path("/bigdat2/user/hesy/spatialmeta/SpatialMETA")
output_root = data_root / "SpaDTA_718_model_input" / "SM"
sample = "R114_T"
samples = {"R114_T", "S15_T", "X49_T", "Y27_T", "Y7_T"}


def resolve_job(sample_name, target_root):
    if sample_name not in samples:
        raise ValueError(f"Unknown ccRCC sample: {sample_name}. Expected one of {sorted(samples)}")
    joint_raw_path = data_root / "ccRCC" / f"adata_joint_{sample_name}_raw.h5ad"
    output_path = target_root / f"{sample_name}.h5ad"
    return joint_raw_path, output_path


def run_sample(sample_name, target_root=output_root):
    joint_raw_path, output_path = resolve_job(sample_name, target_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[job] sample={sample_name} joint_raw_path={joint_raw_path} output_path={output_path}",
        flush=True,
    )
    joint_raw = sc.read_h5ad(joint_raw_path)
    joint_adata = preprocess_aligned_joint_adata(
        joint_raw,
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
        "kind": "ccrcc_aligned_joint_raw_h5ad",
        "joint": str(joint_raw_path),
    }
    model_input.write_h5ad(output_path)
    print(f"[done] sample={sample_name} shape={model_input.shape} output_path={output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Preprocess one ccRCC RNA+SM sample for SpaDTA.")
    parser.add_argument("--sample-name", default=sample, choices=sorted(samples))
    parser.add_argument("--output-root", type=Path, default=output_root)
    args = parser.parse_args()
    run_sample(args.sample_name, args.output_root)


if __name__ == "__main__":
    main()
