import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import scanpy as sc
import spatialmeta as smt
from pyimzml.ImzMLParser import ImzMLParser

from SpaDTA_718.model.preprocess import (
    prepare_spadta_model_input,
    preprocess_sm_st_adatas_to_h5ad,
    select_lower_left_component_indices,
    write_component_summary_csv,
)


data_root = Path("/bigdat2/user/hesy/spatialmeta/SpatialMETA")
output_root = data_root / "SpaDTA_718_model_input" / "SM"
sample = "248_T"
device = "cuda:1"
random_seed = 42
neighbor_radius = 1.5

sm_path = data_root / "GBM" / "MALDI_1" / "raw" / "20201029scilslab_ncfr_glia_combine_root_mean_square.imzML"
st_path = data_root / "GBM" / "10XVisium_2" / "10XVisium 2" / "#UKF248_T_ST" / "outs"


class SubsetImzMLParser:
    def __init__(self, parser, spectrum_indices):
        self.parser = parser
        self.spectrum_indices = list(map(int, spectrum_indices))
        self.coordinates = [parser.coordinates[index] for index in self.spectrum_indices]

    def getspectrum(self, index):
        return self.parser.getspectrum(self.spectrum_indices[index])


def read_raw_data(diagnostic_root):
    imzml_parser = ImzMLParser(str(sm_path))
    coordinates = [coordinate[:2] for coordinate in imzml_parser.coordinates]
    selected_indices, component_rows, selected_row = select_lower_left_component_indices(
        coordinates,
        neighbor_radius=neighbor_radius,
    )
    selected_parser = SubsetImzMLParser(imzml_parser, selected_indices)

    # Historical preprocessing used a global reference across all six tissue blocks.
    mz_reference = smt.pp.get_mz_reference(imzml_parser, ppm_tolerance=5)
    adata_sm = smt.pp.read_sm_imzml_as_anndata(selected_parser, mz_reference)
    adata_sm.obs_names = [str(index) for index in selected_indices]
    write_component_summary_csv(component_rows, diagnostic_root / "sm_component_summary.csv")
    adata_sm.write_h5ad(diagnostic_root / "248_T_sm_raw_lower_left.h5ad")
    print(f"[selected] component={selected_row} shape={adata_sm.shape}", flush=True)
    adata_st = sc.read_visium(str(st_path))
    adata_st.var_names_make_unique()
    return adata_sm, adata_st


def run_sample(sample_name, target_root=output_root, target_device=device):
    if sample_name != "248_T":
        raise ValueError("The GBM preprocessing entry only supports 248_T")
    intermediate_root = target_root / "_intermediate" / sample_name
    intermediate_root.mkdir(parents=True, exist_ok=True)
    output_path = target_root / f"{sample_name}.h5ad"
    print(
        f"[job] sample={sample_name} sm_path={sm_path} st_path={st_path} "
        f"output_path={output_path} device={target_device}",
        flush=True,
    )
    adata_sm, adata_st = read_raw_data(intermediate_root)
    joint_adata = preprocess_sm_st_adatas_to_h5ad(
        adata_sm_raw=adata_sm,
        adata_st_raw=adata_st,
        output_path=intermediate_root / f"{sample_name}_aligned_selected.h5ad",
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
        "kind": "gbm_raw_imzml_visium",
        "sm": str(sm_path),
        "st": str(st_path),
    }
    model_input.write_h5ad(output_path)
    print(f"[done] sample={sample_name} shape={model_input.shape} output_path={output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Preprocess the 248_T RNA+SM sample for SpaDTA.")
    parser.add_argument("--sample-name", default=sample, choices=[sample])
    parser.add_argument("--output-root", type=Path, default=output_root)
    parser.add_argument("--device", default=device)
    args = parser.parse_args()
    run_sample(args.sample_name, args.output_root, args.device)


if __name__ == "__main__":
    main()
