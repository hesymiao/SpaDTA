from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, "/data/user/hesy/projects/SpatialMETA")
from SpaDTA_718.downstream.run_strict_crossmodal_recovery import run_direction


PROJECT_ROOT = Path("/data/user/hesy/projects/SpatialMETA")
SOURCE = PROJECT_ROOT / "spaDTA" / "downstream" / "fig2k_2.py"
CURRENT_ROOT = PROJECT_ROOT / "SpaDTA_718" / "runs" / "sm_downstream" / "fig2k_m1_FMP"
TRAIN_INPUT = Path(
    "/bigdat2/user/hesy/spatialmeta/SpatialMETA/"
    "SpaDTA_718_model_input_preselect800_20260719/SM/m1_FMP.h5ad"
)


def load_plotter():
    spec = importlib.util.spec_from_file_location("spadta718_fig2k_m1_marker_recovery", SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    run_direction("sm_to_st", "cuda:1", "m1_FMP")
    plotter = load_plotter()
    generation_dir = CURRENT_ROOT / "sm_to_st_generation_spatial_top_third"
    output_root = CURRENT_ROOT / "marker_gene_recovery_from_sm_only"

    plotter.sample_name = "m1_FMP"
    plotter.train_input_h5ad = TRAIN_INPUT
    plotter.fig2k_root = CURRENT_ROOT
    plotter.generation_dir = generation_dir
    plotter.output_root = output_root
    plotter.figure_dir = output_root / "figures"
    plotter.group_figure_dir = plotter.figure_dir / "grouped"
    plotter.gene_figure_dir = plotter.figure_dir / "per_gene"
    plotter.autoscale_gene_figure_dir = plotter.figure_dir / "per_gene_autoscaled"
    plotter.table_dir = output_root / "tables"
    plotter.eval_h5ad_path = generation_dir / "eval_generated_st_from_sm_only.h5ad"
    plotter.feature_metrics_csv = generation_dir / "st_feature_metrics.csv"
    plotter.split_metadata_json = generation_dir / "split_metadata.json"
    plotter.MARKER_GROUPS = {
        "neuronal_projection": ["Penk", "Slc17a7", "Ppp1r1b", "Pde1b"],
        "inhibitory_neuron": ["Gad1", "Gad2", "Calb1", "Pvalb"],
        "glial_support": ["Apoe", "Plp1", "Mbp", "Aldoc"],
    }
    plotter.TOP_RECOVERY_GENES = ["Penk", "Pde1b", "Hpca", "Slc17a7", "Ppp1r1b", "Nsg2"]
    plotter.run_fig2k_2()


if __name__ == "__main__":
    main()
