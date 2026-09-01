from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.figure

sys.path.insert(0, "/data/user/hesy/projects/SpatialMETA")
from SpaDTA_718.downstream.run_strict_crossmodal_recovery import run_direction


PROJECT_ROOT = Path("/data/user/hesy/projects/SpatialMETA")
SOURCE = PROJECT_ROOT / "spaDTA" / "downstream" / "fig2k_2.py"
CURRENT_ROOT = PROJECT_ROOT / "SpaDTA_718" / "runs" / "sm_downstream" / "fig2k_X49_T"
CONFIG_PATH = PROJECT_ROOT / "SpaDTA_718" / "runs" / "SM" / "X49_T" / "config.json"
TRAIN_INPUT = Path(json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["input_h5ad_path"])


def load_legacy_plotter():
    spec = importlib.util.spec_from_file_location("spadta718_fig2k_marker_recovery", SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    run_direction("sm_to_st", "cuda:3", "X49_T")
    plotter = load_legacy_plotter()
    generation_dir = CURRENT_ROOT / "sm_to_st_generation_spatial_top_third"
    output_root = CURRENT_ROOT / "marker_gene_recovery_from_sm_only"

    plotter.sample_name = "X49_T"
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
    plotter.MARKER_GROUPS = {"selected_kidney_markers": ["CUBN", "SLC22A12"]}
    plotter.TOP_RECOVERY_GENES = ["CUBN", "SLC22A12"]
    plotter.plot_single_gene = lambda **kwargs: {}
    plotter.plot_group_summary = lambda **kwargs: {}
    plotter.plot_top_recovery_summary = lambda **kwargs: {}
    original_autoscaled_plot = plotter.plot_single_gene_autoscaled_pair

    def plot_autoscaled_png_svg_only(**kwargs):
        original_savefig = matplotlib.figure.Figure.savefig

        def savefig_without_pdf(figure, filename, *args, **savefig_kwargs):
            if str(filename).lower().endswith(".pdf"):
                return None
            return original_savefig(figure, filename, *args, **savefig_kwargs)

        matplotlib.figure.Figure.savefig = savefig_without_pdf
        try:
            return original_autoscaled_plot(**kwargs)
        finally:
            matplotlib.figure.Figure.savefig = original_savefig

    plotter.plot_single_gene_autoscaled_pair = plot_autoscaled_png_svg_only
    plotter.run_fig2k_2()


if __name__ == "__main__":
    main()
