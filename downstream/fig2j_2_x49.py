from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.figure

sys.path.insert(0, "/data/user/hesy/projects/SpatialMETA")
from SpaDTA_718.downstream.run_strict_crossmodal_recovery import run_direction


PROJECT_ROOT = Path("/data/user/hesy/projects/SpatialMETA")
SOURCE = PROJECT_ROOT / "spaDTA" / "downstream" / "fig2j_2.py"
CURRENT_ROOT = PROJECT_ROOT / "SpaDTA_718" / "runs" / "sm_downstream" / "fig2j_X49_T"


def load_legacy_plotter():
    spec = importlib.util.spec_from_file_location("spadta718_fig2j_marker_recovery", SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    run_direction("st_to_sm", "cuda:2", "X49_T")
    plotter = load_legacy_plotter()
    generation_dir = CURRENT_ROOT / "st_to_sm_generation_spatial_top_third"
    output_root = CURRENT_ROOT / "marker_metabolite_recovery_from_st_only"

    plotter.sample_name = "X49_T"
    plotter.fig2j_root = CURRENT_ROOT
    plotter.generation_dir = generation_dir
    plotter.output_root = output_root
    plotter.figure_dir = output_root / "figures"
    plotter.group_figure_dir = plotter.figure_dir / "grouped"
    plotter.metabolite_figure_dir = plotter.figure_dir / "per_metabolite"
    plotter.autoscale_metabolite_figure_dir = plotter.figure_dir / "per_metabolite_autoscaled"
    plotter.table_dir = output_root / "tables"
    plotter.eval_h5ad_path = generation_dir / "eval_generated_sm_from_st_only.h5ad"
    plotter.feature_metrics_csv = generation_dir / "sm_feature_metrics.csv"
    plotter.split_metadata_json = generation_dir / "split_metadata.json"
    plotter.MARKER_METABOLITES = [
        entry
        for entry in plotter.MARKER_METABOLITES
        if float(entry["target_mz"]) in {227.10823423028484, 758.5685152097649}
    ]
    plotter.plot_single_metabolite = lambda **kwargs: {}
    plotter.plot_selected_metabolite_summary = lambda **kwargs: {}
    original_autoscaled_plot = plotter.plot_single_metabolite_autoscaled_pair

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

    plotter.plot_single_metabolite_autoscaled_pair = plot_autoscaled_png_svg_only
    plotter.hmdb_path = PROJECT_ROOT / "spatialmeta" / "data" / "hmdb.csv"
    plotter.de_table_path = (
        PROJECT_ROOT
        / "SpaDTA_718"
        / "runs"
        / "sm_downstream"
        / "fig2f_X49_T"
        / "tables"
        / "X49_T_marker_named_group_Imm_marker_named_cluster_sm_de_full.csv"
    )
    plotter.grouped_figure_title = "X49_T recovery of biologically relevant putative marker metabolites"
    plotter.analysis_selection_note = (
        "本图用于展示 X49_T ccRCC 组织中恢复良好且具有生物学意义的候选代谢峰。候选综合考虑 Pearson、Spearman、"
        "top-10% hotspot overlap、真实非零 spot 比例、Imm_1 富集和肾脏肿瘤/免疫代谢文献，因此是定性展示，不是无偏 marker benchmark。"
    )
    plotter.analysis_annotation_note = (
        "四个名称均为 HMDB exact-mass 推定注释，并使用与该数据相符的常见离子形式匹配。"
        "精确质量不能排除同分异构体，PA 脂质的脂肪酸链位置尤其不能仅凭 MS1 确定；正文必须使用 putative/tentative，不能写成 MS/MS 已确认。"
    )
    plotter.analysis_overall_note = (
        "DMGV 和 PA(40:3) candidate 的恢复最强；19,20-DiHDPA 与 14,15-DHET candidate 的总体相关中等，"
        "但在 Imm_1 富集并保留了可辨认的空间 hotspot。四个峰真实非零 spot 比例均超过 50%，不属于几乎无表达的低信号 marker。"
    )
    plotter.analysis_reference_note = (
        "Biological support: DMGV/AGXT2 kidney metabolism (PMID: 31818439); "
        "19,20-DiHDPA pro-resolutive oxylipin biology (PMID: 38054009); "
        "14,15-DHET oxylipin biology (PMID: 35083437); exact chemical identities require MS/MS confirmation."
    )
    plotter.MARKER_METABOLITES = [
        {
            "target_mz": 227.10823423028484,
            "assigned_name": "Putative dimethylguanidino valeric acid ([M+Na]+)",
            "display_name": "DMGV",
            "ion_form": "putative [M+Na]+",
            "metabolite_class": "methylated arginine catabolite",
            "accession": "HMDB0240212",
            "ppm_error": 0.0014,
            "story_tag": "dmgv_kidney_metabolism",
            "story_label": "AGXT2-linked renal arginine metabolism",
            "annotation_evidence": "HMDB0240212 [M+Na]+ exact-mass match (0.001 ppm); AGXT2 is strongly expressed in kidney and metabolizes dimethylarginines (PMID: 31818439)",
        },
        {
            "target_mz": 758.5685152097649,
            "assigned_name": "Putative PA(40:3) species ([M+H]+)",
            "display_name": "PA(40:3) candidate",
            "ion_form": "putative [M+H]+",
            "metabolite_class": "phosphatidic acid",
            "accession": "HMDB0114914",
            "ppm_error": 0.0013,
            "story_tag": "phosphatidic_acid_membrane_remodeling",
            "story_label": "phosphatidic acid / membrane remodeling",
            "annotation_evidence": "HMDB0114914 [M+H]+ exact-mass match (0.001 ppm); sum composition is plausible, but acyl-chain positions require MS/MS",
        },
        {
            "target_mz": 389.24709140897994,
            "assigned_name": "Putative 19,20-DiHDPA ([M+Na]+)",
            "display_name": "19,20-DiHDPA candidate",
            "ion_form": "putative [M+Na]+",
            "metabolite_class": "DHA-derived oxylipin",
            "accession": "HMDB0010214",
            "ppm_error": 0.1418,
            "story_tag": "dihdpa_immune_oxylipin",
            "story_label": "DHA-derived immune-resolution oxylipin",
            "annotation_evidence": "HMDB0010214 [M+Na]+ exact-mass match (0.142 ppm); strongest Imm_1 enrichment among these candidates and supported DHA-oxylipin biology (PMID: 38054009)",
        },
        {
            "target_mz": 365.24700583231225,
            "assigned_name": "Putative 14,15-DiHETrE/14,15-DHET ([M+Na]+)",
            "display_name": "14,15-DHET candidate",
            "ion_form": "putative [M+Na]+",
            "metabolite_class": "arachidonic-acid-derived oxylipin",
            "accession": "HMDB0002265",
            "ppm_error": 0.0312,
            "story_tag": "dhet_immune_oxylipin",
            "story_label": "arachidonic-acid oxylipin remodeling",
            "annotation_evidence": "HMDB0002265 [M+Na]+ exact-mass match (0.031 ppm); Imm_1-enriched oxylipin candidate with context-dependent inflammatory and vascular biology (PMID: 35083437)",
        },
    ]
    plotter.MARKER_METABOLITES = [
        entry
        for entry in plotter.MARKER_METABOLITES
        if float(entry["target_mz"]) in {227.10823423028484, 758.5685152097649}
    ]
    plotter.run_fig2j_2()


if __name__ == "__main__":
    main()
