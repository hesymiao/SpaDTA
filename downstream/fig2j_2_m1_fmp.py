from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/data/user/hesy/projects/SpatialMETA")
from SpaDTA_718.downstream.run_strict_crossmodal_recovery import run_direction


PROJECT_ROOT = Path("/data/user/hesy/projects/SpatialMETA")
SOURCE = PROJECT_ROOT / "spaDTA" / "downstream" / "fig2j_2.py"
CURRENT_ROOT = PROJECT_ROOT / "SpaDTA_718" / "runs" / "sm_downstream" / "fig2j_m1_FMP"


def load_plotter():
    spec = importlib.util.spec_from_file_location("spadta718_fig2j_m1_marker_recovery", SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    run_direction("st_to_sm", "cuda:0", "m1_FMP")
    plotter = load_plotter()
    generation_dir = CURRENT_ROOT / "st_to_sm_generation_spatial_top_third"
    output_root = CURRENT_ROOT / "marker_metabolite_recovery_from_st_only"

    plotter.sample_name = "m1_FMP"
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
    plotter.hmdb_path = PROJECT_ROOT / "spatialmeta" / "data" / "hmdb.csv"
    plotter.load_best_de_context = lambda: pd.DataFrame(
        columns=[
            "feature",
            "best_enriched_subtype",
            "best_enriched_log2fc",
            "best_enriched_score",
            "best_mean_in",
            "best_mean_out",
        ]
    )
    plotter.grouped_figure_title = "m1_FMP recovery of biologically relevant putative marker metabolites"
    plotter.analysis_selection_note = (
        "本图用于展示恢复良好且具有脑生物学意义的候选峰。候选先要求 Pearson、Spearman 和空间 hotspot 指标均有实际恢复信号，"
        "再依据 FMP-10 反应基团、单标签后的中性质量和脑组织文献筛选；因此它是定性示例，不是无偏 marker benchmark。"
    )
    plotter.analysis_annotation_note = (
        "四个名称均为推定注释：观测 m/z 扣除一个 FMP-10 标签质量 268.1124 Da 后，与 HMDB 中性质量在 5 ppm 内，"
        "且候选分子含伯胺或酚羟基。它们没有在 Vicari et al. 补充表中逐峰进行标准品 MS/MS，因此不能写成已确认鉴定。"
    )
    plotter.analysis_overall_note = (
        "这组候选的 Pearson 为 0.6362-0.9155，Spearman 为 0.5605-0.8045。"
        "其中 5-S-cysteinyldopamine 的三项空间恢复指标最好；Glutathione、Norlaudanosoline 和 Adenosine 也保留了可辨认的空间排序，"
        "但其化学身份仍需 MS/MS 标准品确认。"
    )
    plotter.MARKER_METABOLITES = [
        {
            "target_mz": 540.19567,
            "assigned_name": "Putative 5-S-cysteinyldopamine (single FMP-10 derivative)",
            "display_name": "5-S-Cysteinyldopamine",
            "ion_form": "putative single FMP-10 derivative",
            "metabolite_class": "dopamine thioether / oxidative dopamine metabolite",
            "accession": "HMDB0246842",
            "ppm_error": 0.7050,
            "story_tag": "cysteinyldopamine",
            "story_label": "dopamine oxidation and thiol-conjugation marker",
            "annotation_evidence": "Putative single-FMP mass match to HMDB0246842 (0.705 ppm); dopamine cysteine conjugates occur in human striatum (PMID: 11701754)",
        },
        {
            "target_mz": 575.19567,
            "assigned_name": "Putative glutathione (single FMP-10 derivative)",
            "display_name": "Glutathione",
            "ion_form": "putative single FMP-10 derivative",
            "metabolite_class": "glutathione / cellular redox metabolite",
            "accession": "HMDB0000125",
            "ppm_error": 1.7469,
            "story_tag": "glutathione",
            "story_label": "brain antioxidant and redox marker",
            "annotation_evidence": "Putative single-FMP mass match to HMDB0000125 (1.747 ppm); canonical cellular antioxidant, chemical identity not MS/MS-confirmed here",
        },
        {
            "target_mz": 555.22673,
            "assigned_name": "Putative norlaudanosoline (single FMP-10 derivative)",
            "display_name": "Norlaudanosoline",
            "ion_form": "putative single FMP-10 derivative",
            "metabolite_class": "dopamine-derived tetrahydroisoquinoline",
            "accession": "HMDB0012486",
            "ppm_error": 4.9737,
            "story_tag": "norlaudanosoline",
            "story_label": "dopamine-derived oxidative-stress metabolite",
            "annotation_evidence": "Putative single-FMP mass match to HMDB0012486 (4.974 ppm); detected in murine brain regions (PMID: 34474098)",
        },
        {
            "target_mz": 535.20994,
            "assigned_name": "Putative adenosine (single FMP-10 derivative)",
            "display_name": "Adenosine",
            "ion_form": "putative single FMP-10 derivative",
            "metabolite_class": "purine nucleoside / neuromodulator",
            "accession": "HMDB0000050",
            "ppm_error": 2.9430,
            "story_tag": "adenosine",
            "story_label": "striatal purinergic neuromodulation marker",
            "annotation_evidence": "Putative single-FMP mass match to HMDB0000050 (2.943 ppm); adenosine receptors modulate striatal glutamatergic and dopaminergic signaling (PMID: 36375695)",
        },
    ]
    plotter.run_fig2j_2()


if __name__ == "__main__":
    main()
