from __future__ import annotations

from pathlib import Path


project_root = Path("/data/user/hesy/projects/SpatialMETA")
source_script = project_root / "SpaDTA_718" / "downstream" / "plot_major_class_subtype_de_log2fc.py"

sample = "X49_T"
run_root = project_root / "SpaDTA_718" / "runs" / "sm_downstream"
input_h5ad = run_root / "inputs" / sample / f"{sample}_output.h5ad"
cluster_label_h5ad = input_h5ad
feature_source_h5ad = input_h5ad
output_dir = run_root / "fig2f"

cluster_key = "decalign_linear_clusters"
major_value = "Imm"
norm_layer = "normalized"
log2fc_threshold = 0.2
sm_log2fc_threshold = 1.0
top_labels_per_group = 3
min_subtype_spots = 10
seed = 42
margin_mixed = 0.25
margin_high = 0.6


def load_plotter_namespace(script_path: Path) -> dict[str, object]:
    source = script_path.read_text(encoding="utf-8")
    if "from __future__ import annotations" not in source:
        source = "from __future__ import annotations\n" + source
    source = source.replace("output_pdf", "output_svg")
    source = source.replace("_st_sm_log2fc.pdf", "_st_sm_log2fc.svg")
    source = source.replace('fig.savefig(output_svg, bbox_inches="tight")', 'fig.savefig(output_svg, bbox_inches="tight", format="svg")')
    source = source.replace('"figure_pdf"', '"figure_svg"')

    namespace: dict[str, object] = {
        "__name__": "fig2f_plotter_namespace",
        "__file__": str(script_path),
        "CLUSTER_KEY": cluster_key,
        "NORM_LAYER": norm_layer,
        "LOG2FC_THRESHOLD": log2fc_threshold,
        "SM_LOG2FC_THRESHOLD": sm_log2fc_threshold,
        "TOP_LABELS_PER_GROUP": top_labels_per_group,
        "MIN_SUBTYPE_SPOTS": min_subtype_spots,
        "RANDOM_SEED": seed,
        "MARGIN_MIXED": margin_mixed,
        "MARGIN_HIGH": margin_high,
    }
    exec(compile(source, str(script_path), "exec"), namespace)
    return namespace


def run_fig2f() -> dict[str, object]:
    namespace = load_plotter_namespace(source_script)
    plot_fn = namespace["plot_major_class_subtype_de_log2fc"]
    return plot_fn(
        sample=sample,
        input_h5ad=input_h5ad,
        output_dir=output_dir,
        cluster_label_h5ad=cluster_label_h5ad,
        feature_source_h5ad=feature_source_h5ad,
        cluster_key=cluster_key,
        major_value=major_value,
        norm_layer=norm_layer,
        log2fc_threshold=log2fc_threshold,
        sm_log2fc_threshold=sm_log2fc_threshold,
        top_labels_per_group=top_labels_per_group,
        min_subtype_spots=min_subtype_spots,
        seed=seed,
        margin_mixed=margin_mixed,
        margin_high=margin_high,
    )


if __name__ == "__main__":
    run_fig2f()
