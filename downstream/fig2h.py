from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests


project_root = Path("/data/user/hesy/projects/SpatialMETA")
source_script = project_root / "SpaDTA_718" / "downstream" / "plot_subtype_fig3hi_like_enrichment.py"
prerequisite_script = project_root / "SpaDTA_718" / "downstream" / "fig2f.py"

sample_name = "X49_T"
major_name = "Imm"
subtype_name = "Imm_1"

run_root = project_root / "SpaDTA_718" / "runs" / "sm_downstream"
prerequisite_root = run_root / "fig2f"
de_input_table_dir = prerequisite_root / "tables"
root_dir = run_root / "fig2h"
table_dir = root_dir / "tables"
fig_dir = root_dir / "figures"

log2fc_threshold = 0.2


def run_prerequisite() -> None:
    required = [
        de_input_table_dir / f"{sample_name}_marker_named_group_{major_name}_marker_named_cluster_st_de_full.csv",
        de_input_table_dir / f"{sample_name}_marker_named_group_{major_name}_marker_named_cluster_sm_de_full.csv",
    ]
    if all(path.exists() for path in required):
        return

    spec = importlib.util.spec_from_file_location("fig2f_module", prerequisite_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load prerequisite script: {prerequisite_script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.run_fig2f()


def load_plotter_namespace(script_path: Path) -> dict[str, object]:
    source = script_path.read_text(encoding="utf-8")
    if "from __future__ import annotations" not in source:
        source = "from __future__ import annotations\n" + source

    source = source.replace(
        '    fig.savefig(figure_path, dpi=300, bbox_inches="tight")\n'
        '    plt.close(fig)\n',
        '    svg_path = figure_path.with_suffix(".svg")\n'
        '    fig.savefig(figure_path, dpi=300, bbox_inches="tight")\n'
        '    fig.savefig(svg_path, bbox_inches="tight", format="svg")\n'
        '    plt.close(fig)\n',
    )
    source = source.replace(
        '        "figure": str(figure_path),\n',
        '        "figure_png": str(figure_path),\n'
        '        "figure_svg": str(figure_path.with_suffix(".svg")),\n',
    )
    source = source.replace(
        '    left_edge = max(0.0, float(go_plot["combined_score"].min()) - 4.0)\n',
        '    left_edge = 2500.0\n',
    )
    source = source.replace(
        '    st_table_path = table_dir / f"{sample_name}_marker_named_group_{major_name}_marker_named_cluster_st_de_full.csv"\n'
        '    sm_table_path = table_dir / f"{sample_name}_marker_named_group_{major_name}_marker_named_cluster_sm_de_full.csv"\n',
        '    st_table_path = de_input_table_dir / f"{sample_name}_marker_named_group_{major_name}_marker_named_cluster_st_de_full.csv"\n'
        '    sm_table_path = de_input_table_dir / f"{sample_name}_marker_named_group_{major_name}_marker_named_cluster_sm_de_full.csv"\n',
    )

    namespace: dict[str, object] = {
        "__name__": "fig2h_plotter_namespace",
        "__file__": str(script_path),
    }
    exec(compile(source, str(script_path), "exec"), namespace)

    namespace["sample_name"] = sample_name
    namespace["major_name"] = major_name
    namespace["subtype_name"] = subtype_name
    namespace["root_dir"] = root_dir
    namespace["de_input_table_dir"] = de_input_table_dir
    namespace["table_dir"] = table_dir
    namespace["fig_dir"] = fig_dir
    namespace["log2fc_threshold"] = log2fc_threshold
    namespace["gprofiler_max_retries"] = 1
    namespace["gprofiler_timeout"] = 20
    return namespace


def query_go_bp_with_fallback(
    genes: list[str],
    primary_fn,
    strip_go_id_fn,
) -> pd.DataFrame:
    try:
        return primary_fn(genes)
    except Exception:
        pass

    add_response = requests.post(
        "https://maayanlab.cloud/Enrichr/addList",
        files={
            "list": (None, "\n".join(genes)),
            "description": (None, f"{sample_name}_{subtype_name}_fig2h"),
        },
        timeout=20,
    )
    add_response.raise_for_status()
    user_list_id = int(add_response.json()["userListId"])

    enrich_response = requests.get(
        f"https://maayanlab.cloud/Enrichr/enrich?userListId={user_list_id}&backgroundType=GO_Biological_Process_2023",
        timeout=20,
    )
    enrich_response.raise_for_status()
    result = enrich_response.json().get("GO_Biological_Process_2023", [])

    query_size = max(len(set(genes)), 1)
    rows: list[dict[str, object]] = []
    for item in result:
        rank, term_name, p_value, z_score, combined_score, overlap_genes, adjusted_p_value = item[:7]
        overlap_genes = [str(x) for x in overlap_genes]
        overlap_count = len(overlap_genes)
        query_pct = 100.0 * overlap_count / query_size
        rows.append(
            {
                "rank": int(rank),
                "term": str(term_name),
                "term_display": strip_go_id_fn(term_name),
                "p_value": float(p_value),
                "z_score": float(z_score),
                "combined_score": float(combined_score),
                "overlap_genes": ";".join(overlap_genes),
                "overlap_count": overlap_count,
                "adjusted_p_value": float(adjusted_p_value),
                "query_gene_pct": query_pct,
                "term_size": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
            }
        )

    go_df = pd.DataFrame(rows)
    if go_df.empty:
        return go_df
    go_df = go_df.sort_values(["adjusted_p_value", "combined_score"], ascending=[True, False]).reset_index(drop=True)
    go_df["log10_1_over_fdr"] = -np.log10(np.clip(go_df["adjusted_p_value"].astype(float), 1e-300, None))
    return go_df


def run_fig2h() -> dict[str, object]:
    run_prerequisite()
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    namespace = load_plotter_namespace(source_script)
    primary_go_fn = namespace["query_gprofiler_go_bp"]
    strip_go_id_fn = namespace["strip_go_id"]
    namespace["query_gprofiler_go_bp"] = lambda genes: query_go_bp_with_fallback(genes, primary_go_fn, strip_go_id_fn)
    plot_fn = namespace["plot_subtype_fig3hi_like_enrichment"]
    plot_panel_fn = namespace["plot_go_and_metabolite_enrichment"]

    prefix = f"{sample_name}_{subtype_name}_fig3hi_like_enrichment"
    figure_path = fig_dir / f"{prefix}.png"
    go_table_path = table_dir / f"{sample_name}_{subtype_name}_go_bp_enrichr.csv"
    metab_table_path = table_dir / f"{sample_name}_{subtype_name}_metabolite_group_enrichment_sub_class.csv"
    annotation_table_path = table_dir / f"{sample_name}_{subtype_name}_upregulated_metabolite_annotations.csv"
    query_gene_table_path = table_dir / f"{sample_name}_{subtype_name}_upregulated_genes.csv"
    query_metab_table_path = table_dir / f"{sample_name}_{subtype_name}_upregulated_metabolites.csv"
    summary_path = fig_dir / f"{prefix}_summary.json"

    cached_paths = [
        go_table_path,
        metab_table_path,
        annotation_table_path,
        query_gene_table_path,
        query_metab_table_path,
    ]
    if all(path.exists() for path in cached_paths):
        go_df = pd.read_csv(go_table_path)
        metab_df = pd.read_csv(metab_table_path)
        plot_panel_fn(go_df, metab_df, figure_path, f"{subtype_name.upper()}: GO term and metabolite group enrichment")
        summary = {
            "sample": sample_name,
            "major": major_name,
            "subtype": subtype_name,
            "log2fc_threshold": log2fc_threshold,
            "go_library": "GO_Biological_Process_2023",
            "metabolite_group_level": "sub_class",
            "query_gene_count": int(pd.read_csv(query_gene_table_path)["feature"].astype(str).nunique()),
            "query_metabolite_count": int(pd.read_csv(query_metab_table_path)["feature"].astype(str).nunique()),
            "annotated_query_metabolites": int(pd.read_csv(annotation_table_path)["feature"].astype(str).nunique()),
            "figure_png": str(figure_path),
            "figure_svg": str(figure_path.with_suffix(".svg")),
            "tables": {
                "upregulated_genes": str(query_gene_table_path),
                "go_bp_enrichment": str(go_table_path),
                "upregulated_metabolites": str(query_metab_table_path),
                "metabolite_annotations": str(annotation_table_path),
                "metabolite_group_enrichment": str(metab_table_path),
            },
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"figure": str(figure_path), "summary": str(summary_path), "mode": "cached_replot"}, ensure_ascii=False, indent=2))
        return summary

    return plot_fn(
        sample_name=sample_name,
        major_name=major_name,
        subtype_name=subtype_name,
        root_dir=root_dir,
        table_dir=table_dir,
        fig_dir=fig_dir,
        log2fc_threshold=log2fc_threshold,
    )


if __name__ == "__main__":
    run_fig2h()
