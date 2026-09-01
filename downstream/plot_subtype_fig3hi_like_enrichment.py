import json
import textwrap
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib.colors import LinearSegmentedColormap, Normalize
from molmass import Formula
from scipy.stats import hypergeom


project_root = Path("/data/user/hesy/projects/SpatialMETA")
compare_root = project_root / "compare_method"
hmdb_path = project_root / "spatialmeta" / "data" / "hmdb.csv"
sample_name = "Y27_T"
major_name = "Endo"
subtype_name = "Endo_4"
root_dir = compare_root / "ours" / "runs" / sample_name / "real_annotation_subtype_de"
table_dir = root_dir / "tables"
fig_dir = root_dir / "figures"
log2fc_threshold = 0.2
top_go_terms = 5
top_metabolite_groups = 4
ppm_tolerance = 5.0
go_library = "GO_Biological_Process_2023"
metabolite_group_level = "sub_class"
gprofiler_max_retries = 5
gprofiler_timeout = 120

purple_cmap = LinearSegmentedColormap.from_list("grey_purple", ["#d8d5dd", "#c8b7eb", "#8e61c0"])
adducts = [
    ("add", "H"),
    ("add", "Na"),
    ("add", "K"),
    ("sub", "H"),
    ("sub", "Cl"),
]


def wrap_label(text, width):
    return textwrap.fill(str(text), width=width)


def strip_go_id(term):
    return str(term).split(" (GO:")[0].strip()


def prepare_hmdb():
    hmdb = pd.read_csv(hmdb_path).copy()
    hmdb["monisotopic_molecular_weight"] = hmdb["monisotopic_molecular_weight"].astype(float)
    hmdb.attrs["ppm_tolerance"] = float(ppm_tolerance)
    for method, adduct in adducts:
        mass = Formula(adduct).monoisotopic_mass
        col = f"mz_{method}_{adduct}"
        if method == "add":
            hmdb[col] = hmdb["monisotopic_molecular_weight"] + mass
        else:
            hmdb[col] = hmdb["monisotopic_molecular_weight"] - mass
    return hmdb


def annotate_mz_best_hit(mz, hmdb):
    mz = float(mz)
    tolerance = float(hmdb.attrs["ppm_tolerance"])
    best = None
    for method, adduct in adducts:
        col = f"mz_{method}_{adduct}"
        ppm_error = np.abs(hmdb[col].astype(float) - mz) / mz * 1e6
        idx = int(ppm_error.idxmin())
        min_ppm = float(ppm_error.loc[idx])
        if min_ppm > tolerance:
            continue
        row = hmdb.loc[idx]
        candidate = {
            "feature_mz": mz,
            "ppm_error": min_ppm,
            "adduct_mode": f"{method}_{adduct}",
            "accession": row.get("accession"),
            "metabolite_name": row.get("name"),
            "direct_parent": row.get("direct_parent"),
            "class": row.get("class"),
            "sub_class": row.get("sub_class"),
        }
        if best is None or min_ppm < float(best["ppm_error"]):
            best = candidate
    return best


def query_gprofiler_go_bp(genes):
    last_error = None
    rows = None
    source = "GO:BP"
    for attempt in range(gprofiler_max_retries):
        try:
            response = requests.post(
                "https://biit.cs.ut.ee/gprofiler/api/gost/profile/",
                json={
                    "organism": "hsapiens",
                    "query": genes,
                    "sources": [source],
                    "user_threshold": 1.0,
                    "no_iea": False,
                },
                timeout=gprofiler_timeout,
            )
            response.raise_for_status()
            rows = response.json()["result"]
            break
        except Exception as exc:
            last_error = exc
            if attempt == gprofiler_max_retries - 1:
                raise
            time.sleep(2.0 * (attempt + 1))
    if rows is None:
        raise RuntimeError("Failed to retrieve g:Profiler enrichment results") from last_error

    query_size = max(len(set(genes)), 1)
    out_rows = []
    for row in rows:
        if str(row.get("source")) != source:
            continue
        term_name = str(row.get("name"))
        term_id = str(row.get("native"))
        overlap_count = int(row.get("intersection_size", 0))
        overlap_genes = []
        for item in row.get("intersections", []):
            if isinstance(item, list) and item:
                overlap_genes.append(str(item[0]))
        query_pct = 100.0 * overlap_count / query_size
        adjusted_p = float(row.get("p_value", 1.0))
        composite_score = query_pct * (-np.log10(max(adjusted_p, 1e-300)))
        out_rows.append(
            {
                "rank": int(len(out_rows) + 1),
                "term": f"{term_name} ({term_id})",
                "term_display": strip_go_id(term_name),
                "p_value": adjusted_p,
                "z_score": float("nan"),
                "combined_score": composite_score,
                "overlap_genes": ";".join(overlap_genes),
                "overlap_count": overlap_count,
                "adjusted_p_value": adjusted_p,
                "query_gene_pct": query_pct,
                "term_size": int(row.get("term_size", 0)),
                "precision": float(row.get("precision", 0.0)),
                "recall": float(row.get("recall", 0.0)),
            }
        )
    go_df = pd.DataFrame(out_rows)
    if go_df.empty:
        return go_df
    go_df = go_df.sort_values(["adjusted_p_value", "combined_score"], ascending=[True, False]).reset_index(drop=True)
    go_df["log10_1_over_fdr"] = -np.log10(np.clip(go_df["adjusted_p_value"].astype(float), 1e-300, None))
    return go_df


def metabolite_group_enrichment(annotated_query_df, annotated_bg_df):
    if annotated_query_df.empty or annotated_bg_df.empty:
        return pd.DataFrame(columns=["group", "p_value", "overlap", "background_count", "accessions", "log10_1_over_p"])
    query_unique = annotated_query_df[["accession", metabolite_group_level]].dropna().drop_duplicates().copy()
    bg_unique = annotated_bg_df[["accession", metabolite_group_level]].dropna().drop_duplicates().copy()
    query_accessions = set(query_unique["accession"].astype(str))

    total_background = bg_unique["accession"].astype(str).nunique()
    query_size = len(query_accessions)
    rows = []
    for group, group_df in bg_unique.groupby(metabolite_group_level, observed=True):
        bg_set = set(group_df["accession"].astype(str))
        overlap_accessions = sorted(bg_set.intersection(query_accessions))
        overlap = len(overlap_accessions)
        if overlap == 0:
            continue
        background_count = len(bg_set)
        p_value = float(hypergeom.sf(overlap - 1, total_background, background_count, query_size))
        rows.append(
            {
                "group": str(group),
                "p_value": p_value,
                "overlap": overlap,
                "background_count": background_count,
                "accessions": ";".join(overlap_accessions),
            }
        )
    enrich_df = pd.DataFrame(rows)
    if enrich_df.empty:
        return enrich_df
    enrich_df = enrich_df.sort_values(["p_value", "overlap"], ascending=[True, False]).reset_index(drop=True)
    enrich_df["log10_1_over_p"] = -np.log10(np.clip(enrich_df["p_value"].astype(float), 1e-300, None))
    return enrich_df


def build_query_tables(hmdb):
    st_table_path = table_dir / f"{sample_name}_marker_named_group_{major_name}_marker_named_cluster_st_de_full.csv"
    sm_table_path = table_dir / f"{sample_name}_marker_named_group_{major_name}_marker_named_cluster_sm_de_full.csv"
    st_df = pd.read_csv(st_table_path)
    sm_df = pd.read_csv(sm_table_path)

    gene_query_df = st_df[(st_df["subtype"].astype(str) == subtype_name) & (st_df["log2fc"].astype(float) > log2fc_threshold)].copy()
    gene_query_df = gene_query_df.sort_values(["log2fc", "score"], ascending=[False, False]).reset_index(drop=True)

    metab_query_df = sm_df[(sm_df["subtype"].astype(str) == subtype_name) & (sm_df["log2fc"].astype(float) > log2fc_threshold)].copy()
    metab_query_df = metab_query_df.sort_values(["log2fc", "score"], ascending=[False, False]).reset_index(drop=True)

    annotations = []
    for row in metab_query_df.itertuples(index=False):
        ann = annotate_mz_best_hit(float(row.feature), hmdb)
        if ann is None:
            continue
        ann.update(
            {
                "subtype": str(row.subtype),
                "feature": float(row.feature),
                "log2fc": float(row.log2fc),
                "pval_adj": float(row.pval_adj),
                "score": float(row.score),
            }
        )
        annotations.append(ann)
    annotated_query_df = pd.DataFrame(annotations)
    if not annotated_query_df.empty:
        annotated_query_df = annotated_query_df.sort_values(["ppm_error", "log2fc"], ascending=[True, False]).reset_index(drop=True)

    background_rows = []
    for mz in sorted(sm_df["feature"].astype(float).dropna().unique().tolist()):
        ann = annotate_mz_best_hit(mz, hmdb)
        if ann is not None:
            background_rows.append(ann)
    annotated_bg_df = pd.DataFrame(background_rows)
    if not annotated_bg_df.empty:
        annotated_bg_df = annotated_bg_df.sort_values(["ppm_error", "feature_mz"], ascending=[True, True]).reset_index(drop=True)

    return gene_query_df, metab_query_df, annotated_query_df, annotated_bg_df


def plot_go_and_metabolite_enrichment(go_df, metab_df, figure_path, title):
    if go_df.empty:
        raise RuntimeError("GO enrichment returned no terms.")
    if metab_df.empty:
        raise RuntimeError("Metabolite group enrichment returned no groups.")

    go_plot = go_df.head(top_go_terms).copy().sort_values("combined_score", ascending=True).reset_index(drop=True)
    metab_plot = metab_df.head(top_metabolite_groups).copy().sort_values("overlap", ascending=True).reset_index(drop=True)

    fig = plt.figure(figsize=(11.8, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.08, 0.92], wspace=0.56)
    ax_go = fig.add_subplot(gs[0, 0])
    ax_metab = fig.add_subplot(gs[0, 1])

    go_norm = Normalize(vmin=float(go_plot["log10_1_over_fdr"].min()), vmax=float(go_plot["log10_1_over_fdr"].max()))
    scatter = ax_go.scatter(
        go_plot["combined_score"],
        np.arange(len(go_plot)),
        s=np.clip(go_plot["query_gene_pct"].astype(float) * 30.0, 80, 280),
        c=go_plot["log10_1_over_fdr"],
        cmap=purple_cmap,
        norm=go_norm,
        edgecolors="none",
        zorder=3,
    )
    left_edge = max(0.0, float(go_plot["combined_score"].min()) - 4.0)
    for y, x in zip(np.arange(len(go_plot)), go_plot["combined_score"].astype(float)):
        ax_go.hlines(y=y, xmin=left_edge, xmax=x, color="#dddddd", lw=1.1, zorder=1)
    ax_go.set_yticks(np.arange(len(go_plot)))
    ax_go.set_yticklabels([wrap_label(x, width=22) for x in go_plot["term_display"]], fontsize=10)
    ax_go.set_xlabel("Combined Score", fontsize=11)
    ax_go.set_xlim(left=left_edge)
    ax_go.tick_params(axis="x", labelsize=10)
    ax_go.tick_params(axis="y", length=0)
    ax_go.spines["top"].set_visible(False)
    ax_go.spines["right"].set_visible(False)

    legend_vals = sorted(set(np.round(go_plot["query_gene_pct"].astype(float), 1).tolist()))
    if len(legend_vals) > 3:
        idxs = np.linspace(0, len(legend_vals) - 1, num=3, dtype=int)
        legend_vals = [legend_vals[i] for i in idxs]
    handles = [ax_go.scatter([], [], s=np.clip(v * 30.0, 80, 280), color="#9e9e9e") for v in legend_vals]
    labels = [f"{v:.1f}" for v in legend_vals]
    ax_go.legend(
        handles,
        labels,
        title="% Query genes",
        frameon=False,
        fontsize=9,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(-0.02, 1.23),
        ncol=max(1, len(labels)),
        handletextpad=0.6,
        columnspacing=1.1,
    )
    cbar_go = fig.colorbar(scatter, ax=ax_go, fraction=0.052, pad=0.04)
    cbar_go.ax.set_title("log10(1/FDR)", fontsize=9, pad=8)
    cbar_go.ax.tick_params(labelsize=9)

    metab_norm = Normalize(vmin=float(metab_plot["log10_1_over_p"].min()), vmax=float(metab_plot["log10_1_over_p"].max()))
    bar_colors = purple_cmap(metab_norm(metab_plot["log10_1_over_p"].astype(float).to_numpy()))
    ax_metab.barh(
        np.arange(len(metab_plot)),
        metab_plot["overlap"].astype(float),
        color=bar_colors,
        edgecolor="none",
        height=0.56,
    )
    ax_metab.set_yticks(np.arange(len(metab_plot)))
    ax_metab.set_yticklabels([wrap_label(x, width=18) for x in metab_plot["group"]], fontsize=10)
    ax_metab.set_xlabel("Metabolites in set", fontsize=11)
    ax_metab.tick_params(axis="x", labelsize=10)
    ax_metab.tick_params(axis="y", length=0)
    ax_metab.spines["top"].set_visible(False)
    ax_metab.spines["right"].set_visible(False)
    ax_metab.set_xlim(0, max(1.0, float(metab_plot["overlap"].max()) + 1.5))
    scalar_mappable = plt.cm.ScalarMappable(norm=metab_norm, cmap=purple_cmap)
    scalar_mappable.set_array([])
    cbar_metab = fig.colorbar(scalar_mappable, ax=ax_metab, fraction=0.052, pad=0.04)
    cbar_metab.ax.set_title("log10(1/p)", fontsize=9, pad=8)
    cbar_metab.ax.tick_params(labelsize=9)

    fig.suptitle(title, fontsize=13, y=1.03)
    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_subtype_fig3hi_like_enrichment(
    sample_name: str,
    major_name: str,
    subtype_name: str,
    root_dir: Path,
    table_dir: Path,
    fig_dir: Path,
    log2fc_threshold: float,
):
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{sample_name}_{subtype_name}_fig3hi_like_enrichment"
    figure_path = fig_dir / f"{prefix}.png"
    go_table_path = table_dir / f"{sample_name}_{subtype_name}_go_bp_enrichr.csv"
    metab_table_path = table_dir / f"{sample_name}_{subtype_name}_metabolite_group_enrichment_{metabolite_group_level}.csv"
    annotation_table_path = table_dir / f"{sample_name}_{subtype_name}_upregulated_metabolite_annotations.csv"
    query_gene_table_path = table_dir / f"{sample_name}_{subtype_name}_upregulated_genes.csv"
    query_metab_table_path = table_dir / f"{sample_name}_{subtype_name}_upregulated_metabolites.csv"
    summary_path = fig_dir / f"{prefix}_summary.json"

    hmdb = prepare_hmdb()
    gene_query_df, metab_query_df, annotated_query_df, annotated_bg_df = build_query_tables(hmdb)

    if gene_query_df.empty:
        raise RuntimeError(f"No genes passed log2fc > {log2fc_threshold} for {subtype_name}.")
    if metab_query_df.empty:
        raise RuntimeError(f"No metabolites passed log2fc > {log2fc_threshold} for {subtype_name}.")

    gene_query_df.to_csv(query_gene_table_path, index=False)
    metab_query_df.to_csv(query_metab_table_path, index=False)
    annotated_query_df.to_csv(annotation_table_path, index=False)

    gene_query = gene_query_df["feature"].astype(str).drop_duplicates().tolist()
    go_df = query_gprofiler_go_bp(gene_query)
    go_df.to_csv(go_table_path, index=False)

    metab_df = metabolite_group_enrichment(annotated_query_df, annotated_bg_df)
    metab_df.to_csv(metab_table_path, index=False)

    title = f"{subtype_name.upper()}: GO term and metabolite group enrichment"
    plot_go_and_metabolite_enrichment(go_df, metab_df, figure_path, title)

    summary = {
        "sample": sample_name,
        "major": major_name,
        "subtype": subtype_name,
        "log2fc_threshold": log2fc_threshold,
        "go_library": go_library,
        "metabolite_group_level": metabolite_group_level,
        "query_gene_count": int(gene_query_df["feature"].astype(str).nunique()),
        "query_metabolite_count": int(metab_query_df["feature"].astype(str).nunique()),
        "annotated_query_metabolites": int(annotated_query_df["feature"].astype(str).nunique()) if not annotated_query_df.empty else 0,
        "figure": str(figure_path),
        "tables": {
            "upregulated_genes": str(query_gene_table_path),
            "go_bp_enrichment": str(go_table_path),
            "upregulated_metabolites": str(query_metab_table_path),
            "metabolite_annotations": str(annotation_table_path),
            "metabolite_group_enrichment": str(metab_table_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({"figure": str(figure_path), "summary": str(summary_path)}, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    plot_subtype_fig3hi_like_enrichment(
        sample_name=sample_name,
        major_name=major_name,
        subtype_name=subtype_name,
        root_dir=root_dir,
        table_dir=table_dir,
        fig_dir=fig_dir,
        log2fc_threshold=log2fc_threshold,
    )
