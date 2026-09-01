import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.ticker import FuncFormatter
from scipy import sparse


project_root = Path("/data/user/hesy/projects/SpatialMETA")
compare_root = project_root / "compare_method"

sample = "Y27_T"
input_h5ad = compare_root / "ours" / "runs" / sample / f"{sample}_ours_domains.h5ad"
cluster_label_h5ad = input_h5ad
feature_source_h5ad = input_h5ad
output_dir = compare_root / "ours" / "runs" / sample / "real_annotation_subtype_de"
cluster_key = CLUSTER_KEY
major_value = "Imm"
norm_layer = NORM_LAYER
log2fc_threshold = LOG2FC_THRESHOLD
sm_log2fc_threshold = SM_LOG2FC_THRESHOLD
top_labels_per_group = TOP_LABELS_PER_GROUP
min_subtype_spots = MIN_SUBTYPE_SPOTS
seed = RANDOM_SEED
margin_mixed = MARGIN_MIXED
margin_high = MARGIN_HIGH

CLUSTER_KEY = "decalign_linear_clusters"
NORM_LAYER = "normalized"
MAJOR_KEY = "marker_named_group"
SUBTYPE_KEY = "marker_named_cluster"
LOG2FC_EPS = 1e-3
LOG2FC_THRESHOLD = 0.2
SM_LOG2FC_THRESHOLD = 1.0
TOP_LABELS_PER_GROUP = 3
RANDOM_SEED = 42
MIN_SUBTYPE_SPOTS = 10
MARGIN_MIXED = 0.25
MARGIN_HIGH = 0.6

AXIS_LABEL_FONTSIZE = 24
TITLE_FONTSIZE = AXIS_LABEL_FONTSIZE
TICK_FONTSIZE = 22
ANNOTATION_TEXT_FONTSIZE = 20
FEATURE_LABEL_FONTSIZE = 15
LABEL_MIN_VERTICAL_GAP = 0.55

MARKER_SETS = {
    "Imm": ["CD3D", "CD74", "PTPRC", "NKG7"],
    "Endo": ["CD34", "PECAM1", "APLN"],
    "Stro": ["COL3A1", "ACTA2", "COL1A1", "DCN", "LUM"],
    "Mal": ["NDUFA4L2", "CNDP2", "EGFR", "CA9", "EPCAM", "KRT8", "KRT18", "KRT19"],
}

GROUP_COLOR_PALETTES = {
    "Imm": ["#5B2A86", "#4E55A7", "#9B4F96", "#6F58A4", "#9A86BA", "#B496C9"],
    "Endo": ["#5B2A86", "#4E55A7", "#9B4F96", "#6F58A4", "#9A86BA", "#B496C9"],
    "Stro": ["#7C5638", "#90664A", "#A4775D", "#B88A70", "#CB9F85", "#D9B398"],
    "Mal": ["#C65D4B", "#D57562", "#DF8D79", "#E7A48F", "#EFBBB0", "#F4D0C7"],
    "Mixed": ["#7B2CBF", "#9253C6", "#A96BCD", "#C08BD7", "#D5AEE3", "#E7D1F0"],
}


def natural_sort(values: list[str]) -> list[str]:
    def key(value: str) -> tuple[int, object]:
        text = str(value)
        return (0, int(text)) if text.isdigit() else (1, text)

    return sorted(values, key=key)


def to_dense(x) -> np.ndarray:
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def display_label(value: object) -> str:
    return str(value).upper()


def display_feature_label(value: object, modality_label: str) -> str:
    text = str(value)
    if modality_label == "SM":
        try:
            return f"{float(text):.2f}"
        except ValueError:
            return text
    return text


def load_adata(
    cluster_label_h5ad: Path,
    feature_source_h5ad: Path,
    cluster_key: str,
) -> sc.AnnData:
    label_adata = sc.read_h5ad(cluster_label_h5ad).copy()
    label_adata.obs_names = label_adata.obs_names.astype(str)

    feature_adata = sc.read_h5ad(feature_source_h5ad).copy()
    feature_adata.obs_names = feature_adata.obs_names.astype(str)
    if "name" in feature_adata.var.columns:
        feature_adata.var_names = feature_adata.var["name"].astype(str).values
        feature_adata.var_names_make_unique()

    if not feature_adata.obs_names.equals(label_adata.obs_names):
        raise ValueError("cluster-label h5ad and feature-source h5ad have mismatched obs_names.")
    if cluster_key not in label_adata.obs.columns:
        raise KeyError(f"missing obs[{cluster_key!r}] in {cluster_label_h5ad}")
    feature_adata.obs[cluster_key] = label_adata.obs[cluster_key].astype(str).values
    return feature_adata


def available_marker_sets(adata: sc.AnnData) -> dict[str, list[str]]:
    available: dict[str, list[str]] = {}
    for label, genes in MARKER_SETS.items():
        present = [gene for gene in genes if gene in adata.var_names]
        if not present:
            raise KeyError(f"marker set `{label}` has no available genes in this sample.")
        available[label] = present
    return available


def cluster_gene_means(adata: sc.AnnData, genes: list[str], cluster_key: str, norm_layer: str) -> pd.DataFrame:
    X = adata[:, genes].layers[norm_layer]
    X = to_dense(X).astype(np.float32, copy=False)
    df = pd.DataFrame(X, columns=genes)
    df["cluster"] = adata.obs[cluster_key].astype(str).values
    means = df.groupby("cluster", observed=True).mean()
    means = means.loc[natural_sort(means.index.astype(str).tolist())]
    return means


def annotate_clusters_single_sample(
    adata: sc.AnnData,
    cluster_key: str,
    norm_layer: str,
    major_value: str,
    margin_mixed: float,
    margin_high: float,
) -> tuple[sc.AnnData, pd.DataFrame]:
    marker_sets = available_marker_sets(adata)
    genes = [gene for values in marker_sets.values() for gene in values]
    means = cluster_gene_means(adata, genes, cluster_key, norm_layer)
    zscore = (means - means.mean(axis=0)) / means.std(axis=0).replace(0, np.nan)
    zscore = zscore.fillna(0.0)

    score_df = pd.DataFrame(index=means.index)
    for label, genes_in_set in marker_sets.items():
        score_df[label] = zscore[genes_in_set].mean(axis=1)

    cluster_sizes = adata.obs[cluster_key].astype(str).value_counts()
    rows: list[dict[str, object]] = []
    for cluster in score_df.index.astype(str):
        scores = {label: float(score_df.loc[cluster, label]) for label in marker_sets}
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best, best_score = ranked[0]
        second, second_score = ranked[1]
        margin = float(best_score - second_score)

        if margin < float(margin_mixed):
            broad = "Mixed"
            display = f"{best}/{second}"
            confidence = "low"
        elif margin < float(margin_high):
            broad = best
            display = f"{best}_like"
            confidence = "medium"
        else:
            broad = best
            display = best
            confidence = "high"

        rows.append(
            {
                "cluster": cluster,
                "n_spots": int(cluster_sizes[cluster]),
                "annotation_broad": broad,
                "annotation_display": display,
                "annotation_confidence": confidence,
                "best_type": best,
                "second_type": second,
                "margin": margin,
                **scores,
            }
        )

    annotation_df = pd.DataFrame(rows).sort_values("cluster", key=lambda s: s.astype(int)).reset_index(drop=True)

    target_df = annotation_df.loc[annotation_df["annotation_broad"].eq(major_value)].copy()
    target_df = target_df.sort_values(["n_spots", "cluster"], ascending=[False, True]).reset_index(drop=True)
    target_df["marker_named_group"] = major_value
    target_df["marker_named_cluster"] = [f"{major_value}_{i}" for i in range(1, len(target_df) + 1)]

    cluster_to_group = dict(zip(annotation_df["cluster"].astype(str), annotation_df["annotation_broad"].astype(str)))
    cluster_to_subtype = dict(zip(target_df["cluster"].astype(str), target_df["marker_named_cluster"].astype(str)))

    cluster_values = adata.obs[cluster_key].astype(str)
    adata.obs[MAJOR_KEY] = cluster_values.map(cluster_to_group).fillna("Other").astype(str)
    adata.obs[SUBTYPE_KEY] = cluster_values.map(cluster_to_subtype).fillna("").astype(str)
    adata.obs["marker_annotation_display"] = cluster_values.map(
        dict(zip(annotation_df["cluster"].astype(str), annotation_df["annotation_display"].astype(str)))
    ).astype(str)
    adata.obs["marker_annotation_confidence"] = cluster_values.map(
        dict(zip(annotation_df["cluster"].astype(str), annotation_df["annotation_confidence"].astype(str)))
    ).astype(str)

    return adata, annotation_df.merge(
        target_df.loc[:, ["cluster", "marker_named_group", "marker_named_cluster"]],
        on="cluster",
        how="left",
    )


def subset_and_select_subtypes(
    adata: sc.AnnData,
    major_value: str,
    min_spots: int,
    cluster_key: str,
) -> tuple[sc.AnnData, list[str], pd.DataFrame]:
    sub = adata[adata.obs[MAJOR_KEY].astype(str).eq(major_value).to_numpy()].copy()
    counts = sub.obs[SUBTYPE_KEY].astype(str).value_counts()
    counts = counts.loc[(counts >= min_spots) & (counts.index.astype(str) != "")]
    selected = natural_sort(counts.index.astype(str).tolist())
    sub = sub[sub.obs[SUBTYPE_KEY].astype(str).isin(selected).to_numpy()].copy()
    group_counts = sub.obs[SUBTYPE_KEY].astype(str).value_counts().reindex(selected).astype(int)
    group_info = group_counts.rename_axis("subtype").reset_index(name="n_spots")
    group_info["display_label"] = [display_label(subtype) for subtype in group_info["subtype"].astype(str)]
    original_cluster_map = (
        sub.obs[[SUBTYPE_KEY, cluster_key]]
        .drop_duplicates()
        .assign(_cluster_num=lambda df: pd.to_numeric(df[cluster_key], errors="coerce"))
        .sort_values([SUBTYPE_KEY, "_cluster_num", cluster_key], na_position="last")
        .drop_duplicates(subset=[SUBTYPE_KEY], keep="first")
        .rename(columns={SUBTYPE_KEY: "subtype", cluster_key: "original_cluster"})
        .loc[:, ["subtype", "original_cluster"]]
    )
    group_info = group_info.merge(original_cluster_map, on="subtype", how="left")
    return sub, selected, group_info


def build_modality_adata(adata: sc.AnnData, modality: str, norm_layer: str) -> sc.AnnData:
    mask = adata.var["type"].astype(str).eq(modality).to_numpy()
    subset = adata[:, mask].copy()
    subset.X = subset.layers[norm_layer].copy()
    sc.pp.log1p(subset)
    subset.obs[SUBTYPE_KEY] = subset.obs[SUBTYPE_KEY].astype(str)
    return subset


def rank_modality(adata_mod: sc.AnnData, selected_groups: list[str]) -> pd.DataFrame:
    sc.tl.rank_genes_groups(
        adata_mod,
        groupby=SUBTYPE_KEY,
        groups=selected_groups,
        reference="rest",
        method="wilcoxon",
        use_raw=False,
    )
    result = adata_mod.uns["rank_genes_groups"]
    frames: list[pd.DataFrame] = []
    for group in selected_groups:
        frames.append(
            pd.DataFrame(
                {
                    "subtype": group,
                    "feature": result["names"][group].astype(str),
                    "score": np.asarray(result["scores"][group], dtype=float),
                    "pval": np.asarray(result["pvals"][group], dtype=float),
                    "pval_adj": np.asarray(result["pvals_adj"][group], dtype=float),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def compute_log2fc_table(adata: sc.AnnData, modality: str, selected_groups: list[str], norm_layer: str) -> pd.DataFrame:
    mask = adata.var["type"].astype(str).eq(modality).to_numpy()
    features = adata.var_names[mask].astype(str)
    matrix = to_dense(adata[:, mask].layers[norm_layer]).astype(np.float32, copy=False)
    subtype_values = adata.obs[SUBTYPE_KEY].astype(str).to_numpy()

    rows: list[pd.DataFrame] = []
    for group in selected_groups:
        in_mask = subtype_values == group
        out_mask = subtype_values != group
        mean_in = np.asarray(matrix[in_mask].mean(axis=0), dtype=np.float32).ravel()
        mean_out = np.asarray(matrix[out_mask].mean(axis=0), dtype=np.float32).ravel()
        log2fc = np.log2((mean_in + LOG2FC_EPS) / (mean_out + LOG2FC_EPS))
        rows.append(
            pd.DataFrame(
                {
                    "subtype": group,
                    "feature": features,
                    "mean_in": mean_in,
                    "mean_out": mean_out,
                    "log2fc": log2fc,
                    "modality": modality,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def build_de_table(adata: sc.AnnData, modality: str, selected_groups: list[str], norm_layer: str) -> pd.DataFrame:
    adata_mod = build_modality_adata(adata, modality, norm_layer)
    ranked = rank_modality(adata_mod, selected_groups)
    foldchange = compute_log2fc_table(adata, modality, selected_groups, norm_layer)
    return ranked.merge(foldchange, on=["subtype", "feature"], how="left").sort_values(
        ["subtype", "log2fc", "score"], ascending=[True, False, False]
    ).reset_index(drop=True)


def choose_plot_points(df: pd.DataFrame, modality_label: str, log2fc_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    positive = df.loc[df["log2fc"] > 0].copy()
    positive["modality_label"] = modality_label
    highlight = positive.loc[positive["log2fc"] > log2fc_threshold].copy()

    rows: list[dict[str, object]] = []
    for group, group_df in positive.groupby("subtype", observed=True):
        picked = highlight.loc[highlight["subtype"].astype(str).eq(str(group))].copy()
        rows.append(
            {
                "subtype": str(group),
                "modality": modality_label,
                "positive_points_total": int(group_df.shape[0]),
                "highlight_points": int(picked.shape[0]),
                "log2fc_gt_threshold_points": int((group_df["log2fc"] > log2fc_threshold).sum()),
                "highlight_threshold": float(log2fc_threshold),
            }
        )
    return highlight, pd.DataFrame(rows)


def palette_for(groups: list[str], major_value: str) -> dict[str, str]:
    palette = GROUP_COLOR_PALETTES.get(major_value, GROUP_COLOR_PALETTES["Mixed"])
    return {group: palette[i % len(palette)] for i, group in enumerate(groups)}


def build_annotation_layout(
    plotted: pd.DataFrame,
    x_lookup: dict[str, float],
    top_labels_per_group: int,
    y_lim: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    x_offsets = {1: [0.0], 2: [-0.10, 0.10], 3: [-0.14, 0.0, 0.14]}
    for group, group_df in plotted.groupby("subtype", observed=True):
        base_x = x_lookup[str(group)]
        for modality_label, mod_df in group_df.groupby("modality_label", observed=True):
            top = mod_df.sort_values(["log2fc", "score"], ascending=[False, False]).head(top_labels_per_group).reset_index(drop=True)
            if top.empty:
                continue
            offsets = x_offsets.get(len(top), np.linspace(-0.14, 0.14, len(top)).tolist())
            sign = 1 if str(modality_label) == "ST" else -1
            for rank, (_, row) in enumerate(top.iterrows()):
                display_y = float(row["display_y"])
                text_y = display_y + sign * (0.55 + LABEL_MIN_VERTICAL_GAP * rank)
                text_y = min(text_y, y_lim - 2.0) if sign > 0 else max(text_y, -y_lim + 1.7)
                text_x = base_x + float(offsets[rank])
                if abs(text_x - float(row["plot_x"])) < 0.02:
                    text_x += 0.05 if rank % 2 == 0 else -0.05
                rows.append(
                    {
                        "group": str(group),
                        "modality_label": str(modality_label),
                        "label": display_feature_label(row["feature"], str(modality_label)),
                        "anchor_x": float(row["plot_x"]),
                        "anchor_y": display_y,
                        "text_x": text_x,
                        "text_y": text_y,
                        "ha": "center",
                        "va": "bottom" if sign > 0 else "top",
                        "sign": sign,
                    }
                )

    layout = pd.DataFrame(rows)
    if layout.empty:
        return layout

    adjusted_parts: list[pd.DataFrame] = []
    for (_group, _modality, sign), part in layout.groupby(["group", "modality_label", "sign"], observed=True):
        part = part.sort_values("text_y", ascending=(sign < 0)).reset_index(drop=True)
        adjusted_y = part["text_y"].astype(float).to_numpy(copy=True)
        for i in range(1, len(adjusted_y)):
            if sign > 0:
                adjusted_y[i] = max(adjusted_y[i], adjusted_y[i - 1] + LABEL_MIN_VERTICAL_GAP)
            else:
                adjusted_y[i] = min(adjusted_y[i], adjusted_y[i - 1] - LABEL_MIN_VERTICAL_GAP)
        if sign > 0:
            upper_cap = y_lim - 2.0
            overflow = adjusted_y[-1] - upper_cap
            if overflow > 0:
                adjusted_y = adjusted_y - overflow
        else:
            lower_cap = -y_lim + 1.8
            overflow = lower_cap - adjusted_y[-1]
            if overflow > 0:
                adjusted_y = adjusted_y + overflow
        part["text_y"] = adjusted_y
        adjusted_parts.append(part)

    return pd.concat(adjusted_parts, ignore_index=True)


def annotate_top_features(
    ax: plt.Axes,
    plotted: pd.DataFrame,
    x_lookup: dict[str, float],
    top_labels_per_group: int,
    y_lim: float,
) -> None:
    layout = build_annotation_layout(plotted, x_lookup, top_labels_per_group, y_lim)
    for row in layout.itertuples(index=False):
        ax.annotate(
            row.label,
            xy=(float(row.anchor_x), float(row.anchor_y)),
            xytext=(float(row.text_x), float(row.text_y)),
            textcoords="data",
            ha=str(row.ha),
            va=str(row.va),
            fontsize=FEATURE_LABEL_FONTSIZE,
            color="#000000",
            arrowprops=dict(arrowstyle="-", color="#000000", lw=0.75, shrinkA=0, shrinkB=0, alpha=0.85),
        )


def make_plot(
    group_info: pd.DataFrame,
    st_positive: pd.DataFrame,
    sm_positive: pd.DataFrame,
    st_highlight: pd.DataFrame,
    sm_highlight: pd.DataFrame,
    colors: dict[str, str],
    output_png: Path,
    output_pdf: Path,
    plotted_table: Path,
    seed: int,
    st_log2fc_threshold: float,
    sm_log2fc_threshold: float,
    top_labels_per_group: int,
) -> None:
    rng = np.random.default_rng(seed)
    groups = group_info["subtype"].astype(str).tolist()
    x_positions = np.arange(len(groups), dtype=float) if len(groups) > 1 else np.array([0.0], dtype=float)
    x_lookup = {group: float(x) for group, x in zip(groups, x_positions)}

    def add_coords(df: pd.DataFrame, mirrored: bool) -> pd.DataFrame:
        out = df.copy()
        if out.empty:
            out["plot_x"] = []
            out["display_y"] = []
            return out
        out["plot_x"] = out["subtype"].astype(str).map(x_lookup).astype(float) + rng.uniform(-0.16, 0.16, size=len(out))
        out["display_y"] = -out["log2fc"].astype(float) if mirrored else out["log2fc"].astype(float)
        return out

    st_positive = add_coords(st_positive, mirrored=False)
    sm_positive = add_coords(sm_positive, mirrored=True)
    st_highlight = add_coords(st_highlight, mirrored=False)
    sm_highlight = add_coords(sm_highlight, mirrored=True)

    plotted = pd.concat(
        [
            st_positive.assign(plot_role="background"),
            sm_positive.assign(plot_role="background"),
            st_highlight.assign(plot_role="highlight"),
            sm_highlight.assign(plot_role="highlight"),
        ],
        ignore_index=True,
    )
    plotted.to_csv(plotted_table, index=False)

    fig, ax = plt.subplots(figsize=(13.5, 8.8))
    ax.axhline(0.0, color="#4d4d4d", lw=1.0)
    ax.scatter(st_positive["plot_x"], st_positive["display_y"], s=8, color="#c9ced6", alpha=0.55, linewidths=0, rasterized=True)
    ax.scatter(sm_positive["plot_x"], sm_positive["display_y"], s=8, color="#c9ced6", alpha=0.55, linewidths=0, rasterized=True)

    for group in groups:
        color = colors[group]
        st_mask = st_highlight["subtype"].astype(str).eq(group)
        sm_mask = sm_highlight["subtype"].astype(str).eq(group)
        ax.scatter(st_highlight.loc[st_mask, "plot_x"], st_highlight.loc[st_mask, "display_y"], s=13, color=color, alpha=0.95, linewidths=0, rasterized=True)
        ax.scatter(sm_highlight.loc[sm_mask, "plot_x"], sm_highlight.loc[sm_mask, "display_y"], s=13, color=color, alpha=0.95, linewidths=0, rasterized=True)

    max_gene = float(st_positive["display_y"].max()) if not st_positive.empty else 1.0
    max_met = float(np.abs(sm_positive["display_y"]).max()) if not sm_positive.empty else 1.0
    y_lim = max(max_gene, max_met, 1.0) + 2.0
    ax.set_ylim(-y_lim, y_lim)
    annotate_top_features(ax, st_highlight, x_lookup, top_labels_per_group, y_lim)
    annotate_top_features(ax, sm_highlight, x_lookup, top_labels_per_group, y_lim)
    ax.set_xlim(float(x_positions.min()) - 0.45, float(x_positions.max()) + 0.45)
    ax.set_xticks([x_lookup[group] for group in groups])
    ax.set_xticklabels([f"{row.display_label}" for row in group_info.itertuples(index=False)], rotation=35, ha="right", fontsize=TICK_FONTSIZE)
    for tick in ax.get_xticklabels():
        tick.set_color("#000000")

    ax.set_ylabel("log2(fold change)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{abs(value):g}"))
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax.set_title("Upregulated gene expression and metabolites", fontsize=TITLE_FONTSIZE, pad=16)
    ax.text(
        0.99,
        0.98,
        (
            f"Color = genes log2FC > {st_log2fc_threshold:g}, metabolites log2FC > {sm_log2fc_threshold:g}\n"
            "Grey = filtered positive log2FC points"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=ANNOTATION_TEXT_FONTSIZE,
        color="#000000",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_png, dpi=240, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_major_class_subtype_de_log2fc(
    sample: str,
    input_h5ad: Path,
    output_dir: Path,
    cluster_label_h5ad: Path | None = None,
    feature_source_h5ad: Path | None = None,
    cluster_key: str = CLUSTER_KEY,
    major_value: str = "Imm",
    norm_layer: str = NORM_LAYER,
    log2fc_threshold: float = LOG2FC_THRESHOLD,
    sm_log2fc_threshold: float = SM_LOG2FC_THRESHOLD,
    top_labels_per_group: int = TOP_LABELS_PER_GROUP,
    min_subtype_spots: int = MIN_SUBTYPE_SPOTS,
    seed: int = RANDOM_SEED,
    margin_mixed: float = MARGIN_MIXED,
    margin_high: float = MARGIN_HIGH,
) -> dict[str, object]:
    major_value = str(major_value)
    cluster_label_h5ad = Path(cluster_label_h5ad or input_h5ad)
    feature_source_h5ad = Path(feature_source_h5ad or input_h5ad)
    fig_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{sample}_{MAJOR_KEY}_{major_value}_{SUBTYPE_KEY}"
    group_table = table_dir / f"{prefix}_selected_subtypes.csv"
    st_full_table = table_dir / f"{prefix}_st_de_full.csv"
    sm_full_table = table_dir / f"{prefix}_sm_de_full.csv"
    plotted_table = table_dir / f"{prefix}_st_sm_de_plotted.csv"
    diag_table = table_dir / f"{prefix}_plot_diagnostics.csv"
    annotation_table = table_dir / f"{sample}_single_sample_marker_annotation.csv"
    output_png = fig_dir / f"{prefix}_st_sm_log2fc.png"
    output_pdf = fig_dir / f"{prefix}_st_sm_log2fc.pdf"
    summary_json = output_dir / f"{prefix}_summary.json"

    adata = load_adata(cluster_label_h5ad, feature_source_h5ad, cluster_key)
    adata, annotation_df = annotate_clusters_single_sample(
        adata,
        cluster_key,
        norm_layer,
        major_value,
        margin_mixed,
        margin_high,
    )
    annotation_df.to_csv(annotation_table, index=False)

    selected, selected_groups, group_info = subset_and_select_subtypes(
        adata,
        major_value,
        min_subtype_spots,
        cluster_key,
    )
    group_info.to_csv(group_table, index=False)

    if len(selected_groups) < 2:
        summary = {
            "sample": sample,
            "input_h5ad": str(input_h5ad),
            "cluster_label_h5ad": str(cluster_label_h5ad),
            "feature_source_h5ad": str(feature_source_h5ad),
            "cluster_key": cluster_key,
            "major_key": MAJOR_KEY,
            "major_value": major_value,
            "subtype_key": SUBTYPE_KEY,
            "selected_subtypes": selected_groups,
            "n_selected_spots": int(selected.n_obs),
            "norm_layer": norm_layer,
            "margin_mixed": margin_mixed,
            "margin_high": margin_high,
            "status": "skipped",
            "reason": f"Need at least two {major_value} subtypes with min_spots={min_subtype_spots}.",
            "figure_png": None,
            "figure_pdf": None,
            "tables": {
                "group_table": str(group_table),
                "annotation_table": str(annotation_table),
                "st_full": None,
                "sm_full": None,
                "plotted": None,
                "diagnostics": None,
            },
        }
        summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"figure": None, "summary": str(summary_json), "status": "skipped"}, ensure_ascii=False, indent=2))
        return summary

    st_de = build_de_table(selected, "ST", selected_groups, norm_layer)
    sm_de = build_de_table(selected, "SM", selected_groups, norm_layer)
    st_de.to_csv(st_full_table, index=False)
    sm_de.to_csv(sm_full_table, index=False)

    st_positive = st_de.loc[st_de["log2fc"] > 0].copy()
    st_positive["modality_label"] = "ST"
    sm_positive = sm_de.loc[sm_de["log2fc"] > 0].copy()
    sm_positive["modality_label"] = "SM"
    st_highlight, st_diag = choose_plot_points(st_de, "ST", log2fc_threshold)
    sm_highlight, sm_diag = choose_plot_points(sm_de, "SM", sm_log2fc_threshold)
    pd.concat([st_diag, sm_diag], ignore_index=True).to_csv(diag_table, index=False)

    colors = palette_for(selected_groups, major_value)
    make_plot(
        group_info=group_info,
        st_positive=st_positive,
        sm_positive=sm_positive,
        st_highlight=st_highlight,
        sm_highlight=sm_highlight,
        colors=colors,
        output_png=output_png,
        output_pdf=output_pdf,
        plotted_table=plotted_table,
        seed=seed,
        st_log2fc_threshold=log2fc_threshold,
        sm_log2fc_threshold=sm_log2fc_threshold,
        top_labels_per_group=top_labels_per_group,
    )

    summary = {
        "sample": sample,
        "input_h5ad": str(input_h5ad),
        "cluster_label_h5ad": str(cluster_label_h5ad),
        "feature_source_h5ad": str(feature_source_h5ad),
        "cluster_key": cluster_key,
        "major_key": MAJOR_KEY,
        "major_value": major_value,
        "subtype_key": SUBTYPE_KEY,
        "selected_subtypes": selected_groups,
        "n_selected_spots": int(selected.n_obs),
        "norm_layer": norm_layer,
        "margin_mixed": margin_mixed,
        "margin_high": margin_high,
        "st_log2fc_threshold": log2fc_threshold,
        "sm_log2fc_threshold": sm_log2fc_threshold,
        "log2fc_eps": LOG2FC_EPS,
        "figure_png": str(output_png),
        "figure_pdf": str(output_pdf),
        "tables": {
            "group_table": str(group_table),
            "annotation_table": str(annotation_table),
            "st_full": str(st_full_table),
            "sm_full": str(sm_full_table),
            "plotted": str(plotted_table),
            "diagnostics": str(diag_table),
        },
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"figure": str(output_png), "summary": str(summary_json)}, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    plot_major_class_subtype_de_log2fc(
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
