from __future__ import annotations

import argparse
import gzip
import json
import re
import subprocess
import textwrap
import urllib.request
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import hypergeom, norm


ROOT = Path("/data/user/hesy/projects/SpatialMETA")
DATA_DIR = Path(
    "/bigdat2/user/hesy/spatialmeta/SpatialMETA/smart/SMART_data/Mouse_Brain_E18_S1"
)
RUN_DIR = ROOT / "SpaDTA_718/runs/ATAC/Mouse_Brain_E18_S1"
OUTPUT_DIR = ROOT / "SpaDTA_718/runs/atac_downstream/fig4d"
GTF_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/"
    "release_M25/gencode.vM25.annotation.gtf.gz"
)
GTF_NAME = "gencode.vM25.annotation.gtf.gz"
GENE_SET_LIBRARIES = {
    "GO:BP": "GO_Biological_Process_2023",
    "REAC": "Reactome_2022",
}

RNA_QUERY_LIMIT = 250
ATAC_PEAK_LIMIT = 750
MAX_PEAK_TSS_DISTANCE = 100_000
TOP_TERMS_PER_CLUSTER = 3
FDR_THRESHOLD = 0.05
TARGET_CLUSTER = "8"
TOP_RNA_TERMS = 5
TOP_ATAC_TERMS = 4
PURPLE_CMAP = LinearSegmentedColormap.from_list(
    "grey_purple", ["#d8d5dd", "#c8b7eb", "#8e61c0"]
)


def natural_key(value: str) -> tuple[object, ...]:
    return tuple(int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(value)))


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


def load_labels_and_regions() -> tuple[pd.Index, np.ndarray, pd.DataFrame, dict[str, str]]:
    spot_ids = pd.Index(
        pd.read_csv(
            RUN_DIR / "saved_epoch_embeddings/epoch_0300/spot_ids.csv", dtype=str
        )["spot_id"].astype(str)
    )
    labels = pd.read_csv(
        RUN_DIR / "final_protocol/epoch_0300/spot_labels.csv", dtype=str
    )["mclust_label"].astype(str).to_numpy()
    if len(spot_ids) != len(labels):
        raise ValueError("SpaDTA spot IDs and labels have different lengths")

    annotation = pd.read_csv(DATA_DIR / "anno.csv", dtype=str).set_index("barcode")
    truth = annotation.loc[spot_ids, "cluster"].astype(str).to_numpy()
    rows = []
    cluster_to_region: dict[str, str] = {}
    for cluster in sorted(pd.unique(labels), key=natural_key):
        mask = labels == cluster
        counts = pd.Series(truth[mask]).value_counts()
        region = str(counts.index[0])
        cluster_to_region[cluster] = region
        rows.append(
            {
                "spadta_cluster": cluster,
                "majority_region": region,
                "cluster_spots": int(mask.sum()),
                "majority_spots": int(counts.iloc[0]),
                "region_purity": float(counts.iloc[0] / mask.sum()),
            }
        )
    return spot_ids, labels, pd.DataFrame(rows), cluster_to_region


def load_aligned_matrix(filename: str, spot_ids: pd.Index) -> tuple[sparse.csr_matrix, np.ndarray]:
    adata = ad.read_h5ad(DATA_DIR / filename)
    adata.obs_names = adata.obs_names.astype(str)
    positions = adata.obs_names.get_indexer(spot_ids)
    if (positions < 0).any():
        raise ValueError(f"{filename} does not contain all SpaDTA spots")
    matrix = adata.X[positions].tocsr().astype(np.float32)
    features = adata.var_names.astype(str).to_numpy()
    return matrix, features


def normalize_log1p(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    library_size = np.asarray(matrix.sum(axis=1)).ravel()
    scale = np.divide(
        1e4,
        library_size,
        out=np.zeros_like(library_size, dtype=np.float32),
        where=library_size > 0,
    )
    normalized = sparse.diags(scale, format="csr") @ matrix
    normalized.data = np.log1p(normalized.data)
    return normalized.tocsr()


def rna_differential(matrix: sparse.csr_matrix, features: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    matrix = normalize_log1p(matrix)
    matrix_sq = matrix.copy()
    matrix_sq.data **= 2
    total_sum = np.asarray(matrix.sum(axis=0)).ravel()
    total_sq = np.asarray(matrix_sq.sum(axis=0)).ravel()
    total_detected = np.asarray(matrix.getnnz(axis=0)).ravel()
    rows = []

    duplicate = pd.Index(features).duplicated(keep="first")
    excluded = np.array([
        bool(re.match(r"^(mt-|Rpl\d|Rps\d)", gene, flags=re.IGNORECASE))
        for gene in features
    ])
    eligible = (~duplicate) & (~excluded) & (total_detected >= 10)

    for cluster in sorted(pd.unique(labels), key=natural_key):
        mask = labels == cluster
        n_in = int(mask.sum())
        n_out = int((~mask).sum())
        sum_in = np.asarray(matrix[mask].sum(axis=0)).ravel()
        sq_in = np.asarray(matrix_sq[mask].sum(axis=0)).ravel()
        sum_out = total_sum - sum_in
        sq_out = total_sq - sq_in
        mean_in = sum_in / n_in
        mean_out = sum_out / n_out
        var_in = np.maximum((sq_in - sum_in**2 / n_in) / max(n_in - 1, 1), 0)
        var_out = np.maximum((sq_out - sum_out**2 / n_out) / max(n_out - 1, 1), 0)
        standard_error = np.sqrt(var_in / n_in + var_out / n_out)
        z_score = np.divide(
            mean_in - mean_out,
            standard_error,
            out=np.zeros_like(mean_in),
            where=standard_error > 0,
        )
        p_value = 2 * norm.sf(np.abs(z_score))
        p_value[~eligible] = 1.0
        fdr = bh_adjust(p_value)
        detected_in = np.asarray(matrix[mask].getnnz(axis=0)).ravel()
        detected_out = total_detected - detected_in
        idx = np.flatnonzero(eligible)
        frame = pd.DataFrame(
            {
                "spadta_cluster": cluster,
                "feature": features[idx],
                "mean_in": mean_in[idx],
                "mean_out": mean_out[idx],
                "mean_difference": (mean_in - mean_out)[idx],
                "detected_fraction_in": detected_in[idx] / n_in,
                "detected_fraction_out": detected_out[idx] / n_out,
                "z_score": z_score[idx],
                "p_value": p_value[idx],
                "fdr": fdr[idx],
            }
        )
        rows.append(frame.sort_values(["fdr", "mean_difference"], ascending=[True, False]))
    return pd.concat(rows, ignore_index=True)


def atac_differential(matrix: sparse.csr_matrix, features: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    print("ATAC differential: binarizing matrix", flush=True)
    binary = matrix.copy()
    binary.data = np.ones_like(binary.data)
    total_detected = np.asarray(binary.getnnz(axis=0)).ravel()
    eligible = total_detected >= 10
    rows = []
    clusters = sorted(pd.unique(labels), key=natural_key)
    membership = sparse.csr_matrix(
        np.vstack([(labels == cluster).astype(np.float32) for cluster in clusters])
    )
    print("ATAC differential: aggregating all clusters", flush=True)
    detected_by_cluster = np.asarray((membership @ binary).todense())
    print("ATAC differential: building ranked tables", flush=True)
    for cluster_index, cluster in enumerate(clusters):
        n_in = int((labels == cluster).sum())
        n_out = int(len(labels) - n_in)
        detected_in = detected_by_cluster[cluster_index]
        detected_out = total_detected - detected_in
        fraction_in = detected_in / n_in
        fraction_out = detected_out / n_out
        pooled = total_detected / len(labels)
        standard_error = np.sqrt(pooled * (1 - pooled) * (1 / n_in + 1 / n_out))
        z_score = np.divide(
            fraction_in - fraction_out,
            standard_error,
            out=np.zeros_like(fraction_in, dtype=float),
            where=standard_error > 0,
        )
        p_value = 2 * norm.sf(np.abs(z_score))
        p_value[~eligible] = 1.0
        fdr = bh_adjust(p_value)
        candidate = eligible & (fraction_in > fraction_out)
        idx = np.flatnonzero(candidate)
        if len(idx) > 10_000:
            priority = np.lexsort((-fraction_in[idx] + fraction_out[idx], fdr[idx]))
            idx = idx[priority[:10_000]]
        frame = pd.DataFrame(
            {
                "spadta_cluster": cluster,
                "feature": features[idx],
                "detected_fraction_in": fraction_in[idx],
                "detected_fraction_out": fraction_out[idx],
                "fraction_difference": (fraction_in - fraction_out)[idx],
                "z_score": z_score[idx],
                "p_value": p_value[idx],
                "fdr": fdr[idx],
            }
        )
        rows.append(frame.sort_values(["fdr", "fraction_difference"], ascending=[True, False]))
        print(f"ATAC differential: cluster {cluster} complete ({len(frame)} retained peaks)", flush=True)
    return pd.concat(rows, ignore_index=True)


def download_gtf(path: Path) -> None:
    if path.exists() and path.stat().st_size > 1_000_000:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {GTF_URL}")
    urllib.request.urlretrieve(GTF_URL, path)


def parse_gtf_tss(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    records: dict[str, list[tuple[int, str]]] = {}
    gene_name_pattern = re.compile(r'gene_name "([^"]+)"')
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "gene":
                continue
            match = gene_name_pattern.search(fields[8])
            if match is None:
                continue
            chrom, start, end, strand = fields[0], int(fields[3]), int(fields[4]), fields[6]
            tss = start - 1 if strand == "+" else end - 1
            records.setdefault(chrom, []).append((tss, match.group(1)))
    output = {}
    for chrom, entries in records.items():
        entries.sort(key=lambda item: item[0])
        output[chrom] = (
            np.asarray([item[0] for item in entries], dtype=np.int64),
            np.asarray([item[1] for item in entries], dtype=object),
        )
    return output


def link_peaks_to_nearest_gene(peaks: pd.Series, tss_by_chrom: dict[str, tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    rows = []
    peak_pattern = re.compile(r"^([^:]+):(\d+)-(\d+)$")
    for peak in peaks.astype(str).drop_duplicates():
        match = peak_pattern.match(peak)
        if match is None or match.group(1) not in tss_by_chrom:
            continue
        chrom, start, end = match.group(1), int(match.group(2)), int(match.group(3))
        midpoint = (start + end) // 2
        positions, genes = tss_by_chrom[chrom]
        insertion = int(np.searchsorted(positions, midpoint))
        candidates = [i for i in (insertion - 1, insertion) if 0 <= i < len(positions)]
        nearest = min(candidates, key=lambda i: abs(int(positions[i]) - midpoint))
        distance = abs(int(positions[nearest]) - midpoint)
        if distance <= MAX_PEAK_TSS_DISTANCE:
            rows.append(
                {
                    "feature": peak,
                    "chromosome": chrom,
                    "peak_midpoint": midpoint,
                    "nearest_gene": str(genes[nearest]),
                    "tss_distance_bp": distance,
                }
            )
    return pd.DataFrame(rows)


def load_gene_sets(reference_dir: Path) -> dict[str, tuple[str, set[str]]]:
    reference_dir.mkdir(parents=True, exist_ok=True)
    gene_sets: dict[str, tuple[str, set[str]]] = {}
    for source, library in GENE_SET_LIBRARIES.items():
        path = reference_dir / f"{library}.gmt"
        if not path.exists() or path.stat().st_size < 1_000:
            url = (
                "https://maayanlab.cloud/Enrichr/geneSetLibrary"
                f"?mode=text&libraryName={library}"
            )
            subprocess.run(
                ["curl", "-sS", "--fail", "--max-time", "120", "-o", str(path), url],
                check=True,
            )
        with path.open() as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 3:
                    continue
                term = fields[0].strip()
                genes = {gene.upper() for gene in fields[2:] if gene.strip()}
                if genes:
                    gene_sets[f"{source}|{term}"] = (source, genes)
    return gene_sets


def local_enrichment(
    query: list[str],
    background: list[str],
    gene_sets: dict[str, tuple[str, set[str]]],
) -> pd.DataFrame:
    background_set = {gene.upper() for gene in background}
    query_set = {gene.upper() for gene in query}.intersection(background_set)
    population_size = len(background_set)
    query_size = len(query_set)
    rows = []
    for key, (source, library_genes) in gene_sets.items():
        term = key.split("|", 1)[1]
        term_genes = library_genes.intersection(background_set)
        overlap_genes = sorted(query_set.intersection(term_genes))
        overlap = len(overlap_genes)
        if overlap < 2 or len(term_genes) < 3:
            continue
        p_value = float(
            hypergeom.sf(overlap - 1, population_size, len(term_genes), query_size)
        )
        rows.append(
            {
                "source": source,
                "term_id": "",
                "term": term,
                "p_value": p_value,
                "overlap_count": overlap,
                "term_size": len(term_genes),
                "query_size": query_size,
                "gene_ratio": overlap / max(query_size, 1),
                "overlap_genes": ";".join(overlap_genes),
            }
        )
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    output["fdr"] = bh_adjust(output["p_value"].to_numpy())
    return output.sort_values(["fdr", "gene_ratio"], ascending=[True, False]).reset_index(drop=True)


def run_enrichment(
    rna_de: pd.DataFrame,
    atac_de: pd.DataFrame,
    peak_links: pd.DataFrame,
    clusters: list[str],
    gene_sets: dict[str, tuple[str, set[str]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rna_background = sorted(rna_de["feature"].astype(str).unique())
    atac_background = sorted(peak_links["nearest_gene"].astype(str).unique())
    rna_results = []
    atac_results = []
    for cluster in clusters:
        rna_query = (
            rna_de.loc[
                rna_de["spadta_cluster"].eq(cluster)
                & rna_de["fdr"].lt(FDR_THRESHOLD)
                & rna_de["mean_difference"].gt(0)
                & rna_de["detected_fraction_in"].ge(0.10)
            ]
            .sort_values(["fdr", "mean_difference"], ascending=[True, False])
            .head(RNA_QUERY_LIMIT)["feature"]
            .astype(str)
            .tolist()
        )
        if len(rna_query) >= 5:
            enrich = local_enrichment(rna_query, rna_background, gene_sets)
            if not enrich.empty:
                enrich.insert(0, "spadta_cluster", cluster)
                rna_results.append(enrich)

        peak_query = (
            atac_de.loc[
                atac_de["spadta_cluster"].eq(cluster)
                & atac_de["fdr"].lt(FDR_THRESHOLD)
                & atac_de["fraction_difference"].gt(0)
                & atac_de["detected_fraction_in"].ge(0.05)
            ]
            .sort_values(["fdr", "fraction_difference"], ascending=[True, False])
            .head(ATAC_PEAK_LIMIT)
            .merge(peak_links, on="feature", how="inner")["nearest_gene"]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        if len(peak_query) >= 5:
            enrich = local_enrichment(peak_query, atac_background, gene_sets)
            if not enrich.empty:
                enrich.insert(0, "spadta_cluster", cluster)
                atac_results.append(enrich)
    rna = pd.concat(rna_results, ignore_index=True) if rna_results else pd.DataFrame()
    atac = pd.concat(atac_results, ignore_index=True) if atac_results else pd.DataFrame()
    return rna, atac


def wrap(text: str, width: int = 35) -> str:
    return textwrap.fill(str(text), width=width)


def term_display(term: str) -> str:
    return re.sub(r"\s+(?:\(GO:\d+\)|R-HSA-\d+)$", "", str(term)).strip()


def stable_norm(values: pd.Series) -> Normalize:
    vmin = float(values.min())
    vmax = float(values.max())
    if np.isclose(vmin, vmax):
        vmin = 0.0
        vmax = max(vmax, 1e-8)
    return Normalize(vmin=vmin, vmax=vmax)


def plot_joint_enrichment(
    rna_enrichment: pd.DataFrame,
    atac_enrichment: pd.DataFrame,
    output_stems: list[Path],
    region: str,
) -> None:
    rna_plot = (
        rna_enrichment.loc[rna_enrichment["fdr"].lt(FDR_THRESHOLD)]
        .sort_values(["fdr", "gene_ratio"], ascending=[True, False])
        .head(TOP_RNA_TERMS)
        .copy()
        .sort_values("p_value", ascending=False)
        .reset_index(drop=True)
    )
    atac_plot = (
        atac_enrichment.sort_values(["fdr", "gene_ratio"], ascending=[True, False])
        .head(TOP_ATAC_TERMS)
        .copy()
        .sort_values("overlap_count", ascending=True)
        .reset_index(drop=True)
    )
    if rna_plot.empty or atac_plot.empty:
        raise RuntimeError("RNA or ATAC enrichment returned no terms for cluster 8")

    rna_plot["log10_1_over_fdr"] = -np.log10(
        np.clip(rna_plot["fdr"].astype(float), 1e-300, None)
    )
    rna_plot["enrichment_score"] = -np.log10(
        np.clip(rna_plot["p_value"].astype(float), 1e-300, None)
    )
    rna_plot["query_gene_pct"] = rna_plot["gene_ratio"].astype(float) * 100.0
    atac_plot["log10_1_over_fdr"] = -np.log10(
        np.clip(atac_plot["fdr"].astype(float), 1e-300, None)
    )

    fig = plt.figure(figsize=(11.8, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.08, 0.92], wspace=0.56)
    ax_rna = fig.add_subplot(gs[0, 0])
    ax_atac = fig.add_subplot(gs[0, 1])

    rna_norm = stable_norm(rna_plot["log10_1_over_fdr"])
    scatter = ax_rna.scatter(
        rna_plot["enrichment_score"],
        np.arange(len(rna_plot)),
        s=np.clip(rna_plot["query_gene_pct"] * 30.0, 80, 280),
        c=rna_plot["log10_1_over_fdr"],
        cmap=PURPLE_CMAP,
        norm=rna_norm,
        edgecolors="none",
        zorder=3,
    )
    left_edge = max(0.0, float(rna_plot["enrichment_score"].min()) - 4.0)
    for y, x in zip(np.arange(len(rna_plot)), rna_plot["enrichment_score"]):
        ax_rna.hlines(y=y, xmin=left_edge, xmax=x, color="#dddddd", lw=1.1, zorder=1)
    ax_rna.set_yticks(np.arange(len(rna_plot)))
    ax_rna.set_yticklabels(
        [wrap(term_display(x), width=22) for x in rna_plot["term"]], fontsize=10
    )
    ax_rna.set_xlabel("-log10(p)", fontsize=11)
    ax_rna.set_xlim(left=left_edge)
    ax_rna.tick_params(axis="x", labelsize=10)
    ax_rna.tick_params(axis="y", length=0)
    ax_rna.spines["top"].set_visible(False)
    ax_rna.spines["right"].set_visible(False)

    legend_vals = sorted(set(np.round(rna_plot["query_gene_pct"], 1).tolist()))
    if len(legend_vals) > 3:
        idxs = np.linspace(0, len(legend_vals) - 1, num=3, dtype=int)
        legend_vals = [legend_vals[i] for i in idxs]
    handles = [
        ax_rna.scatter([], [], s=np.clip(v * 30.0, 80, 280), color="#9e9e9e")
        for v in legend_vals
    ]
    ax_rna.legend(
        handles,
        [f"{v:.1f}" for v in legend_vals],
        title="% Query genes",
        frameon=False,
        fontsize=9,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(-0.02, 1.23),
        ncol=max(1, len(legend_vals)),
        handletextpad=0.6,
        columnspacing=1.1,
    )
    cbar_rna = fig.colorbar(scatter, ax=ax_rna, fraction=0.052, pad=0.04)
    cbar_rna.ax.set_title("log10(1/FDR)", fontsize=9, pad=8)
    cbar_rna.ax.tick_params(labelsize=9)

    atac_norm = stable_norm(atac_plot["log10_1_over_fdr"])
    bar_colors = PURPLE_CMAP(atac_norm(atac_plot["log10_1_over_fdr"].to_numpy()))
    ax_atac.barh(
        np.arange(len(atac_plot)),
        atac_plot["overlap_count"].astype(float),
        color=bar_colors,
        edgecolor="none",
        height=0.56,
    )
    ax_atac.set_yticks(np.arange(len(atac_plot)))
    ax_atac.set_yticklabels(
        [wrap(term_display(x), width=18) for x in atac_plot["term"]], fontsize=10
    )
    ax_atac.set_xlabel("ATAC peak-linked genes in set", fontsize=11)
    ax_atac.tick_params(axis="x", labelsize=10)
    ax_atac.tick_params(axis="y", length=0)
    ax_atac.spines["top"].set_visible(False)
    ax_atac.spines["right"].set_visible(False)
    ax_atac.set_xlim(0, max(1.0, float(atac_plot["overlap_count"].max()) + 1.5))
    scalar_mappable = plt.cm.ScalarMappable(norm=atac_norm, cmap=PURPLE_CMAP)
    scalar_mappable.set_array([])
    cbar_atac = fig.colorbar(scalar_mappable, ax=ax_atac, fraction=0.052, pad=0.04)
    cbar_atac.ax.set_title("log10(1/FDR)", fontsize=9, pad=8)
    cbar_atac.ax.tick_params(labelsize=9)

    fig.suptitle(
        f"C{TARGET_CLUSTER} ({region}): RNA and ATAC peak-linked gene enrichment",
        fontsize=13,
        y=1.03,
    )
    fig.tight_layout()
    for output_stem in output_stems:
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight", format="svg")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Fig. 4d enrichment analyses for E18 SpaDTA clusters.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    spot_ids, labels, cluster_metadata, _ = load_labels_and_regions()
    target_metadata = cluster_metadata.loc[
        cluster_metadata["spadta_cluster"].astype(str).eq(TARGET_CLUSTER)
    ].copy()
    if target_metadata.empty:
        raise ValueError(f"Target SpaDTA cluster {TARGET_CLUSTER} is absent")
    target_metadata.to_csv(args.output_dir / "cluster_region_mapping.csv", index=False)
    rna_de_path = args.output_dir / "rna_cluster_differential.csv.gz"
    atac_de_path = args.output_dir / "atac_cluster_differential.csv.gz"
    peak_link_path = args.output_dir / "atac_peak_nearest_gene.csv.gz"
    rna_enrich_path = args.output_dir / "rna_go_reactome_enrichment.csv"
    atac_enrich_path = args.output_dir / "atac_peak_linked_go_reactome_enrichment.csv"

    if args.recompute or not rna_de_path.exists():
        rna_matrix, rna_features = load_aligned_matrix("adata_RNA.h5ad", spot_ids)
        rna_de = rna_differential(rna_matrix, rna_features, labels)
        rna_de.to_csv(rna_de_path, index=False, compression="gzip")
        del rna_matrix
    else:
        rna_de = pd.read_csv(rna_de_path, dtype={"spadta_cluster": str})

    if args.recompute or not atac_de_path.exists():
        print("Loading aligned ATAC matrix", flush=True)
        atac_matrix, atac_features = load_aligned_matrix("adata_ATAC.h5ad", spot_ids)
        print(f"Loaded ATAC matrix {atac_matrix.shape} with {atac_matrix.nnz} nonzero entries", flush=True)
        atac_de = atac_differential(atac_matrix, atac_features, labels)
        print("Saving ATAC differential table", flush=True)
        atac_de.to_csv(atac_de_path, index=False, compression="gzip")
        del atac_matrix
    else:
        atac_de = pd.read_csv(atac_de_path, dtype={"spadta_cluster": str})

    if args.recompute or not peak_link_path.exists():
        if "atac_features" not in locals():
            _, atac_features = load_aligned_matrix("adata_ATAC.h5ad", spot_ids)
        gtf_path = args.output_dir / "reference" / GTF_NAME
        download_gtf(gtf_path)
        peak_links = link_peaks_to_nearest_gene(pd.Series(atac_features), parse_gtf_tss(gtf_path))
        peak_links.to_csv(peak_link_path, index=False, compression="gzip")
    else:
        peak_links = pd.read_csv(peak_link_path)

    if args.recompute or not (rna_enrich_path.exists() and atac_enrich_path.exists()):
        gene_sets = load_gene_sets(args.output_dir / "reference")
        rna_enrich, atac_enrich = run_enrichment(
            rna_de,
            atac_de,
            peak_links,
            [TARGET_CLUSTER],
            gene_sets,
        )
        rna_enrich.to_csv(rna_enrich_path, index=False)
        atac_enrich.to_csv(atac_enrich_path, index=False)
    else:
        rna_enrich = pd.read_csv(rna_enrich_path, dtype={"spadta_cluster": str})
        atac_enrich = pd.read_csv(atac_enrich_path, dtype={"spadta_cluster": str})
        rna_enrich = rna_enrich.loc[
            rna_enrich["spadta_cluster"].astype(str).eq(TARGET_CLUSTER)
        ].copy()
        atac_enrich = atac_enrich.loc[
            atac_enrich["spadta_cluster"].astype(str).eq(TARGET_CLUSTER)
        ].copy()
        rna_enrich.to_csv(rna_enrich_path, index=False)
        atac_enrich.to_csv(atac_enrich_path, index=False)

    plot_joint_enrichment(
        rna_enrich,
        atac_enrich,
        [
            args.output_dir / "fig4d_cluster8_rna_atac_enrichment",
            args.output_dir / "fig4d_rna_go_reactome",
            args.output_dir / "fig4d_atac_peak_linked_go_reactome",
        ],
        str(target_metadata.iloc[0]["majority_region"]),
    )

    summary = {
        "sample": "Mouse_Brain_E18_S1",
        "target_cluster": TARGET_CLUSTER,
        "target_region": str(target_metadata.iloc[0]["majority_region"]),
        "rna_test": "Welch-style normal approximation on library-normalized log1p counts",
        "atac_test": "two-proportion z-test on peak detection",
        "multiple_testing": "Benjamini-Hochberg within each cluster and modality",
        "enrichment": "local hypergeometric GO:BP 2023 and Reactome 2022 with measured-feature custom background",
        "atac_annotation": f"nearest GENCODE vM25 gene TSS within {MAX_PEAK_TSS_DISTANCE} bp",
        "important_limitation": "ATAC enrichment is peak-linked gene enrichment, not sequence motif enrichment",
    }
    (args.output_dir / "fig4d_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
