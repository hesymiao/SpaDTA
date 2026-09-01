from __future__ import annotations

import concurrent.futures
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy import sparse
from scipy.spatial import cKDTree


@dataclass
class DownstreamConfig:
    sample_name: str
    result_h5ad: Path
    gt_h5ad: Path
    output_dir: Path
    loss_csv: Path | None = None
    full_metrics_csv: Path | None = None
    default_metrics_csv: Path | None = None
    defaultcluster_h5ad: Path | None = None
    cluster_key: str = "decalign_linear_clusters"
    recluster_key: str = "decalign_linear_clusters"
    embedding_key: str = "X_emb_decalign_linear"
    reconstruction_layer: str = "reconstruction_decalign_linear"
    normalized_layer: str = "normalized"
    contribution_st_key: str = "contribution_st_decalign_linear"
    contribution_sm_key: str = "contribution_sm_decalign_linear"
    embedding_homo_st_key: str = "X_emb_homo_st_decalign_linear"
    embedding_homo_sm_key: str = "X_emb_homo_sm_decalign_linear"
    embedding_homo_joint_key: str = "X_emb_homo_joint_decalign_linear"
    recluster_n_neighbors: int = 15
    recluster_resolution: float = 1.0
    recluster_random_seed: int = 0
    spatial_match_threshold: float = 5.0
    min_valid_spatial_matches: int = 10
    title: str | None = None
    clean_output: bool = False

    def normalize(self) -> DownstreamConfig:
        self.result_h5ad = Path(self.result_h5ad).expanduser()
        self.gt_h5ad = Path(self.gt_h5ad).expanduser()
        self.output_dir = Path(self.output_dir).expanduser()
        self.loss_csv = Path(self.loss_csv).expanduser() if self.loss_csv else None
        self.full_metrics_csv = Path(self.full_metrics_csv).expanduser() if self.full_metrics_csv else None
        self.default_metrics_csv = Path(self.default_metrics_csv).expanduser() if self.default_metrics_csv else None
        self.defaultcluster_h5ad = Path(self.defaultcluster_h5ad).expanduser() if self.defaultcluster_h5ad else self.result_h5ad
        if self.title is None:
            self.title = f"{self.sample_name} downstream plots using model clusters"
        return self

    @property
    def figure_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def table_dir(self) -> Path:
        return self.output_dir / "tables"

    @property
    def annotated_h5ad(self) -> Path:
        return self.output_dir / f"{self.sample_name}_downstream.h5ad"

    @property
    def summary_json(self) -> Path:
        return self.output_dir / "summary.json"

    @property
    def readme_path(self) -> Path:
        return self.output_dir / "README.txt"


def resolve_downstream_inputs(
    sample_name: str,
    run_root: str | Path,
    output_root: str | Path,
    config_name: str,
    gt_root: str | Path,
    processed_root: str | Path,
) -> dict[str, Path]:
    run_root = Path(run_root)
    output_root = Path(output_root)
    gt_root = Path(gt_root)

    config_dir = run_root / config_name
    result_h5ad = config_dir / f"{sample_name}_{config_name}.h5ad"
    loss_csv = config_dir / f"{sample_name}_{config_name}_loss.csv"
    full_metrics_csv = config_dir / f"{sample_name}_metrics_full.csv"
    gt_h5ad = gt_root / f"adata_joint_{sample_name}_hvf2800.h5ad"
    sample_output_dir = output_root / sample_name

    if not result_h5ad.exists():
        raise FileNotFoundError(f"Missing result h5ad: {result_h5ad}")
    if not gt_h5ad.exists():
        raise FileNotFoundError(f"Missing ground truth h5ad: {gt_h5ad}")

    return {
        "result_h5ad": result_h5ad,
        "loss_csv": loss_csv,
        "full_metrics_csv": full_metrics_csv,
        "gt_h5ad": gt_h5ad,
        "sample_output_dir": sample_output_dir,
    }


def ensure_dirs(cfg: DownstreamConfig) -> None:
    if cfg.clean_output and cfg.output_dir.exists():
        shutil.rmtree(cfg.output_dir)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.figure_dir.mkdir(parents=True, exist_ok=True)
    cfg.table_dir.mkdir(parents=True, exist_ok=True)


def sorted_categories(values: pd.Series) -> list[str]:
    categories = [str(x) for x in pd.Series(values).dropna().astype(str).unique().tolist()]

    def sort_key(value: str) -> tuple[int, object]:
        return (0, int(value)) if value.isdigit() else (1, value)

    return sorted(categories, key=sort_key)


def to_dense(values: np.ndarray | sparse.spmatrix) -> np.ndarray:
    return values.toarray() if sparse.issparse(values) else np.asarray(values)


def rowwise_cosine(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    x1 = np.asarray(x1, dtype=np.float32)
    x2 = np.asarray(x2, dtype=np.float32)
    dot = np.sum(x1 * x2, axis=1)
    denom = np.linalg.norm(x1, axis=1) * np.linalg.norm(x2, axis=1)
    cosine = np.divide(
        dot,
        denom,
        out=np.zeros_like(dot, dtype=np.float32),
        where=denom > 1e-12,
    )
    return np.clip(cosine, -1.0, 1.0)


def angular_similarity_from_cos(cosine: np.ndarray) -> np.ndarray:
    cosine = np.clip(np.asarray(cosine, dtype=np.float32), -1.0, 1.0)
    return 1.0 - (np.arccos(cosine) / np.pi)


def compute_spatialmeta_like_contributions(
    homo_st: np.ndarray,
    homo_sm: np.ndarray,
    homo_joint: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cos_st = rowwise_cosine(homo_st, homo_joint)
    cos_sm = rowwise_cosine(homo_sm, homo_joint)
    angular_st = angular_similarity_from_cos(cos_st)
    angular_sm = angular_similarity_from_cos(cos_sm)
    contribution_st = np.clip(angular_st - angular_sm + 0.5, 0.0, 1.0)
    contribution_sm = 1.0 - contribution_st
    return contribution_st.astype(np.float32), contribution_sm.astype(np.float32)


def load_defaultcluster_column(obs_names: pd.Index, cfg: DownstreamConfig) -> pd.Series | None:
    if cfg.defaultcluster_h5ad is None or not cfg.defaultcluster_h5ad.exists():
        return None

    adata = sc.read_h5ad(cfg.defaultcluster_h5ad, backed="r")
    try:
        if cfg.recluster_key not in adata.obs.columns:
            return None
        values = adata.obs[cfg.recluster_key].copy()
        values.index = values.index.astype(str)
        return values.reindex(obs_names.astype(str)).astype("string")
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()


def load_pred_adata(cfg: DownstreamConfig) -> sc.AnnData:
    adata = sc.read_h5ad(cfg.result_h5ad).copy()
    adata.obs_names = adata.obs_names.astype(str)

    if "name" in adata.var.columns:
        adata.var_names = adata.var["name"].astype(str).values
        adata.var_names_make_unique()

    adata.obsm["X_emb"] = np.asarray(adata.obsm[cfg.embedding_key]).copy()
    adata.obs["cluster_original"] = adata.obs[cfg.cluster_key].astype(str)

    defaultcluster_values = load_defaultcluster_column(adata.obs_names, cfg)
    if defaultcluster_values is None or defaultcluster_values.isna().all():
        sc.pp.neighbors(
            adata,
            use_rep="X_emb",
            n_neighbors=cfg.recluster_n_neighbors,
            random_state=cfg.recluster_random_seed,
        )
        sc.tl.umap(adata, random_state=cfg.recluster_random_seed)
        sc.tl.leiden(
            adata,
            resolution=cfg.recluster_resolution,
            key_added=cfg.recluster_key,
            random_state=cfg.recluster_random_seed,
        )
    else:
        adata.obs[cfg.recluster_key] = defaultcluster_values.astype(str).values
        sc.pp.neighbors(
            adata,
            use_rep="X_emb",
            n_neighbors=cfg.recluster_n_neighbors,
            random_state=cfg.recluster_random_seed,
        )
        sc.tl.umap(adata, random_state=cfg.recluster_random_seed)

    adata.obs["cluster"] = adata.obs[cfg.recluster_key].astype(str)
    adata.obs["cluster"] = pd.Categorical(
        adata.obs["cluster"],
        categories=sorted_categories(adata.obs["cluster"]),
        ordered=True,
    )

    has_contribution_parts = all(
        key in adata.obsm
        for key in (
            cfg.embedding_homo_st_key,
            cfg.embedding_homo_sm_key,
            cfg.embedding_homo_joint_key,
        )
    )
    if has_contribution_parts:
        contribution_st, contribution_sm = compute_spatialmeta_like_contributions(
            homo_st=np.asarray(adata.obsm[cfg.embedding_homo_st_key], dtype=np.float32),
            homo_sm=np.asarray(adata.obsm[cfg.embedding_homo_sm_key], dtype=np.float32),
            homo_joint=np.asarray(adata.obsm[cfg.embedding_homo_joint_key], dtype=np.float32),
        )
        adata.obs[cfg.contribution_st_key] = contribution_st
        adata.obs[cfg.contribution_sm_key] = contribution_sm
        adata.uns["contribution_method_decalign_linear"] = "spatialmeta_like_angular_similarity_to_homo_joint"

    adata.obs["contribution_st"] = adata.obs[cfg.contribution_st_key].astype(float)
    adata.obs["contribution_sm"] = adata.obs[cfg.contribution_sm_key].astype(float)
    adata.layers["reconstruction"] = adata.layers[cfg.reconstruction_layer].copy()
    return adata


def merge_pathology_annotation(adata: sc.AnnData, cfg: DownstreamConfig) -> tuple[sc.AnnData, dict[str, object]]:
    gt = sc.read_h5ad(cfg.gt_h5ad)
    try:
        gt.obs_names = gt.obs_names.astype(str)
        gt_key = "pathological_annotation" if "pathological_annotation" in gt.obs.columns else "annotation"
        if gt_key not in gt.obs.columns:
            raise KeyError("真值文件中不存在 pathological_annotation 或 annotation 列。")

        pred_spatial = np.asarray(adata.obsm["spatial"]) if "spatial" in adata.obsm else None
        gt_spatial = np.asarray(gt.obsm["spatial"]) if "spatial" in gt.obsm else None
        common_barcodes = gt.obs_names.intersection(adata.obs_names)
        barcode_coord_mean_distance = np.nan

        matched = pd.Series(pd.NA, index=adata.obs_names, dtype="object")
        pathology_match_distance = pd.Series(np.nan, index=adata.obs_names, dtype=float)

        if pred_spatial is None or gt_spatial is None:
            raise RuntimeError("缺少 spatial 坐标，无法执行空间匹配病理标签。")

        tree = cKDTree(gt_spatial)
        distances, indices = tree.query(pred_spatial, k=1)
        matched_mask = distances < cfg.spatial_match_threshold
        if int(np.sum(matched_mask)) <= cfg.min_valid_spatial_matches:
            raise RuntimeError("空间匹配病理标签失败，匹配到的 spot 太少。")

        matched.iloc[np.where(matched_mask)[0]] = gt.obs[gt_key].astype(object).to_numpy()[indices[matched_mask]]
        pathology_match_distance.iloc[:] = distances
        matching_mode = "spatial"

        matched_count = int(matched.notna().sum())
        adata.obs["pathological_annotation"] = matched
        adata.obs["pathology_match_distance"] = pathology_match_distance
        adata.obs["pathology_matched"] = matched.notna().to_numpy()
        adata.obs["pathology_matching_mode"] = matching_mode
        adata.obs["pathological_annotation_plot"] = matched.astype("string").fillna("unmatched")
        adata.obs["pathological_annotation_plot"] = pd.Categorical(
            adata.obs["pathological_annotation_plot"],
            categories=sorted_categories(adata.obs["pathological_annotation_plot"]),
            ordered=True,
        )

        metadata = {
            "matching_mode": matching_mode,
            "matched_count": matched_count,
            "unmatched_count": int(adata.n_obs - matched_count),
            "common_barcodes": int(len(common_barcodes)),
            "barcode_coord_mean_distance": barcode_coord_mean_distance,
            "spatial_match_threshold": cfg.spatial_match_threshold,
        }
        valid_distances = pathology_match_distance[matched.notna()]
        metadata["spatial_match_distance_mean"] = float(valid_distances.mean())
        metadata["spatial_match_distance_max"] = float(valid_distances.max())
        return adata, metadata
    finally:
        if getattr(gt, "file", None) is not None:
            gt.file.close()


def pick_high_variance_feature(adata: sc.AnnData, feature_type: str, cfg: DownstreamConfig) -> str:
    mask = adata.var["type"].astype(str).eq(feature_type).values
    names = adata.var_names[mask]
    matrix = adata[:, names].layers[cfg.normalized_layer]
    values = to_dense(matrix)
    variances = values.var(axis=0)
    return str(names[int(np.argmax(variances))])


def pick_gene_feature(adata: sc.AnnData, cfg: DownstreamConfig) -> str:
    preferred = ["Plp1", "Mbp", "Slc17a7", "Gad1", "Gfap", "Mog", "Snap25"]
    for feature in preferred:
        if feature in adata.var_names:
            return feature
    return pick_high_variance_feature(adata, "ST", cfg)


def pick_metabolite_feature(adata: sc.AnnData, cfg: DownstreamConfig) -> str:
    target = 137.0458
    sm_features = adata.var.loc[adata.var["type"].astype(str).eq("SM"), "name"].astype(str).tolist()

    candidates: list[tuple[str, float]] = []
    for feature in sm_features:
        try:
            candidates.append((feature, abs(float(feature) - target)))
        except ValueError:
            continue

    if candidates:
        candidates.sort(key=lambda item: item[1])
        if candidates[0][1] < 0.1:
            return candidates[0][0]

    return pick_high_variance_feature(adata, "SM", cfg)


def save_loss_curves(cfg: DownstreamConfig) -> list[str]:
    if cfg.loss_csv is None or (not cfg.loss_csv.exists()) or cfg.loss_csv.stat().st_size == 0:
        return []

    try:
        loss_df = pd.read_csv(cfg.loss_csv)
    except pd.errors.EmptyDataError:
        return []

    columns = loss_df.columns.tolist()
    if not columns:
        return []

    n_cols = 3
    n_rows = math.ceil(len(columns) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3.2 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for axis, column in zip(axes, columns):
        axis.plot(np.arange(1, len(loss_df) + 1), loss_df[column], linewidth=1.8, color="#1f77b4")
        axis.set_title(column.replace("epoch_", "").replace("_list", ""))
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25, linewidth=0.5)

    for axis in axes[len(columns):]:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(cfg.figure_dir / "loss_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return columns


def save_scanpy_plot(plot_func, output_path: Path, **kwargs) -> None:
    plt.close("all")
    plot_func(show=False, **kwargs)
    fig = plt.gcf()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def metric_absmax(values: np.ndarray) -> float:
    vmax = float(np.nanmax(np.abs(values)))
    return max(vmax, 1e-6)


def spot_size(n_obs: int) -> float:
    return float(np.clip(12000.0 / max(n_obs, 1), 3.0, 18.0))


def scatter_panel(
    axis: plt.Axes,
    coords: np.ndarray,
    values: np.ndarray,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    invert_y: bool,
    size: float,
) -> None:
    scatter = axis.scatter(
        coords[:, 0],
        coords[:, 1],
        c=values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=size,
        linewidths=0,
        rasterized=True,
    )
    if invert_y:
        axis.invert_yaxis()
    axis.set_title(title, fontsize=11)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    plt.colorbar(scatter, ax=axis, fraction=0.046, pad=0.02)


def save_cosine_overview(adata: sc.AnnData, cfg: DownstreamConfig) -> dict[str, float]:
    homo_st = np.asarray(adata.obsm[cfg.embedding_homo_st_key], dtype=np.float32)
    homo_sm = np.asarray(adata.obsm[cfg.embedding_homo_sm_key], dtype=np.float32)
    homo_joint = np.asarray(adata.obsm[cfg.embedding_homo_joint_key], dtype=np.float32)
    umap = np.asarray(adata.obsm["X_umap"], dtype=np.float32)
    spatial = np.asarray(adata.obsm["spatial"], dtype=np.float32)

    cos_st = rowwise_cosine(homo_st, homo_joint)
    cos_sm = rowwise_cosine(homo_sm, homo_joint)
    cos_delta = cos_st - cos_sm
    contribution_st = adata.obs["contribution_st"].astype(float).to_numpy()

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle(f"{cfg.sample_name} Modality Contribution Overview", fontsize=16, y=0.98)

    umap_size = spot_size(len(umap))
    spatial_size = spot_size(len(spatial))
    cos_st_abs = metric_absmax(cos_st)
    cos_sm_abs = metric_absmax(cos_sm)
    cos_delta_abs = metric_absmax(cos_delta)

    scatter_panel(axes[0, 0], umap, cos_st, "UMAP: Raw Cosine ST vs Joint", "coolwarm", -cos_st_abs, cos_st_abs, False, umap_size)
    scatter_panel(axes[0, 1], umap, cos_sm, "UMAP: Raw Cosine SM vs Joint", "coolwarm", -cos_sm_abs, cos_sm_abs, False, umap_size)
    scatter_panel(axes[0, 2], umap, cos_delta, "UMAP: Cosine Delta ST - SM", "coolwarm", -cos_delta_abs, cos_delta_abs, False, umap_size)
    scatter_panel(axes[0, 3], umap, contribution_st, "UMAP: ST Contribution", "viridis", 0.0, 1.0, False, umap_size)
    scatter_panel(axes[1, 0], spatial, cos_st, "Spatial: Raw Cosine ST vs Joint", "coolwarm", -cos_st_abs, cos_st_abs, True, spatial_size)
    scatter_panel(axes[1, 1], spatial, cos_sm, "Spatial: Raw Cosine SM vs Joint", "coolwarm", -cos_sm_abs, cos_sm_abs, True, spatial_size)
    scatter_panel(axes[1, 2], spatial, cos_delta, "Spatial: Cosine Delta ST - SM", "coolwarm", -cos_delta_abs, cos_delta_abs, True, spatial_size)
    scatter_panel(axes[1, 3], spatial, contribution_st, "Spatial: ST Contribution", "viridis", 0.0, 1.0, True, spatial_size)

    fig.tight_layout()
    fig.savefig(cfg.figure_dir / "modality_contribution_overview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "mean_raw_cosine_st_joint": float(np.mean(cos_st)),
        "mean_raw_cosine_sm_joint": float(np.mean(cos_sm)),
        "mean_raw_cosine_delta_st_minus_sm": float(np.mean(cos_delta)),
        "mean_st_contribution": float(np.mean(contribution_st)),
    }


def save_umap_plots(adata: sc.AnnData, cfg: DownstreamConfig) -> None:
    save_scanpy_plot(
        sc.pl.umap,
        cfg.figure_dir / "umap_clusters_pathology.png",
        adata=adata,
        color=["cluster", "pathological_annotation_plot"],
        ncols=2,
        frameon=False,
    )
    save_scanpy_plot(
        sc.pl.umap,
        cfg.figure_dir / "umap_contributions.png",
        adata=adata,
        color=["contribution_st", "contribution_sm"],
        ncols=2,
        frameon=False,
        color_map="viridis",
    )


def save_spatial_plots(adata: sc.AnnData, gene_feature: str, metabolite_feature: str, cfg: DownstreamConfig) -> None:
    contribution_cmap = LinearSegmentedColormap.from_list(
        "contribution_map",
        ["#2ec4b6", "#ffffff", "#ff9f1c"],
    )

    save_scanpy_plot(
        sc.pl.spatial,
        cfg.figure_dir / "spatial_clusters_pathology.png",
        adata=adata,
        img_key="hires",
        color=["cluster", "pathological_annotation_plot"],
        ncols=2,
        alpha_img=0.7,
        size=1.4,
    )
    save_scanpy_plot(
        sc.pl.spatial,
        cfg.figure_dir / "spatial_normalized_markers.png",
        adata=adata,
        img_key="hires",
        color=[gene_feature, metabolite_feature],
        layer=cfg.normalized_layer,
        ncols=2,
        color_map="vlag",
        alpha_img=0.15,
        size=1.4,
        vmin="p1",
        vmax="p99",
    )
    save_scanpy_plot(
        sc.pl.spatial,
        cfg.figure_dir / "spatial_reconstruction_markers.png",
        adata=adata,
        img_key="hires",
        color=[gene_feature, metabolite_feature],
        layer="reconstruction",
        ncols=2,
        color_map="vlag",
        alpha_img=0.15,
        size=1.4,
        vmin="p1",
        vmax="p99",
    )
    save_scanpy_plot(
        sc.pl.spatial,
        cfg.figure_dir / "spatial_contributions.png",
        adata=adata,
        img_key="hires",
        color=["contribution_st", "contribution_sm"],
        ncols=2,
        color_map=contribution_cmap,
        alpha_img=0.1,
        size=1.5,
    )


def save_violin_plot(adata: sc.AnnData, cfg: DownstreamConfig) -> None:
    obs_df = adata.obs
    violin_df = pd.concat(
        [
            obs_df[["cluster", "contribution_st"]].rename(columns={"contribution_st": "contribution"}).assign(modality="ST"),
            obs_df[["cluster", "contribution_sm"]].rename(columns={"contribution_sm": "contribution"}).assign(modality="SM"),
        ],
        ignore_index=True,
    )
    violin_df.to_csv(cfg.table_dir / "cluster_contribution_long.csv", index=False)

    fig, axis = plt.subplots(figsize=(13, 4.5))
    sns.violinplot(
        data=violin_df,
        x="cluster",
        y="contribution",
        hue="modality",
        split=True,
        inner="quart",
        palette=["#2ec4b6", "#FFCC70"],
        scale="width",
        bw=0.2,
        cut=0,
        ax=axis,
    )
    axis.set_xlabel("cluster")
    axis.set_ylabel("contribution")
    axis.set_title("Modality contribution by cluster")
    plt.xticks(rotation=90)
    fig.tight_layout()
    fig.savefig(cfg.figure_dir / "cluster_contribution_violin.png", dpi=220, bbox_inches="tight")
    fig.savefig(cfg.figure_dir / "cluster_contribution_violin.svg", bbox_inches="tight", format="svg")
    plt.close(fig)


def copy_optional_metrics(cfg: DownstreamConfig) -> dict[str, str]:
    copied: dict[str, str] = {}
    for key, source in {
        "metrics_full": cfg.full_metrics_csv,
        "metrics_defaultcluster": cfg.default_metrics_csv,
    }.items():
        if source is not None and source.exists():
            destination = cfg.table_dir / source.name
            shutil.copy2(source, destination)
            copied[key] = str(destination)
    return copied


def save_summary_tables(adata: sc.AnnData, pathology_metadata: dict[str, object], cfg: DownstreamConfig) -> dict[str, object]:
    contribution_summary = (
        adata.obs.groupby("cluster", observed=True)[["contribution_st", "contribution_sm"]]
        .agg(["mean", "median", "std"])
        .round(6)
    )
    contribution_summary.to_csv(cfg.table_dir / "cluster_contribution_summary.csv")

    pathology_crosstab = pd.crosstab(
        adata.obs["cluster"],
        adata.obs["pathological_annotation_plot"],
        dropna=False,
    )
    pathology_crosstab.to_csv(cfg.table_dir / "cluster_pathology_crosstab_counts.csv")

    pathology_fraction = pd.crosstab(
        adata.obs["cluster"],
        adata.obs["pathological_annotation_plot"],
        normalize="index",
        dropna=False,
    )
    pathology_fraction.to_csv(cfg.table_dir / "cluster_pathology_crosstab_fraction.csv")

    fig, axis = plt.subplots(figsize=(12, 5))
    sns.heatmap(pathology_fraction, cmap="viridis", linewidths=0.3, ax=axis)
    axis.set_title("Cluster vs pathology fraction")
    axis.set_xlabel("pathological annotation")
    axis.set_ylabel("cluster")
    fig.tight_layout()
    fig.savefig(cfg.figure_dir / "cluster_pathology_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    pathology_counts = adata.obs["pathological_annotation_plot"].value_counts(dropna=False)
    pathology_counts.to_csv(cfg.table_dir / "pathology_counts.csv", header=["n_spots"])

    copied_metrics = copy_optional_metrics(cfg)
    return {
        "n_spots_pred": int(adata.n_obs),
        "n_features": int(adata.n_vars),
        "n_clusters": int(adata.obs["cluster"].nunique()),
        "pathology_counts": pathology_counts.to_dict(),
        "pathology_matching": pathology_metadata,
        "copied_metrics": copied_metrics,
    }


def save_readme(
    summary: dict[str, object],
    gene_feature: str,
    metabolite_feature: str,
    pathology_metadata: dict[str, object],
    loss_columns: list[str],
    cosine_summary: dict[str, float],
    cfg: DownstreamConfig,
) -> None:
    lines = [
        str(cfg.title),
        f"sample: {cfg.sample_name}",
        f"result_h5ad: {cfg.result_h5ad}",
        f"defaultcluster_h5ad: {cfg.defaultcluster_h5ad}",
        f"groundtruth_h5ad: {cfg.gt_h5ad}",
        f"loss_csv: {cfg.loss_csv}",
        f"gene_feature: {gene_feature}",
        f"metabolite_feature: {metabolite_feature}",
        f"cluster_source_key_original: {cfg.cluster_key}",
        f"cluster_source_key_current: {cfg.recluster_key}",
        f"cluster_n_neighbors: {cfg.recluster_n_neighbors}",
        f"cluster_resolution: {cfg.recluster_resolution}",
        f"matched_pathology_spots: {pathology_metadata['matched_count']}",
        f"pathology_matching_mode: {pathology_metadata['matching_mode']}",
        f"pathology_common_barcodes: {pathology_metadata['common_barcodes']}",
        f"pathology_barcode_coord_mean_distance: {pathology_metadata['barcode_coord_mean_distance']}",
        f"n_spots_pred: {summary['n_spots_pred']}",
        f"n_features: {summary['n_features']}",
        f"n_clusters: {summary['n_clusters']}",
        f"mean_raw_cosine_st_joint: {cosine_summary['mean_raw_cosine_st_joint']}",
        f"mean_raw_cosine_sm_joint: {cosine_summary['mean_raw_cosine_sm_joint']}",
        f"mean_raw_cosine_delta_st_minus_sm: {cosine_summary['mean_raw_cosine_delta_st_minus_sm']}",
        f"mean_st_contribution: {cosine_summary['mean_st_contribution']}",
        f"loss_columns: {', '.join(loss_columns)}",
    ]
    if "spatial_match_threshold" in pathology_metadata:
        lines.append(f"pathology_spatial_match_threshold: {pathology_metadata['spatial_match_threshold']}")
    if "spatial_match_distance_mean" in pathology_metadata:
        lines.append(f"pathology_spatial_match_distance_mean: {pathology_metadata['spatial_match_distance_mean']}")
    if "spatial_match_distance_max" in pathology_metadata:
        lines.append(f"pathology_spatial_match_distance_max: {pathology_metadata['spatial_match_distance_max']}")
    cfg.readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_downstream(
    sample_name: str,
    result_h5ad: str | Path,
    gt_h5ad: str | Path,
    output_dir: str | Path,
    loss_csv: str | Path | None = None,
    full_metrics_csv: str | Path | None = None,
    default_metrics_csv: str | Path | None = None,
    defaultcluster_h5ad: str | Path | None = None,
    cluster_key: str = "decalign_linear_clusters",
    recluster_key: str = "decalign_linear_clusters",
    embedding_key: str = "X_emb_decalign_linear",
    reconstruction_layer: str = "reconstruction_decalign_linear",
    normalized_layer: str = "normalized",
    contribution_st_key: str = "contribution_st_decalign_linear",
    contribution_sm_key: str = "contribution_sm_decalign_linear",
    recluster_n_neighbors: int = 15,
    recluster_resolution: float = 1.0,
    recluster_random_seed: int = 0,
    spatial_match_threshold: float = 5.0,
    min_valid_spatial_matches: int = 10,
    title: str | None = None,
    clean_output: bool = False,
) -> dict[str, object]:
    cfg = DownstreamConfig(
        sample_name=sample_name,
        result_h5ad=Path(result_h5ad),
        gt_h5ad=Path(gt_h5ad),
        output_dir=Path(output_dir),
        loss_csv=Path(loss_csv) if loss_csv else None,
        full_metrics_csv=Path(full_metrics_csv) if full_metrics_csv else None,
        default_metrics_csv=Path(default_metrics_csv) if default_metrics_csv else None,
        defaultcluster_h5ad=Path(defaultcluster_h5ad) if defaultcluster_h5ad else None,
        cluster_key=cluster_key,
        recluster_key=recluster_key,
        embedding_key=embedding_key,
        reconstruction_layer=reconstruction_layer,
        normalized_layer=normalized_layer,
        contribution_st_key=contribution_st_key,
        contribution_sm_key=contribution_sm_key,
        recluster_n_neighbors=recluster_n_neighbors,
        recluster_resolution=recluster_resolution,
        recluster_random_seed=recluster_random_seed,
        spatial_match_threshold=spatial_match_threshold,
        min_valid_spatial_matches=min_valid_spatial_matches,
        title=title,
        clean_output=clean_output,
    ).normalize()

    ensure_dirs(cfg)

    sc.settings.autoshow = False
    sns.set_theme(style="whitegrid", context="talk")

    adata = load_pred_adata(cfg)
    adata, pathology_metadata = merge_pathology_annotation(adata, cfg)

    gene_feature = pick_gene_feature(adata, cfg)
    metabolite_feature = pick_metabolite_feature(adata, cfg)

    loss_columns = save_loss_curves(cfg)
    cosine_summary = save_cosine_overview(adata, cfg)
    save_umap_plots(adata, cfg)
    save_spatial_plots(adata, gene_feature, metabolite_feature, cfg)
    save_violin_plot(adata, cfg)
    summary = save_summary_tables(adata, pathology_metadata, cfg)

    adata.write_h5ad(cfg.annotated_h5ad)

    payload = {
        "sample": cfg.sample_name,
        "gene_feature": gene_feature,
        "metabolite_feature": metabolite_feature,
        "cluster_key_original": cfg.cluster_key,
        "cluster_key_current": cfg.recluster_key,
        "cluster_n_neighbors": cfg.recluster_n_neighbors,
        "cluster_resolution": cfg.recluster_resolution,
        "matched_pathology_spots": pathology_metadata["matched_count"],
        "pathology_matching": pathology_metadata,
        "cosine_summary": cosine_summary,
        "summary": summary,
        "result_h5ad": str(cfg.result_h5ad),
        "defaultcluster_h5ad": str(cfg.defaultcluster_h5ad) if cfg.defaultcluster_h5ad else None,
        "annotated_h5ad": str(cfg.annotated_h5ad),
        "figures_dir": str(cfg.figure_dir),
        "tables_dir": str(cfg.table_dir),
    }
    cfg.summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    save_readme(summary, gene_feature, metabolite_feature, pathology_metadata, loss_columns, cosine_summary, cfg)

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def run_downstream_for_sample(
    sample_name: str,
    run_root: str | Path,
    output_root: str | Path,
    config_name: str,
    gt_root: str | Path,
    processed_root: str | Path,
    clean_output: bool,
) -> str:
    paths = resolve_downstream_inputs(
        sample_name=sample_name,
        run_root=run_root,
        output_root=output_root,
        config_name=config_name,
        gt_root=gt_root,
        processed_root=processed_root,
    )

    run_downstream(
        sample_name=sample_name,
        result_h5ad=paths["result_h5ad"],
        gt_h5ad=paths["gt_h5ad"],
        output_dir=paths["sample_output_dir"],
        loss_csv=paths["loss_csv"],
        full_metrics_csv=paths["full_metrics_csv"],
        default_metrics_csv=None,
        defaultcluster_h5ad=paths["result_h5ad"],
        cluster_key="decalign_linear_clusters",
        recluster_key="decalign_linear_clusters",
        embedding_key="X_emb_decalign_linear",
        reconstruction_layer="reconstruction_decalign_linear",
        normalized_layer="normalized",
        contribution_st_key="contribution_st_decalign_linear",
        contribution_sm_key="contribution_sm_decalign_linear",
        recluster_n_neighbors=15,
        recluster_resolution=1.0,
        recluster_random_seed=0,
        spatial_match_threshold=5.0,
        min_valid_spatial_matches=10,
        title=f"{sample_name} downstream plots using model clusters",
        clean_output=clean_output,
    )
    return sample_name


def run_downstream_for_samples(
    sample_names: list[str],
    run_root: str | Path,
    output_root: str | Path,
    config_name: str,
    gt_root: str | Path,
    processed_root: str | Path,
    clean_output: bool,
    worker_count: int,
) -> list[str]:
    finished_samples: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {
            executor.submit(
                run_downstream_for_sample,
                sample_name,
                run_root,
                output_root,
                config_name,
                gt_root,
                processed_root,
                clean_output,
            ): sample_name
            for sample_name in sample_names
        }
        for future in concurrent.futures.as_completed(future_map):
            sample_name = future_map[future]
            try:
                finished_name = future.result()
            except Exception as exc:
                raise RuntimeError(f"{sample_name} downstream failed") from exc
            print(f"[downstream complete] {finished_name}", flush=True)
            finished_samples.append(finished_name)
    return finished_samples
