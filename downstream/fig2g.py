from __future__ import annotations

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.colors import LinearSegmentedColormap
from scipy import sparse
from scipy.interpolate import splprep, splev
from scipy.spatial import Delaunay


project_root = Path("/data/user/hesy/projects/SpatialMETA")

sample = "X49_T"
run_root = project_root / "SpaDTA_718" / "runs" / "sm_downstream"
input_h5ad = run_root / "inputs" / sample / f"{sample}_output.h5ad"
cluster_label_h5ad = input_h5ad
feature_source_h5ad = input_h5ad
output_dir = run_root / "fig2g"

CLUSTER_KEY = "decalign_linear_clusters"
MAJOR_KEY = "marker_named_group"
SUBTYPE_KEY = "marker_named_cluster"
LAYER = "normalized"
MARGIN_MIXED = 0.25
MARGIN_HIGH = 0.6

cluster_key = CLUSTER_KEY
major_value = "Imm"
subtype_value = "Imm_1"
st_features = [
    "CD74",
    "B2M",
    "IFI44L",
    "GLRX",
    "GSTA1",
    "CXCL14",
    "MMP1",
]
sm_features = [
    "162.11245183994234",
    "204.12333017003994",
    "184.0946410826148",
    "140.06830740592687",
    "227.10823423028484",
    "365.24700583231225",
    "389.24709140897994",
    "804.5505835336571",
]
margin_mixed = MARGIN_MIXED
margin_high = MARGIN_HIGH

MARKER_SETS = {
    "Imm": ["CD3D", "CD74", "PTPRC", "NKG7"],
    "Endo": ["CD34", "PECAM1", "APLN"],
    "Stro": ["COL3A1", "ACTA2", "COL1A1", "DCN", "LUM"],
    "Mal": ["NDUFA4L2", "CNDP2", "EGFR", "CA9", "EPCAM", "KRT8", "KRT18", "KRT19"],
}

EXPRESSION_CMAP = LinearSegmentedColormap.from_list(
    "expression_soft_teal",
    ["#f7f4ea", "#dcefe7", "#9fd3c7", "#4aa0b5", "#1f5a89"],
)

BOUNDARY_COLOR = "#111111"
BOUNDARY_LINEWIDTH = 1.45
BOUNDARY_ALPHA_SCALE = 2.35
BOUNDARY_SMOOTHNESS = 0.0015


def to_dense(x) -> np.ndarray:
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def natural_sort(values: list[str]) -> list[str]:
    def key(value: str) -> tuple[int, object]:
        text = str(value)
        return (0, int(text)) if text.isdigit() else (1, text)

    return sorted(values, key=key)


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


def annotate_single_sample_group(
    adata: sc.AnnData,
    major_value: str,
    cluster_key: str,
    margin_mixed: float,
    margin_high: float,
) -> tuple[sc.AnnData, pd.DataFrame]:
    adata.obs[cluster_key] = adata.obs[cluster_key].astype(str)

    marker_sets = {k: [g for g in v if g in adata.var_names] for k, v in MARKER_SETS.items()}
    genes = [g for values in marker_sets.values() for g in values]
    X = adata[:, genes].layers[LAYER]
    X = to_dense(X).astype(np.float32, copy=False)
    df = pd.DataFrame(X, columns=genes)
    df["cluster"] = adata.obs[cluster_key].astype(str).values
    means = df.groupby("cluster", observed=True).mean()
    means = means.loc[natural_sort(means.index.astype(str).tolist())]
    zscore = (means - means.mean(axis=0)) / means.std(axis=0).replace(0, np.nan)
    zscore = zscore.fillna(0.0)

    scores = pd.DataFrame(index=means.index)
    for label, genes_in_set in marker_sets.items():
        scores[label] = zscore[genes_in_set].mean(axis=1)

    rows: list[dict[str, object]] = []
    cluster_sizes = adata.obs[cluster_key].astype(str).value_counts()
    for cluster in scores.index.astype(str):
        score_map = {label: float(scores.loc[cluster, label]) for label in marker_sets}
        ranked = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)
        best, best_score = ranked[0]
        second, second_score = ranked[1]
        margin = float(best_score - second_score)
        broad = "Mixed" if margin < float(margin_mixed) else best
        rows.append({"cluster": cluster, "n_spots": int(cluster_sizes[cluster]), "annotation_broad": broad, **score_map})
    annotation_df = pd.DataFrame(rows).sort_values("cluster", key=lambda s: s.astype(int)).reset_index(drop=True)

    group_df = annotation_df.loc[annotation_df["annotation_broad"].eq(major_value)].copy()
    group_df = group_df.sort_values(["n_spots", "cluster"], ascending=[False, True]).reset_index(drop=True)
    group_df["marker_named_group"] = major_value
    group_df["marker_named_cluster"] = [f"{major_value}_{i}" for i in range(1, len(group_df) + 1)]

    cluster_to_group = dict(zip(annotation_df["cluster"].astype(str), annotation_df["annotation_broad"].astype(str)))
    cluster_to_subtype = dict(zip(group_df["cluster"].astype(str), group_df["marker_named_cluster"].astype(str)))
    adata.obs[MAJOR_KEY] = adata.obs[cluster_key].astype(str).map(cluster_to_group).fillna("Other").astype(str)
    adata.obs[SUBTYPE_KEY] = adata.obs[cluster_key].astype(str).map(cluster_to_subtype).fillna("").astype(str)
    return adata, group_df


def to_hires_coords(adata: sc.AnnData, sample: str) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    spatial_block = adata.uns["spatial"][sample]
    image = np.asarray(spatial_block["images"]["hires"])
    scalefactors = spatial_block["scalefactors"]
    scale = float(scalefactors["tissue_hires_scalef"])
    coords = np.asarray(adata.obsm["spatial"], dtype=float)[:, :2] * scale
    return coords, image, scalefactors


def feature_expression(adata: sc.AnnData, feature: str) -> np.ndarray:
    idx = np.where(adata.var_names.astype(str) == str(feature))[0]
    if len(idx) == 0:
        raise KeyError(f"Feature {feature!r} not found in adata.var_names")
    values = adata.layers[LAYER][:, idx[0]]
    values = to_dense(values).astype(np.float32, copy=False).ravel()
    return values


def typical_neighbor_distance(coords: np.ndarray) -> float:
    if coords.shape[0] < 2:
        return 1.0
    deltas = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((deltas**2).sum(axis=2))
    dist[dist == 0] = np.inf
    return float(np.median(np.min(dist, axis=1)))


def polygon_area(boundary: np.ndarray) -> float:
    if boundary.shape[0] < 3:
        return 0.0
    pts = boundary[:-1] if np.allclose(boundary[0], boundary[-1]) else boundary
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def alpha_shape_boundaries(coords: np.ndarray, alpha: float) -> list[np.ndarray]:
    if coords.shape[0] < 4:
        return []
    tri = Delaunay(coords)
    edges: dict[tuple[int, int], int] = {}
    for simplex in tri.simplices:
        pts = coords[simplex]
        a = np.linalg.norm(pts[1] - pts[0])
        b = np.linalg.norm(pts[2] - pts[1])
        c = np.linalg.norm(pts[0] - pts[2])
        area2 = abs(np.cross(pts[1] - pts[0], pts[2] - pts[0]))
        if area2 <= 1e-9:
            continue
        radius = a * b * c / (2.0 * area2)
        if radius > alpha:
            continue
        for i, j in ((0, 1), (1, 2), (2, 0)):
            edge = tuple(sorted((int(simplex[i]), int(simplex[j]))))
            edges[edge] = edges.get(edge, 0) + 1

    boundary_edges = [edge for edge, count in edges.items() if count == 1]
    if not boundary_edges:
        return []

    adjacency: dict[int, list[int]] = {}
    for i, j in boundary_edges:
        adjacency.setdefault(i, []).append(j)
        adjacency.setdefault(j, []).append(i)

    loops: list[np.ndarray] = []
    visited_nodes: set[int] = set()
    min_area = typical_neighbor_distance(coords) ** 2 * 0.35

    for start in adjacency:
        if start in visited_nodes:
            continue
        stack = [start]
        component: set[int] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            visited_nodes.add(node)
            stack.extend(adjacency.get(node, []))

        if len(component) < 3:
            continue

        comp_start = min(component)
        neighbors = [node for node in adjacency.get(comp_start, []) if node in component]
        if not neighbors:
            continue

        ordered = [comp_start, neighbors[0]]
        prev, current = comp_start, neighbors[0]
        max_steps = len(component) + 2
        while len(ordered) <= max_steps:
            candidates = [node for node in adjacency.get(current, []) if node in component and node != prev]
            if not candidates:
                break
            next_node = candidates[0]
            ordered.append(next_node)
            if next_node == comp_start:
                break
            prev, current = current, next_node

        if len(ordered) < 4 or ordered[-1] != comp_start:
            continue

        boundary = coords[np.array(ordered)]
        if polygon_area(boundary) < min_area:
            continue
        loops.append(boundary)

    loops.sort(key=polygon_area, reverse=True)
    return loops


def smooth_closed_boundary(boundary: np.ndarray) -> np.ndarray:
    if boundary.shape[0] < 5:
        return boundary
    if not np.allclose(boundary[0], boundary[-1]):
        boundary = np.vstack([boundary, boundary[0]])
    try:
        tck, _ = splprep(
            [boundary[:, 0], boundary[:, 1]],
            s=BOUNDARY_SMOOTHNESS * boundary.shape[0] * typical_neighbor_distance(boundary) ** 2,
            per=True,
        )
        u = np.linspace(0.0, 1.0, max(160, boundary.shape[0] * 8))
        x, y = splev(u, tck)
        return np.column_stack([x, y])
    except Exception:
        return boundary


def draw_boundary(ax: plt.Axes, target_coords: np.ndarray, scope_coords: np.ndarray) -> bool:
    if target_coords.shape[0] < 4:
        return False
    alpha = typical_neighbor_distance(scope_coords) * BOUNDARY_ALPHA_SCALE
    boundaries = alpha_shape_boundaries(target_coords, alpha)
    if not boundaries:
        boundaries = alpha_shape_boundaries(target_coords, alpha * 1.5)
    if not boundaries:
        return False
    for boundary in boundaries:
        boundary = smooth_closed_boundary(boundary)
        ax.plot(
            boundary[:, 0],
            boundary[:, 1],
            color=BOUNDARY_COLOR,
            linewidth=BOUNDARY_LINEWIDTH,
            solid_joinstyle="round",
            solid_capstyle="round",
            zorder=5,
        )
    return True


def gray_fill_value(image: np.ndarray) -> np.ndarray:
    channels = 1 if image.ndim == 2 else image.shape[2]
    if np.issubdtype(image.dtype, np.integer):
        value = np.full((channels,), 225, dtype=image.dtype)
    else:
        value = np.full((channels,), 0.88, dtype=image.dtype)
    return value[0] if channels == 1 else value


def tissue_mask_from_image(image: np.ndarray) -> np.ndarray:
    image_array = np.asarray(image)
    if np.issubdtype(image_array.dtype, np.integer):
        threshold = 245
        if image_array.ndim == 2:
            return image_array < threshold
        return np.any(image_array < threshold, axis=2)
    threshold = 0.96
    if image_array.ndim == 2:
        return image_array < threshold
    return np.any(image_array < threshold, axis=2)


def build_region_preserving_background(
    image: np.ndarray,
    region_coords: np.ndarray,
    point_diameter: float,
) -> np.ndarray:
    background = np.array(image, copy=True)
    tissue_mask = tissue_mask_from_image(background)
    if region_coords.shape[0] == 0:
        fill_value = gray_fill_value(background)
        if background.ndim == 2:
            background[tissue_mask] = fill_value
        else:
            background[tissue_mask, :] = fill_value
        return background

    height, width = background.shape[:2]
    mask = np.zeros((height, width), dtype=bool)
    radius = max(2, int(round(point_diameter * 0.78)))
    radius2 = radius * radius

    for x_coord, y_coord in region_coords:
        x_center = int(round(float(x_coord)))
        y_center = int(round(float(y_coord)))
        x0 = max(0, x_center - radius)
        x1 = min(width, x_center + radius + 1)
        y0 = max(0, y_center - radius)
        y1 = min(height, y_center + radius + 1)
        yy_local, xx_local = np.ogrid[y0:y1, x0:x1]
        local = (xx_local - x_center) ** 2 + (yy_local - y_center) ** 2 <= radius2
        mask[y0:y1, x0:x1] |= local

    fill_value = gray_fill_value(background)
    if background.ndim == 2:
        background[tissue_mask & (~mask)] = fill_value
    else:
        background[tissue_mask & (~mask), :] = fill_value
    return background


def save_undergraph_background(
    adata: sc.AnnData,
    sample: str,
    major_value: str,
    output_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords, image, scalefactors = to_hires_coords(adata, sample)
    scope_mask = adata.obs[MAJOR_KEY].astype(str).eq(major_value).to_numpy()
    scope_coords = coords[scope_mask]
    point_diameter = float(scalefactors["spot_diameter_fullres"]) * float(scalefactors["tissue_hires_scalef"])
    background = build_region_preserving_background(image, scope_coords, point_diameter)

    fig, ax = plt.subplots(figsize=(10.8, 8.0))
    ax.imshow(background, origin="upper")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return coords, background, scope_mask


def plot_feature(
    adata: sc.AnnData,
    sample: str,
    feature: str,
    modality: str,
    major_value: str,
    target_value: str,
    background_image: np.ndarray,
    coords: np.ndarray,
    output_base: Path,
) -> dict[str, object]:
    target_mask = adata.obs[SUBTYPE_KEY].astype(str).eq(target_value).to_numpy()
    tissue_mask = adata.obs["in_tissue"].astype(bool).to_numpy() if "in_tissue" in adata.obs.columns else np.ones(adata.n_obs, dtype=bool)
    plot_mask = adata.obs[MAJOR_KEY].astype(str).eq(major_value).to_numpy() & tissue_mask
    compare_label = f"{target_value} vs other {major_value}"

    expr = feature_expression(adata, feature)
    plot_expr = expr[plot_mask]
    plot_coords = coords[plot_mask]
    target_expr = expr[target_mask]
    target_coords = coords[target_mask]
    other_mask = plot_mask & (~target_mask)
    other_expr = expr[other_mask]
    if target_expr.size == 0 or plot_expr.size == 0 or other_expr.size == 0:
        raise RuntimeError(f"invalid plotting scope for {feature} / {target_value}")

    scaled = np.log1p(plot_expr)
    vmin = float(np.quantile(scaled, 0.02))
    vmax = float(np.quantile(scaled, 0.98))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.imshow(background_image, origin="upper", alpha=1.0)
    sca = ax.scatter(
        plot_coords[:, 0],
        plot_coords[:, 1],
        c=scaled,
        cmap=EXPRESSION_CMAP,
        s=18.0,
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
        alpha=0.92,
        rasterized=True,
    )
    boundary_drawn = draw_boundary(ax, target_coords, plot_coords)

    mean_in = float(target_expr.mean())
    mean_out = float(other_expr.mean())
    log2fc = float(np.log2((mean_in + 1e-3) / (mean_out + 1e-3)))

    ax.set_title(str(feature), fontsize=18, fontweight="bold", pad=16)
    ax.text(
        0.5,
        0.975,
        f"{compare_label}, log$_2$ FC = {log2fc:.2f}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="normal",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(sca, ax=ax, fraction=0.04, pad=0.015, shrink=0.76, aspect=24)
    cbar.ax.set_title("Scaled\nexpression", fontsize=14, pad=5)
    cbar.ax.tick_params(labelsize=11, length=2, pad=3)

    png_path = Path(f"{output_base}.png")
    svg_path = Path(f"{output_base}.svg")
    fig.tight_layout()
    fig.savefig(png_path, dpi=260, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight", format="svg")
    plt.close(fig)
    return {
        "feature": feature,
        "modality": modality,
        "target_value": target_value,
        "comparison_label": compare_label,
        "output_png": str(png_path),
        "output_svg": str(svg_path),
        "log2fc": log2fc,
        "boundary_drawn": boundary_drawn,
        "scaled_vmin": vmin,
        "scaled_vmax": vmax,
    }


def run_fig2g() -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    adata = load_adata(cluster_label_h5ad, feature_source_h5ad, cluster_key)
    adata, group_df = annotate_single_sample_group(
        adata,
        major_value,
        cluster_key,
        margin_mixed,
        margin_high,
    )
    if subtype_value not in set(adata.obs[SUBTYPE_KEY].astype(str)):
        raise RuntimeError(f"Subtype {subtype_value!r} not found in sample {sample}")

    background_path = output_dir / f"{sample}_{major_value.lower()}_{subtype_value}_undergraph.png"
    coords, background_image, _ = save_undergraph_background(adata, sample, major_value, background_path)

    results: list[dict[str, object]] = []
    for feature in st_features:
        output_base = output_dir / f"{sample}_{subtype_value}_{feature}_slice_expression"
        results.append(plot_feature(adata, sample, feature, "ST", major_value, subtype_value, background_image, coords, output_base))
    for feature in sm_features:
        output_base = output_dir / f"{sample}_{subtype_value}_{feature}_slice_expression"
        results.append(plot_feature(adata, sample, feature, "SM", major_value, subtype_value, background_image, coords, output_base))

    selection_rows = []
    for row in results:
        selection_rows.append({"feature": row["feature"], "modality": row["modality"], "log2fc": row["log2fc"]})
    pd.DataFrame(selection_rows).to_csv(output_dir / f"{sample}_{subtype_value}_selected_features.csv", index=False)

    summary = {
        "sample": sample,
        "major_value": major_value,
        "target_value": subtype_value,
        "background_png": str(background_path),
        "group_clusters": group_df.to_dict(orient="records"),
        "st_features": st_features,
        "sm_features": sm_features,
        "results": results,
    }
    summary_path = output_dir / f"{sample}_{subtype_value}_slice_expression_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "outputs": [r["output_png"] for r in results]}, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    run_fig2g()
