from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import scanpy as sc
from scanpy.plotting import palettes as sc_palettes
from scipy.spatial import cKDTree

sample = 'm3_FMP'
input_h5ad = Path(f"/data/user/hesy/projects/SpatialMETA/spaDTA/runs/first/{sample}/{sample}_output.h5ad")
output_path = Path(f"/data/user/hesy/projects/SpatialMETA/spaDTA/runs/first/{sample}/{sample}_ori_cluster.svg")
cluster_key = "decalign_linear_clusters"
plot_title = sample
alpha_img = 0.72


def sort_categories(values: list[str]) -> list[str]:
    def key_fn(value: str) -> tuple[int, object]:
        return (0, int(value)) if value.isdigit() else (1, value)

    return sorted(values, key=key_fn)


def set_cluster_palette(adata: sc.AnnData, cluster_key: str) -> None:
    categories = adata.obs[cluster_key].cat.categories.tolist()
    color_key = f"{cluster_key}_colors"
    existing = adata.uns.get(color_key)
    if existing is not None and len(existing) >= len(categories):
        adata.uns[color_key] = [mcolors.to_hex(color) for color in existing[: len(categories)]]
        return
    if len(categories) <= 20:
        base = sc_palettes.default_20
    elif len(categories) <= 28:
        base = sc_palettes.default_28
    elif len(categories) <= 102:
        base = sc_palettes.default_102
    else:
        cmap = plt.get_cmap("gist_ncar", len(categories))
        base = [mcolors.to_hex(cmap(index)) for index in range(len(categories))]
    adata.uns[color_key] = [mcolors.to_hex(base[index % len(base)]) for index in range(len(categories))]


def infer_library_id(adata: sc.AnnData) -> Optional[str]:
    spatial_block = adata.uns.get("spatial")
    if not isinstance(spatial_block, dict) or not spatial_block:
        return None
    return next(iter(spatial_block.keys()))


def infer_img_key(adata: sc.AnnData, library_id: Optional[str]) -> Optional[str]:
    if library_id is None:
        return None
    library_block = adata.uns.get("spatial", {}).get(library_id, {})
    image_block = library_block.get("images", {})
    if "hires" in image_block:
        return "hires"
    if "lowres" in image_block:
        return "lowres"
    return None


def to_plot_coords(adata: sc.AnnData, library_id: Optional[str], img_key: Optional[str]) -> tuple[np.ndarray, Optional[np.ndarray]]:
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)[:, :2]
    if library_id is None or img_key is None:
        return coords, None
    spatial_block = adata.uns["spatial"][library_id]
    image = np.asarray(spatial_block["images"][img_key])
    scalefactors = spatial_block["scalefactors"]
    scale_key = "tissue_hires_scalef" if img_key == "hires" else "tissue_lowres_scalef"
    scale = float(scalefactors[scale_key])
    return coords * scale, image


def fit_grid_transform(adata: sc.AnnData, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "array_col" not in adata.obs.columns or "array_row" not in adata.obs.columns:
        raise KeyError("adata.obs['array_col'] and adata.obs['array_row'] are required for hex tiling.")
    col = adata.obs["array_col"].to_numpy(dtype=np.float64)
    row = adata.obs["array_row"].to_numpy(dtype=np.float64)
    design = np.column_stack([col, row, np.ones(adata.n_obs, dtype=np.float64)])
    transform, _, _, _ = np.linalg.lstsq(design, coords.astype(np.float64, copy=False), rcond=None)
    return col, row, transform


def canonicalize_delta(delta_col: int, delta_row: int) -> tuple[int, int]:
    if delta_col < 0 or (delta_col == 0 and delta_row < 0):
        return (-delta_col, -delta_row)
    return (delta_col, delta_row)


def infer_neighbor_deltas(adata: sc.AnnData, coords: np.ndarray) -> list[tuple[int, int]]:
    col = adata.obs["array_col"].to_numpy(dtype=np.int64)
    row = adata.obs["array_row"].to_numpy(dtype=np.int64)
    tree = cKDTree(coords)
    distances, indices = tree.query(coords, k=min(7, adata.n_obs))
    nearest = distances[:, 1]
    positive = nearest[np.isfinite(nearest) & (nearest > 0)]
    if positive.size == 0:
        raise RuntimeError("failed to estimate nearest-neighbor spacing")
    distance_limit = float(np.quantile(positive, 0.2)) * 1.03
    counts: dict[tuple[int, int], int] = {}
    for spot_idx in range(adata.n_obs):
        for neighbor_rank in range(1, indices.shape[1]):
            distance = float(distances[spot_idx, neighbor_rank])
            if not np.isfinite(distance) or distance <= 0 or distance > distance_limit:
                continue
            neighbor_idx = int(indices[spot_idx, neighbor_rank])
            delta = canonicalize_delta(
                int(col[neighbor_idx] - col[spot_idx]),
                int(row[neighbor_idx] - row[spot_idx]),
            )
            if delta == (0, 0):
                continue
            counts[delta] = counts.get(delta, 0) + 1
    if not counts:
        raise RuntimeError("failed to infer lattice deltas from nearest neighbors")
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0][0] ** 2 + item[0][1] ** 2, item[0]))
    chosen: list[tuple[int, int]] = []
    for delta, _ in ordered:
        if not chosen:
            chosen.append(delta)
            continue
        if any(existing == delta for existing in chosen):
            continue
        if len(chosen) == 1:
            cross = chosen[0][0] * delta[1] - chosen[0][1] * delta[0]
            if cross == 0:
                continue
        chosen.append(delta)
        if len(chosen) == 3:
            break
    if len(chosen) < 3:
        raise RuntimeError("failed to infer three lattice neighbor directions")
    return chosen


def build_neighbor_vectors(transform: np.ndarray, neighbor_deltas: list[tuple[int, int]]) -> list[np.ndarray]:
    vectors = []
    for delta_col, delta_row in neighbor_deltas:
        vector = delta_col * transform[0] + delta_row * transform[1]
        vector = np.asarray(vector, dtype=np.float64)
        vectors.append(vector)
        vectors.append(-vector)
    return vectors


def build_voronoi_hex_template(neighbor_vectors: list[np.ndarray]) -> np.ndarray:
    vertices = []
    tolerance = 1e-6
    for first_idx in range(len(neighbor_vectors)):
        first = neighbor_vectors[first_idx]
        first_rhs = 0.5 * float(np.dot(first, first))
        for second_idx in range(first_idx + 1, len(neighbor_vectors)):
            second = neighbor_vectors[second_idx]
            matrix = np.stack([first, second], axis=0)
            if abs(np.linalg.det(matrix)) <= tolerance:
                continue
            rhs = np.array([first_rhs, 0.5 * float(np.dot(second, second))], dtype=np.float64)
            candidate = np.linalg.solve(matrix, rhs)
            if all(
                float(np.dot(candidate, vector)) <= 0.5 * float(np.dot(vector, vector)) + tolerance
                for vector in neighbor_vectors
            ):
                vertices.append(candidate)
    if not vertices:
        raise RuntimeError("failed to construct Voronoi hexagon")
    unique_vertices = []
    for vertex in vertices:
        if any(np.allclose(vertex, existing, atol=1e-5) for existing in unique_vertices):
            continue
        unique_vertices.append(vertex)
    polygon = np.asarray(unique_vertices, dtype=np.float64)
    center = polygon.mean(axis=0)
    angles = np.arctan2(polygon[:, 1] - center[1], polygon[:, 0] - center[0])
    polygon = polygon[np.argsort(angles)]
    return polygon.astype(np.float32, copy=False)


def build_hexagon_vertices(coords: np.ndarray, hex_template: np.ndarray) -> list[np.ndarray]:
    return [(hex_template + center).astype(np.float32, copy=False) for center in coords]


def build_legend(ax: plt.Axes, categories: list[str], colors: list[str]) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            marker="h",
            color="none",
            markerfacecolor=color,
            markeredgecolor=color,
            markersize=10,
            linewidth=0.0,
            label=category,
        )
        for category, color in zip(categories, colors)
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        handletextpad=0.5,
        fontsize=10,
    )


def prepare_adata(input_h5ad: Path, cluster_key: str) -> sc.AnnData:
    if not input_h5ad.exists():
        raise FileNotFoundError(input_h5ad)
    adata = sc.read_h5ad(input_h5ad)
    if "spatial" not in adata.obsm:
        raise KeyError("adata.obsm['spatial'] is required.")
    if cluster_key not in adata.obs.columns:
        raise KeyError(f"missing obs[{cluster_key!r}]")
    categories = sort_categories(pd.unique(adata.obs[cluster_key].astype(str)).tolist())
    adata.obs[cluster_key] = pd.Categorical(
        adata.obs[cluster_key].astype(str),
        categories=categories,
        ordered=True,
    )
    set_cluster_palette(adata, cluster_key)
    return adata


def plot_cluster(
    input_h5ad: Path,
    output_path: Path,
    cluster_key: str,
    plot_title: str,
    alpha_img: float,
) -> Path:
    adata = prepare_adata(input_h5ad, cluster_key)
    library_id = infer_library_id(adata)
    img_key = infer_img_key(adata, library_id)
    coords, image = to_plot_coords(adata, library_id, img_key)
    col, row, transform = fit_grid_transform(adata, coords)
    fitted_coords = np.column_stack([col, row, np.ones(adata.n_obs, dtype=np.float64)]) @ transform
    neighbor_deltas = infer_neighbor_deltas(adata, fitted_coords)
    neighbor_vectors = build_neighbor_vectors(transform, neighbor_deltas)
    hex_template = build_voronoi_hex_template(neighbor_vectors)
    categories = adata.obs[cluster_key].cat.categories.tolist()
    colors = adata.uns[f"{cluster_key}_colors"]
    label_codes = adata.obs[cluster_key].cat.codes.to_numpy()
    facecolors = [colors[index] for index in label_codes]
    hexagons = build_hexagon_vertices(fitted_coords, hex_template=hex_template)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if image is not None:
        image_height, image_width = image.shape[:2]
        figure_width = 9.0
        figure_height = max(6.0, figure_width * image_height / image_width)
    else:
        x_span = float(fitted_coords[:, 0].max() - fitted_coords[:, 0].min()) if fitted_coords.shape[0] else 1.0
        y_span = float(fitted_coords[:, 1].max() - fitted_coords[:, 1].min()) if fitted_coords.shape[0] else 1.0
        figure_width = 9.0
        figure_height = max(6.0, figure_width * y_span / max(x_span, 1.0))

    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    if image is not None:
        ax.imshow(image, origin="upper", alpha=alpha_img, interpolation="none")
    poly = PolyCollection(
        hexagons,
        facecolors=facecolors,
        edgecolors="#ffffff",
        linewidths=0.35,
        antialiaseds=True,
        zorder=3,
    )
    ax.add_collection(poly)
    ax.autoscale_view()
    if image is None:
        ax.invert_yaxis()
    build_legend(ax, categories, colors)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(plot_title)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    plot_path = plot_cluster(
        input_h5ad=input_h5ad,
        output_path=output_path,
        cluster_key=cluster_key,
        plot_title=plot_title,
        alpha_img=alpha_img,
    )
    print(plot_path)
