#!/usr/bin/env python3
"""Render one SpaDTA spatial-domain figure from precomputed mclust labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")

from matplotlib import colors as mcolors
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


project_root = Path(__file__).resolve().parents[2]
output_root = project_root / "compare_method/common/now_result/fig/SpaDTA"
processed_root = Path("/bigdat2/user/hesy/spatialmeta/SpatialMETA/processed")
spadta_root = project_root / "SpaDTA_718/runs/SM"
max_png_bytes = 1_000_000
raster_max_dimension = 900
cluster_colors = [
    "#1f77b4", "#ff7f0e", "#279e68", "#d62728", "#aa40fc",
    "#8c564b", "#e377c2", "#b5bd61", "#17becf", "#aec7e8",
    "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94",
    "#f7b6d2", "#dbdb8d", "#9edae5", "#ad494a", "#8c6d31",
]


def load_spatial_data(sample: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    reference = ad.read_h5ad(processed_root / f"{sample}.h5ad", backed="r")
    try:
        obs = reference.obs[["array_col", "array_row"]].copy()
        obs.index = reference.obs_names.astype(str)
        coords = np.asarray(reference.obsm["spatial"], dtype=np.float64)[:, :2]
        library_id = next(iter(reference.uns["spatial"]))
        spatial = reference.uns["spatial"][library_id]
        image_key = "hires" if "hires" in spatial["images"] else "lowres"
        image = np.asarray(spatial["images"][image_key])
        scale_key = "tissue_hires_scalef" if image_key == "hires" else "tissue_lowres_scalef"
        coords *= float(spatial["scalefactors"][scale_key])
    finally:
        reference.file.close()
    return obs, coords, image


def fitted_hexagons(obs: pd.DataFrame, coords: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    design = np.column_stack(
        [obs["array_col"].to_numpy(float), obs["array_row"].to_numpy(float), np.ones(len(obs))]
    )
    transform, _, _, _ = np.linalg.lstsq(design, coords, rcond=None)
    fitted = design @ transform
    deltas = [(2, 0), (1, 1), (1, -1)]
    vectors = []
    for delta_col, delta_row in deltas:
        vector = delta_col * transform[0] + delta_row * transform[1]
        vectors.extend([vector, -vector])
    vertices = []
    for first_index, first in enumerate(vectors):
        for second in vectors[first_index + 1 :]:
            matrix = np.stack([first, second])
            if abs(np.linalg.det(matrix)) <= 1e-6:
                continue
            rhs = np.array([np.dot(first, first), np.dot(second, second)]) * 0.5
            candidate = np.linalg.solve(matrix, rhs)
            if all(np.dot(candidate, vector) <= np.dot(vector, vector) * 0.5 + 1e-6 for vector in vectors):
                vertices.append(candidate)
    unique = []
    for vertex in vertices:
        if not any(np.allclose(vertex, previous, atol=1e-5) for previous in unique):
            unique.append(vertex)
    template = np.asarray(unique)
    center = template.mean(axis=0)
    angles = np.arctan2(template[:, 1] - center[1], template[:, 0] - center[0])
    template = template[np.argsort(angles)] * 0.96
    return fitted, [template + point for point in fitted]


def constrain_png(path: Path) -> None:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
    image.save(path, format="PNG", optimize=True, compress_level=9)
    while path.stat().st_size > max_png_bytes and max(image.size) > 600:
        image = image.resize(
            (max(1, round(image.width * 0.9)), max(1, round(image.height * 0.9))),
            Image.Resampling.LANCZOS,
        )
        image.save(path, format="PNG", optimize=True, compress_level=9)
    if path.stat().st_size > max_png_bytes:
        image.quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(
            path, format="PNG", optimize=True, compress_level=9
        )
    if path.stat().st_size > max_png_bytes:
        raise RuntimeError(f"could not reduce {path} below {max_png_bytes} bytes")


def load_precomputed_labels(sample: str) -> tuple[list[str], np.ndarray]:
    snapshot = spadta_root / sample / "final_protocol/epoch_0300"
    labels = pd.read_csv(snapshot / "spot_labels.csv")
    return labels["spot_id"].astype(str).tolist(), labels["mclust_label"].to_numpy()


def render(sample: str, output_dir: Path) -> tuple[Path, Path]:
    obs, coords, image = load_spatial_data(sample)
    spot_ids, labels = load_precomputed_labels(sample)
    label_series = pd.Series(labels, index=pd.Index(spot_ids), dtype=object)
    missing = obs.index.difference(label_series.index)
    extra = label_series.index.difference(obs.index)
    if len(missing) or len(extra):
        raise ValueError(f"{sample}: spot mismatch (missing={len(missing)}, extra={len(extra)})")
    labels = label_series.loc[obs.index].to_numpy(dtype=object)
    labeled = ~pd.isna(labels)
    categories = sorted(pd.unique(pd.Series(labels[labeled]).astype(str)).tolist(), key=lambda value: int(value))
    if len(categories) > len(cluster_colors):
        raise ValueError(f"fixed palette supports at most {len(cluster_colors)} clusters")
    color_map = dict(zip(categories, [mcolors.to_hex(color) for color in cluster_colors[:len(categories)]]))
    _, polygons = fitted_hexagons(obs, coords)
    labeled_polygons = [polygon for polygon, keep in zip(polygons, labeled) if keep]
    facecolors = [color_map[str(label)] for label in labels[labeled]]

    image_height, image_width = image.shape[:2]
    scale = raster_max_dimension / max(image_height, image_width)
    figure_size = (image_width * scale / 100.0, image_height * scale / 100.0)
    figure = plt.figure(figsize=figure_size, dpi=100, frameon=False)
    axis = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    axis.imshow(image, origin="upper", alpha=0.72, interpolation="none")
    axis.add_collection(
        PolyCollection(
            labeled_polygons,
            facecolors=facecolors,
            edgecolors="#ffffff",
            linewidths=0.32,
            antialiaseds=True,
            zorder=3,
        )
    )
    axis.set_xlim(-0.5, image_width - 0.5)
    axis.set_ylim(image_height - 0.5, -0.5)
    axis.set_aspect("equal")
    axis.axis("off")

    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / f"{sample}.svg"
    png_path = output_dir / f"{sample}.png"
    figure.savefig(svg_path, format="svg", bbox_inches=None, pad_inches=0, transparent=False)
    figure.savefig(png_path, format="png", dpi=100, bbox_inches=None, pad_inches=0, transparent=False)
    plt.close(figure)
    constrain_png(png_path)
    return png_path, svg_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", default="X49_T")
    parser.add_argument("--output-dir", type=Path, default=output_root)
    args = parser.parse_args()
    png_path, svg_path = render(args.sample, args.output_dir)
    print(f"wrote {png_path}")
    print(f"wrote {svg_path}")


if __name__ == "__main__":
    main()
