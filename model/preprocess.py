from __future__ import annotations

import csv
import numpy as np
import scanpy as sc
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors


MODEL_INPUT_VERSION = "spadta718_v1"


def _fixed_pca(values: np.ndarray, n_components: int = 30) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    centered = values - values.mean(axis=0, keepdims=True)
    n_components = min(max(int(n_components), 1), centered.shape[0], centered.shape[1])
    if not np.any(centered):
        return np.zeros((centered.shape[0], n_components), dtype=np.float32)
    if centered.shape[0] > centered.shape[1]:
        covariance = centered.T @ centered
        _, eigenvectors = np.linalg.eigh(covariance)
        components = eigenvectors[:, -n_components:][:, ::-1]
        return (centered @ components).astype(np.float32, copy=False)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return (centered @ vt[:n_components].T).astype(np.float32, copy=False)


def _symmetric_knn_graph(values: np.ndarray, k: int) -> sparse.csr_matrix:
    n_obs = int(values.shape[0])
    if n_obs <= 1:
        return sparse.csr_matrix((n_obs, n_obs), dtype=np.float32)
    k = min(max(int(k), 1), n_obs - 1)
    indices = NearestNeighbors(n_neighbors=k + 1).fit(values).kneighbors(values, return_distance=False)[:, 1:]
    rows = np.repeat(np.arange(n_obs, dtype=np.int64), k)
    cols = indices.reshape(-1).astype(np.int64, copy=False)
    graph = sparse.csr_matrix((np.ones(rows.size, dtype=np.float32), (rows, cols)), shape=(n_obs, n_obs))
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    return graph


def prepare_spadta_model_input(
    adata,
    *,
    modality: str,
    expression_graph_k: int,
    spatial_context_k: int = 12,
):
    """Create the complete, immutable h5ad consumed by SpaDTA training."""
    adata = adata.copy()
    if "type" not in adata.var or "spatial" not in adata.obsm:
        raise ValueError("SpaDTA input requires var['type'] and obsm['spatial']")
    modality = str(modality).lower()
    if modality not in {"sm", "atac"}:
        raise ValueError("modality must be 'sm' or 'atac'")

    if modality == "sm":
        if "counts" not in adata.layers:
            raise ValueError("SM preprocessing requires layers['counts']")
        adata.X = adata.layers["counts"].copy()
        normalize_total_joint_adata_sm_st(adata, target_sum_SM=1e3, target_sum_ST=None)

    X = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
    X = np.asarray(X, dtype=np.float32)
    encoder_input = np.log1p(np.clip(X, a_min=0.0, a_max=None)).astype(np.float32, copy=False)
    adata.layers["spadta_encoder_input"] = sparse.csr_matrix(encoder_input)

    types = adata.var["type"].astype(str).to_numpy()
    st_mask, sm_mask = types == "ST", types == "SM"
    if not st_mask.any() or not sm_mask.any():
        raise ValueError("SpaDTA input must contain both ST and SM features")
    st_values, sm_values = encoder_input[:, st_mask], encoder_input[:, sm_mask]
    st_std = np.clip(st_values.std(axis=0), 1e-4, None)
    sm_std = np.clip(sm_values.std(axis=0), 1e-4, None)
    st_pca = _fixed_pca((st_values - st_values.mean(axis=0)) / st_std)
    sm_pca = _fixed_pca((sm_values - sm_values.mean(axis=0)) / sm_std)
    adata.obsm["spadta_expression_pca_st"] = st_pca
    adata.obsm["spadta_expression_pca_sm"] = sm_pca

    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)[:, :2]
    coord_std = np.clip(coords.std(axis=0), 1e-4, None)
    adata.obsm["spadta_spatial_standardized"] = ((coords - coords.mean(axis=0)) / coord_std).astype(np.float32)
    adata.obsp["spadta_graph_spatial"] = _symmetric_knn_graph(coords, 6)
    adata.obsp["spadta_graph_expression_st"] = _symmetric_knn_graph(st_pca, expression_graph_k)
    adata.obsp["spadta_graph_expression_sm"] = _symmetric_knn_graph(sm_pca, expression_graph_k)

    n_neighbors = min(int(spatial_context_k) + 1, adata.n_obs)
    distances, indices = NearestNeighbors(n_neighbors=n_neighbors).fit(coords).kneighbors(coords)
    neighbor_idx = indices[:, 1:].astype(np.int64, copy=False)
    base_dist = distances[:, 1:].astype(np.float32, copy=False)
    local_scale = np.clip(np.median(base_dist, axis=1, keepdims=True), 1e-4, None)
    neighbor_rel = ((coords[neighbor_idx] - coords[:, None, :]) / local_scale[:, :, None]).astype(np.float32)
    adata.obsm["spadta_spatial_neighbor_idx"] = neighbor_idx
    adata.obsm["spadta_spatial_neighbor_rel"] = neighbor_rel.reshape(adata.n_obs, -1)
    adata.obsm["spadta_spatial_neighbor_dist"] = np.linalg.norm(neighbor_rel, axis=2).astype(np.float32)
    adata.uns["spadta_model_input"] = {
        "version": MODEL_INPUT_VERSION,
        "modality": modality,
        "expression_graph_k": int(expression_graph_k),
        "spatial_graph_k": 6,
        "spatial_context_k": int(neighbor_idx.shape[1]),
        "st_representation": "counts" if modality == "sm" else "precomputed_RNA_features",
        "sm_representation": "library_size_1000" if modality == "sm" else "ATAC_LSI",
    }
    return adata


def validate_spadta_model_input(adata, *, expression_graph_k: int, spatial_context_k: int) -> None:
    required_layers = {"spadta_encoder_input"}
    required_obsm = {
        "spadta_expression_pca_st", "spadta_expression_pca_sm", "spadta_spatial_standardized",
        "spadta_spatial_neighbor_idx", "spadta_spatial_neighbor_rel", "spadta_spatial_neighbor_dist",
    }
    required_obsp = {"spadta_graph_spatial", "spadta_graph_expression_st", "spadta_graph_expression_sm"}
    missing = [f"layers[{key}]" for key in required_layers if key not in adata.layers]
    missing += [f"obsm[{key}]" for key in required_obsm if key not in adata.obsm]
    missing += [f"obsp[{key}]" for key in required_obsp if key not in adata.obsp]
    metadata = adata.uns.get("spadta_model_input", {})
    if metadata.get("version") != MODEL_INPUT_VERSION:
        missing.append(f"uns['spadta_model_input'].version={MODEL_INPUT_VERSION}")
    if int(metadata.get("expression_graph_k", -1)) != int(expression_graph_k):
        missing.append(f"expression_graph_k={expression_graph_k}")
    if int(metadata.get("spatial_context_k", -1)) != min(int(spatial_context_k), max(adata.n_obs - 1, 0)):
        missing.append(f"spatial_context_k={spatial_context_k}")
    if missing:
        raise ValueError("Input is not a model-ready SpaDTA h5ad; run the modality preprocessing script. Missing: " + ", ".join(missing))


def _assign_subset_back(adata, mask: np.ndarray, values) -> None:
    if sparse.issparse(adata.X):
        x_lil = adata.X.tolil(copy=True)
        x_lil[:, mask] = values
        adata.X = x_lil.tocsr()
    else:
        adata.X[:, mask] = values


def normalize_total_joint_adata_sm_st(
    joint_adata,
    target_sum_SM: int | float | None = 1e4,
    target_sum_ST: int | float | None = 1e4,
) -> None:
    if "type" not in joint_adata.var.columns:
        raise KeyError("joint_adata.var must contain a 'type' column with 'SM'/'ST' labels")

    feature_types = joint_adata.var["type"].astype(str).to_numpy()

    if target_sum_SM is not None:
        sm_mask = feature_types == "SM"
        if sm_mask.any():
            joint_adata_sm = joint_adata[:, sm_mask].copy()
            sc.pp.normalize_total(joint_adata_sm, target_sum=target_sum_SM)
            _assign_subset_back(joint_adata, sm_mask, joint_adata_sm.X)

    if target_sum_ST is not None:
        st_mask = feature_types == "ST"
        if st_mask.any():
            joint_adata_st = joint_adata[:, st_mask].copy()
            sc.pp.normalize_total(joint_adata_st, target_sum=target_sum_ST)
            _assign_subset_back(joint_adata, st_mask, joint_adata_st.X)


from pathlib import Path
import math
import random

import numexpr as ne
import spatialmeta as smt
import torch


def seed_preprocess(random_seed: int) -> None:
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False


def normalize_non_inplace(adata) -> None:
    adata.layers["normalized"] = adata.X.copy()
    sc.pp.normalize_total(adata, layer="normalized")
    sc.pp.log1p(adata, layer="normalized")


def filter_zero_total_obs(adata):
    x = adata.X
    totals = np.asarray(x.sum(axis=1)).reshape(-1)
    keep_mask = totals > 0
    if keep_mask.all():
        return adata
    return adata[keep_mask].copy()


def describe_spatial_connected_components(
    coords,
    neighbor_radius: float = 1.5,
) -> tuple[list[dict[str, float | int]], np.ndarray]:
    coords = np.asarray(coords, dtype=np.float64)
    tree = cKDTree(coords[:, :2])
    pairs = tree.query_pairs(r=float(neighbor_radius))
    rows: list[int] = []
    cols: list[int] = []
    for left_index, right_index in pairs:
        rows.extend([left_index, right_index])
        cols.extend([right_index, left_index])

    graph = sparse.coo_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)),
        shape=(coords.shape[0], coords.shape[0]),
    ).tocsr()
    n_components, labels = sparse.csgraph.connected_components(graph, directed=False)

    component_rows: list[dict[str, float | int]] = []
    for component_id in range(int(n_components)):
        component_index = np.flatnonzero(labels == component_id)
        component_coords = coords[component_index, :2]
        component_rows.append(
            {
                "component_id": int(component_id),
                "n_obs": int(component_index.size),
                "xmin": float(component_coords[:, 0].min()),
                "xmax": float(component_coords[:, 0].max()),
                "ymin": float(component_coords[:, 1].min()),
                "ymax": float(component_coords[:, 1].max()),
                "centroid_x": float(component_coords[:, 0].mean()),
                "centroid_y": float(component_coords[:, 1].mean()),
            }
        )
    return component_rows, labels


def select_lower_left_component_indices(coords, neighbor_radius: float = 1.5):
    component_rows, labels = describe_spatial_connected_components(
        coords,
        neighbor_radius=neighbor_radius,
    )
    selected_row = min(
        component_rows,
        key=lambda row: (
            float(row["xmin"]),
            -float(row["ymin"]),
            -int(row["n_obs"]),
            int(row["component_id"]),
        ),
    )
    selected_indices = np.flatnonzero(labels == int(selected_row["component_id"]))
    return selected_indices, component_rows, selected_row


def describe_sm_connected_components(
    adata_sm,
    neighbor_radius: float = 1.5,
) -> tuple[list[dict[str, float | int]], np.ndarray]:
    if "spatial" not in adata_sm.obsm:
        raise KeyError("adata_sm.obsm['spatial'] is required")
    return describe_spatial_connected_components(
        adata_sm.obsm["spatial"],
        neighbor_radius=neighbor_radius,
    )


def select_sm_lower_left_component(adata_sm, neighbor_radius: float = 1.5):
    selected_indices, component_rows, selected_row = select_lower_left_component_indices(
        adata_sm.obsm["spatial"],
        neighbor_radius=neighbor_radius,
    )
    return adata_sm[selected_indices].copy(), component_rows, selected_row


def write_component_summary_csv(rows, output_path: str | Path) -> None:
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def apply_legacy_st_qc_log_columns(adata) -> None:
    if "n_genes_by_counts" in adata.obs:
        adata.obs["log1p_n_genes_by_counts"] = np.array(
            [math.log1p(int(value)) for value in adata.obs["n_genes_by_counts"].to_numpy()],
            dtype=np.float64,
        )
    if "total_counts" in adata.obs:
        total_counts = adata.obs["total_counts"].to_numpy(dtype=np.float32)
        adata.obs["log1p_total_counts"] = ne.evaluate("log1p(total_counts)").astype(np.float32)
    if "total_counts_mt" in adata.obs:
        total_counts_mt = adata.obs["total_counts_mt"].to_numpy(dtype=np.float32)
        adata.obs["log1p_total_counts_mt"] = ne.evaluate("log1p(total_counts_mt)").astype(
            np.float32
        )


def resolve_sm_csv_path(sm_input_path: str | Path) -> Path:
    sm_path = Path(sm_input_path).expanduser()
    if sm_path.is_file():
        if sm_path.suffix.lower() != ".csv":
            raise ValueError(f"SM 输入文件必须是 csv: {sm_path}")
        return sm_path
    if not sm_path.exists():
        raise FileNotFoundError(f"找不到 SM 输入路径: {sm_path}")

    csv_candidates = sorted(sm_path.glob("*.csv"))
    if len(csv_candidates) == 1:
        return csv_candidates[0]
    if len(csv_candidates) > 1:
        raise ValueError(f"SM 输入目录包含多个 csv，请明确指定文件: {sm_path}")

    recursive_candidates = sorted(sm_path.rglob("*.csv"))
    if len(recursive_candidates) == 1:
        return recursive_candidates[0]
    if len(recursive_candidates) > 1:
        raise ValueError(f"SM 输入目录递归找到多个 csv，请明确指定文件: {sm_path}")
    raise FileNotFoundError(f"SM 输入目录下没有找到 csv 文件: {sm_path}")


def resolve_st_dir_path(st_input_path: str | Path) -> Path:
    st_path = Path(st_input_path).expanduser()
    if not st_path.exists():
        raise FileNotFoundError(f"找不到 ST 输入路径: {st_path}")

    if (st_path / "filtered_feature_bc_matrix.h5").exists():
        return st_path
    outs_path = st_path / "outs"
    if outs_path.exists() and (outs_path / "filtered_feature_bc_matrix.h5").exists():
        return outs_path
    raise FileNotFoundError(
        f"ST 输入路径既不是 Visium outs 目录，也不包含 outs/filtered_feature_bc_matrix.h5: {st_path}"
    )


def read_raw_sm_st(sm_input_path: str | Path, st_input_path: str | Path):
    sm_csv_path = resolve_sm_csv_path(sm_input_path)
    st_dir_path = resolve_st_dir_path(st_input_path)
    adata_sm = smt.pp.read_sm_csv_as_anndata(str(sm_csv_path))
    adata_st = sc.read_visium(str(st_dir_path))
    adata_st.var_names_make_unique()
    return adata_sm, adata_st


def preprocess_sm_for_joint(
    adata_sm,
    min_total_intensity_raw: float = 0.0,
):
    adata_sm = adata_sm.copy()
    smt.pp.calculate_qc_metrics_sm(adata_sm)
    if min_total_intensity_raw > 0:
        adata_sm = smt.pp.filter_cells_sm(
            adata_sm,
            min_total_intensity=min_total_intensity_raw,
        )
    return adata_sm


def preprocess_st_for_joint(
    adata_st,
    min_counts: int = 0,
    min_genes: int = 0,
):
    adata_st = adata_st.copy()
    adata_st.var["mt"] = adata_st.var_names.str.startswith("mt-")
    sc.pp.calculate_qc_metrics(adata_st, qc_vars=["mt"], inplace=True)
    adata_st = smt.pp.removeHsp_mt_Rpl_Dnaj(adata_st)
    apply_legacy_st_qc_log_columns(adata_st)
    if min_counts > 0:
        sc.pp.filter_cells(adata_st, min_counts=min_counts)
    if min_genes > 0:
        sc.pp.filter_cells(adata_st, min_genes=min_genes)
    return adata_st


def prepare_alignment_inputs_for_joint(
    adata_st,
    adata_sm,
    top_genes: int = 2000,
    top_metabolites: int = 800,
    rotation_degrees: float = -90.0,
):
    adata_st_scored = filter_zero_total_obs(adata_st.copy())
    normalize_non_inplace(adata_st_scored)
    smt.pp.spatial_variable(
        adata_st_scored,
        n_top_variable=top_genes,
        add_key="highly_variable_moranI",
        layer="normalized",
    )
    st_names = adata_st_scored.var_names[adata_st_scored.var["highly_variable_moranI"]]
    adata_st_selected = adata_st[:, st_names].copy()
    adata_st_selected.var["highly_variable_moranI"] = True

    adata_sm_scored = filter_zero_total_obs(adata_sm.copy())
    sc.pp.normalize_total(adata_sm_scored, target_sum=1e3)
    smt.pp.spatial_variable(
        adata_sm_scored,
        n_top_variable=top_metabolites,
        add_key="highly_variable_moranI",
    )
    sm_names = adata_sm_scored.var_names[adata_sm_scored.var["highly_variable_moranI"]]
    adata_sm_selected = adata_sm[:, sm_names].copy()
    adata_sm_selected.var["highly_variable_moranI"] = True

    # Alignment uses normalized copies of exactly the features retained for training.
    adata_st_align = filter_zero_total_obs(adata_st_selected.copy())
    normalize_non_inplace(adata_st_align)
    adata_st_align = filter_zero_total_obs(adata_st_align)
    st_xmin = adata_st_align.obsm["spatial"][:, 0].min()
    st_xmax = adata_st_align.obsm["spatial"][:, 0].max()
    adata_st_align.obsm["spatial_normalized"] = (
        (adata_st_align.obsm["spatial"] - st_xmin) / (st_xmax - st_xmin) * 100
    )

    adata_sm_align = filter_zero_total_obs(adata_sm_selected.copy())
    sc.pp.normalize_total(adata_sm_align, target_sum=1e3)
    sm_xmin = adata_sm_align.obsm["spatial"][:, 0].min()
    sm_xmax = adata_sm_align.obsm["spatial"][:, 0].max()
    smt.pp.spot_transform_by_manual(
        adata=adata_sm_align,
        rotation=rotation_degrees,
        spatial_key_SM="spatial",
        new_spatial_key_SM="spatial",
    )
    adata_sm_align.obsm["spatial_normalized"] = (
        (adata_sm_align.obsm["spatial"] - sm_xmin) / (sm_xmax - sm_xmin) * 100
    )
    adata_sm_selected = adata_sm_selected[adata_sm_align.obs_names].copy()
    return adata_st_selected, adata_sm_selected, adata_st_align, adata_sm_align


def run_alignment_for_joint(
    adata_st_align,
    adata_sm_align,
    debug_path: str | Path,
    device: str = "cuda:1",
    n_latent: int = 10,
    max_epoch: int = 128,
):
    debug_path = Path(debug_path).expanduser()
    debug_path.mkdir(parents=True, exist_ok=True)
    model = smt.model.AlignmentModule(
        adata_st=adata_st_align,
        adata_sm=adata_sm_align,
        n_latent=n_latent,
        device=device,
    )
    model.fit_vae(max_epoch=max_epoch)
    rasterized = model.get_rasterized_feature_map()
    model.random_sample_inside_image_spot(rasterized)
    result = model.fit_alignment(
        rasterized,
        debug_path=str(debug_path),
        align_sm_spot_to="ST",
        align_sm_feature_to_st_feature=False,
    )
    best_key = min(result, key=lambda key: result[key]["loss"][-1])
    adata_sm_aligned = adata_sm_align.copy()
    adata_sm_aligned.obsm["spatial_transformed"] = result[best_key]["pointsIt"]
    return adata_sm_aligned


def get_visium_sample_key_for_joint(adata_st) -> str:
    spatial_uns = adata_st.uns.get("spatial")
    if not spatial_uns:
        raise KeyError("ST 数据缺少 uns['spatial']")
    return next(iter(spatial_uns.keys()))


def reassign_sm_to_st_grid(
    adata_sm_raw,
    adata_st_target,
    adata_sm_aligned,
    min_total_intensity_reassign: float = 0.0,
    n_neighbors: int = 5,
    dist_fold: float = 1.5,
):
    adata_sm_for_reassign = adata_sm_raw.copy()
    adata_sm_for_reassign.obsm = dict(adata_sm_aligned.obsm)
    adata_sm_for_reassign.uns["spatial"] = adata_st_target.uns["spatial"]

    sample_key = get_visium_sample_key_for_joint(adata_st_target)
    scale = adata_st_target.uns["spatial"][sample_key]["scalefactors"]["tissue_hires_scalef"]
    adata_sm_for_reassign.obsm["spatial_transformed_scaled"] = (
        adata_sm_for_reassign.obsm["spatial_transformed"] / scale
    )
    adata_sm_for_reassign.obsm["spatial"] = adata_sm_for_reassign.obsm["spatial_transformed_scaled"]

    st_dot_df = smt.pp.ST_spot_sample(adata_st_target, "spatial")
    min_dist = smt.pp.calculate_min_dist(adata_st_target)
    adata_sm_new, adata_st_new = smt.pp.spot_align_byknn(
        st_dot_df,
        adata_sm_for_reassign,
        adata_st_target,
        min_dist=min_dist,
        n_neighbors=n_neighbors,
        dist_fold=dist_fold,
    )
    smt.pp.calculate_qc_metrics_sm(adata_sm_new)
    if min_total_intensity_reassign > 0:
        adata_sm_new = smt.pp.filter_cells_sm(
            adata_sm_new,
            min_total_intensity=min_total_intensity_reassign,
        )
    return adata_sm_new, adata_st_new


def attach_reassigned_st_metadata(adata_st_base, adata_st_new):
    adata_st_joint = adata_st_base.copy()
    adata_st_joint.obs["x_coord"] = adata_st_new.obs["x_coord"].to_numpy()
    adata_st_joint.obs["y_coord"] = adata_st_new.obs["y_coord"].to_numpy()
    adata_st_joint.obs["spot_name"] = adata_st_new.obs["spot_name"].astype(str).to_numpy()
    adata_st_joint.obs["x_coord"] = adata_st_joint.obs["x_coord"].astype(int)
    adata_st_joint.obs["y_coord"] = adata_st_joint.obs["y_coord"].astype(int)
    return adata_st_joint


def build_joint_preselected_h5ad(
    adata_sm_new,
    adata_st_joint,
):
    joint_raw = smt.pp.joint_adata_sm_st(
        adata_SM_new=adata_sm_new,
        adata_ST_new=adata_st_joint,
    )
    joint_raw.layers["counts"] = joint_raw.X.copy()
    smt.pp.normalize_total_joint_adata_sm_st(
        joint_raw,
        target_sum_SM=1e4,
        target_sum_ST=1e4,
    )
    joint_raw.layers["normalized"] = joint_raw.X.copy()
    joint_raw.raw = joint_raw
    joint_raw.var["highly_variable_moranI"] = True
    harmonize_joint_obs_log1p_columns(joint_raw)
    return joint_raw


def preprocess_aligned_joint_adata(
    joint_adata,
    *,
    joint_top_genes: int = 2000,
    joint_top_metabolites: int = 800,
):
    """Run the official post-alignment SpatialMETA filtering pipeline."""
    joint_adata = joint_adata.copy()
    joint_adata = smt.pp.removeHSP_MT_RPL_DNAJ(joint_adata)
    joint_adata.layers["counts"] = joint_adata.X.copy()
    smt.pp.normalize_total_joint_adata_sm_st(
        joint_adata,
        target_sum_SM=1e4,
        target_sum_ST=1e4,
    )
    joint_adata.layers["normalized"] = joint_adata.X.copy()
    joint_adata.raw = joint_adata
    smt.pp.spatial_variable_joint_adata_sm_st(
        joint_adata,
        n_top_genes=joint_top_genes,
        n_top_metabolites=joint_top_metabolites,
        add_key="highly_variable_moranI",
    )
    joint_v = joint_adata[:, joint_adata.var["highly_variable_moranI"]].copy()
    harmonize_joint_obs_log1p_columns(joint_v)
    return joint_v


def harmonize_joint_obs_log1p_columns(joint_adata) -> None:
    obs = joint_adata.obs
    if "n_genes_by_counts" in obs.columns:
        obs["log1p_n_genes_by_counts"] = np.array(
            [math.log1p(int(value)) for value in obs["n_genes_by_counts"].to_numpy()],
            dtype=np.float64,
        )
    for source_key, target_key in [
        ("total_counts", "log1p_total_counts"),
        ("total_counts_mt", "log1p_total_counts_mt"),
    ]:
        if source_key not in obs.columns:
            continue
        value = obs[source_key].to_numpy(dtype=np.float32, copy=False)
        obs[target_key] = ne.evaluate("log1p(value)").astype(np.float32, copy=False)


def preprocess_sm_st_to_h5ad(
    sm_input_path: str | Path,
    st_input_path: str | Path,
    output_path: str | Path,
    device: str = "cuda:1",
    random_seed: int = 42,
    min_total_intensity_raw: float = 0.0,
    min_total_intensity_reassign: float = 0.0,
    min_counts: int = 0,
    min_genes: int = 0,
    align_max_epoch: int = 128,
    align_n_latent: int = 10,
    rotation_degrees: float = -90.0,
    n_neighbors: int = 5,
    dist_fold: float = 1.5,
    joint_top_genes: int = 2000,
    joint_top_metabolites: int = 800,
):
    seed_preprocess(random_seed)
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    adata_sm_raw, adata_st_raw = read_raw_sm_st(sm_input_path=sm_input_path, st_input_path=st_input_path)
    adata_sm_prep = preprocess_sm_for_joint(
        adata_sm_raw,
        min_total_intensity_raw=min_total_intensity_raw,
    )
    adata_st_prep = preprocess_st_for_joint(
        adata_st_raw,
        min_counts=min_counts,
        min_genes=min_genes,
    )
    adata_st_selected, adata_sm_selected, adata_st_align, adata_sm_align = prepare_alignment_inputs_for_joint(
        adata_st=adata_st_prep,
        adata_sm=adata_sm_prep,
        top_genes=joint_top_genes,
        top_metabolites=joint_top_metabolites,
        rotation_degrees=rotation_degrees,
    )
    adata_sm_aligned = run_alignment_for_joint(
        adata_st_align=adata_st_align,
        adata_sm_align=adata_sm_align,
        debug_path=output_path.parent / f"{output_path.stem}_alignment_debug",
        device=device,
        n_latent=align_n_latent,
        max_epoch=align_max_epoch,
    )
    adata_sm_new, adata_st_new = reassign_sm_to_st_grid(
        adata_sm_raw=adata_sm_selected,
        adata_st_target=adata_st_selected,
        adata_sm_aligned=adata_sm_aligned,
        min_total_intensity_reassign=min_total_intensity_reassign,
        n_neighbors=n_neighbors,
        dist_fold=dist_fold,
    )
    adata_st_joint = attach_reassigned_st_metadata(
        adata_st_base=adata_st_selected,
        adata_st_new=adata_st_new,
    )
    joint_v = build_joint_preselected_h5ad(
        adata_sm_new=adata_sm_new,
        adata_st_joint=adata_st_joint,
    )
    joint_v.write_h5ad(output_path)
    return joint_v


def preprocess_sm_st_adatas_to_h5ad(
    *,
    adata_sm_raw,
    adata_st_raw,
    output_path: str | Path,
    device: str = "cuda:1",
    random_seed: int = 42,
    min_total_intensity_raw: float = 0.0,
    min_total_intensity_reassign: float = 0.0,
    min_counts: int = 0,
    min_genes: int = 0,
    align_max_epoch: int = 128,
    align_n_latent: int = 10,
    rotation_degrees: float = -90.0,
    n_neighbors: int = 5,
    dist_fold: float = 1.5,
    joint_top_genes: int = 2000,
    joint_top_metabolites: int = 800,
):
    """Select final features once, then align and reassign those same features."""
    seed_preprocess(random_seed)
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata_sm_prep = preprocess_sm_for_joint(adata_sm_raw, min_total_intensity_raw=min_total_intensity_raw)
    adata_st_prep = preprocess_st_for_joint(adata_st_raw, min_counts=min_counts, min_genes=min_genes)
    adata_st_selected, adata_sm_selected, adata_st_align, adata_sm_align = prepare_alignment_inputs_for_joint(
        adata_st=adata_st_prep,
        adata_sm=adata_sm_prep,
        top_genes=joint_top_genes,
        top_metabolites=joint_top_metabolites,
        rotation_degrees=rotation_degrees,
    )
    adata_sm_aligned = run_alignment_for_joint(
        adata_st_align=adata_st_align,
        adata_sm_align=adata_sm_align,
        debug_path=output_path.parent / f"{output_path.stem}_alignment_debug",
        device=device,
        n_latent=align_n_latent,
        max_epoch=align_max_epoch,
    )
    adata_sm_new, adata_st_new = reassign_sm_to_st_grid(
        adata_sm_raw=adata_sm_selected,
        adata_st_target=adata_st_selected,
        adata_sm_aligned=adata_sm_aligned,
        min_total_intensity_reassign=min_total_intensity_reassign,
        n_neighbors=n_neighbors,
        dist_fold=dist_fold,
    )
    adata_st_joint = attach_reassigned_st_metadata(adata_st_selected, adata_st_new)
    joint_v = build_joint_preselected_h5ad(
        adata_sm_new=adata_sm_new,
        adata_st_joint=adata_st_joint,
    )
    joint_v.write_h5ad(output_path)
    return joint_v
