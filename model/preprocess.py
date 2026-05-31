from __future__ import annotations

import numpy as np
import scanpy as sc
from scipy import sparse


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
    align_st_top: int = 2000,
    align_sm_top: int = 500,
    rotation_degrees: float = -90.0,
):
    adata_st_align = adata_st.copy()
    normalize_non_inplace(adata_st_align)
    smt.pp.spatial_variable(
        adata_st_align,
        n_top_variable=align_st_top,
        add_key="highly_variable_moranI",
        layer="normalized",
    )
    adata_st_align = adata_st_align[:, adata_st_align.var["highly_variable_moranI"]].copy()
    st_xmin = adata_st_align.obsm["spatial"][:, 0].min()
    st_xmax = adata_st_align.obsm["spatial"][:, 0].max()
    adata_st_align.obsm["spatial_normalized"] = (
        (adata_st_align.obsm["spatial"] - st_xmin) / (st_xmax - st_xmin) * 100
    )

    adata_sm_align = adata_sm.copy()
    sc.pp.normalize_total(adata_sm_align, target_sum=1e3)
    smt.pp.spatial_variable(
        adata_sm_align,
        n_top_variable=align_sm_top,
        add_key="highly_variable_moranI",
    )
    adata_sm_align = adata_sm_align[:, adata_sm_align.var["highly_variable_moranI"]].copy()
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
    return adata_st_align, adata_sm_align


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


def build_joint_processed_h5ad(
    adata_sm_new,
    adata_st_joint,
    joint_top_genes: int = 2000,
    joint_top_metabolites: int = 800,
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
    smt.pp.spatial_variable_joint_adata_sm_st(
        joint_raw,
        n_top_genes=joint_top_genes,
        n_top_metabolites=joint_top_metabolites,
        add_key="highly_variable_moranI",
    )
    joint_v = joint_raw[:, joint_raw.var["highly_variable_moranI"]].copy()
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
    align_st_top: int = 2000,
    align_sm_top: int = 500,
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
    adata_st_align, adata_sm_align = prepare_alignment_inputs_for_joint(
        adata_st=adata_st_prep,
        adata_sm=adata_sm_prep,
        align_st_top=align_st_top,
        align_sm_top=align_sm_top,
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
        adata_sm_raw=adata_sm_prep,
        adata_st_target=adata_st_prep,
        adata_sm_aligned=adata_sm_aligned,
        min_total_intensity_reassign=min_total_intensity_reassign,
        n_neighbors=n_neighbors,
        dist_fold=dist_fold,
    )
    adata_st_joint = attach_reassigned_st_metadata(
        adata_st_base=adata_st_prep,
        adata_st_new=adata_st_new,
    )
    joint_v = build_joint_processed_h5ad(
        adata_sm_new=adata_sm_new,
        adata_st_joint=adata_st_joint,
        joint_top_genes=joint_top_genes,
        joint_top_metabolites=joint_top_metabolites,
    )
    joint_v.write_h5ad(output_path)
    return joint_v
