import argparse
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import read_h5ad
from scipy import sparse
from scipy.sparse import dia_matrix
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from SpaDTA_718.model.preprocess import prepare_spadta_model_input


paired_root = Path("/bigdat2/user/hesy/spatialmeta/SpatialMETA/smart/SMART_data")
output_root = Path("/bigdat2/user/hesy/spatialmeta/SpatialMETA/SpaDTA_718_model_input/ATAC")
sample = "Mouse_Brain_E18_S1"

rna_file = "adata_RNA.h5ad"
atac_file = "adata_ATAC.h5ad"
top_rna_genes = 1500
min_rna_cells = 10
min_rna_genes = 200
spatial_tolerance = 1e-6
atac_pca_components = 60
pca_random_seed = 42


def to_csr_float32(values):
    if sparse.issparse(values):
        return values.tocsr().astype(np.float32)
    return sparse.csr_matrix(np.asarray(values, dtype=np.float32))


def ensure_name_column(var):
    var = var.copy()
    if "name" not in var.columns:
        var["name"] = var.index.astype(str)
    else:
        var["name"] = var["name"].astype(str)
    return var


def get_spatial_matrix(adata):
    if "spatial" not in adata.obsm:
        raise KeyError("input adata must contain obsm['spatial']")
    spatial = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    if spatial.ndim != 2 or spatial.shape[1] < 2:
        raise ValueError(f"spatial coordinates must be 2D with >= 2 columns, got {spatial.shape}")
    return spatial[:, :2].astype(np.float32, copy=False)


def resolve_job(sample_name, target_root):
    sample_dir = paired_root / sample_name
    if not sample_dir.exists() and paired_root.name == sample_name:
        sample_dir = paired_root
    if not sample_dir.exists():
        raise FileNotFoundError(f"Paired input directory not found for sample {sample_name}: {sample_dir}")
    return sample_dir / rna_file, sample_dir / atac_file, target_root / f"{sample_name}.h5ad"


def select_rna_feature_names(adata_rna):
    selection = adata_rna.copy()
    selection.var_names_make_unique()
    selection.X = to_csr_float32(selection.X)
    sc.pp.filter_genes(selection, min_cells=min_rna_cells)
    sc.pp.filter_cells(selection, min_genes=min_rna_genes)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.pp.highly_variable_genes(selection, n_top_genes=top_rna_genes, flavor="seurat_v3")
    return selection.var_names[selection.var["highly_variable"]].astype(str).tolist()


def smart_tfidf(adata, scale_factor=1e4):
    counts = adata.X
    n_peaks = np.asarray(counts.sum(axis=1)).reshape(-1)
    inverse_depth = dia_matrix((1.0 / n_peaks, 0), shape=(n_peaks.size, n_peaks.size))
    tf = inverse_depth @ counts
    tf = np.log1p(tf * float(scale_factor))
    idf = np.log1p(np.asarray(adata.shape[0] / counts.sum(axis=0)).reshape(-1))
    idf_matrix = dia_matrix((idf, 0), shape=(idf.size, idf.size))
    adata.X = np.nan_to_num(tf @ idf_matrix, 0)


def preprocess_atac_smart(adata_atac):
    adata_atac = adata_atac.copy()
    adata_atac.var_names_make_unique()
    adata_atac.X = to_csr_float32(adata_atac.X)
    adata_atac.var = ensure_name_column(adata_atac.var)
    atac_spatial = get_spatial_matrix(adata_atac)
    smart_tfidf(adata_atac)
    sc.pp.normalize_per_cell(adata_atac, counts_per_cell_after=1e4)
    sc.pp.log1p(adata_atac)
    matrix = adata_atac.X.toarray() if sparse.issparse(adata_atac.X) else np.asarray(adata_atac.X)
    features = PCA(
        n_components=atac_pca_components,
        random_state=pca_random_seed,
    ).fit_transform(matrix).astype(np.float32)
    var = pd.DataFrame(index=[f"ATAC_PC_{index + 1}" for index in range(features.shape[1])])
    var["name"] = var.index.astype(str)
    var["type"] = "SM"
    result = sc.AnnData(X=features, obs=adata_atac.obs.copy(), var=var)
    result.obsm["spatial"] = atac_spatial
    return result


def build_joint_rna_atac(rna_path, atac_path):
    adata_rna_raw = read_h5ad(rna_path)
    adata_rna_raw.obs_names = adata_rna_raw.obs_names.astype(str)
    adata_rna_raw.var_names_make_unique()
    adata_rna_raw.X = to_csr_float32(adata_rna_raw.X)
    selected_rna_vars = select_rna_feature_names(adata_rna_raw)

    adata_atac = preprocess_atac_smart(read_h5ad(atac_path))
    adata_atac.obs_names = adata_atac.obs_names.astype(str)
    shared_obs_names = adata_rna_raw.obs_names.intersection(adata_atac.obs_names)
    if len(shared_obs_names) == 0:
        raise ValueError("RNA and ATAC do not share any obs_names.")

    adata_rna = adata_rna_raw[shared_obs_names, selected_rna_vars].copy()
    adata_atac = adata_atac[shared_obs_names].copy()
    rna_spatial = get_spatial_matrix(adata_rna)
    atac_spatial = get_spatial_matrix(adata_atac)
    max_spatial_diff = float(np.abs(rna_spatial - atac_spatial).max())
    if max_spatial_diff > spatial_tolerance:
        raise ValueError(
            f"RNA/ATAC spatial mismatch too large: max_abs_diff={max_spatial_diff:.6f}, "
            f"tolerance={spatial_tolerance:.6f}"
        )

    var_rna = ensure_name_column(adata_rna.var)
    var_rna["type"] = "ST"
    var_atac = ensure_name_column(adata_atac.var)
    var_atac["type"] = "SM"
    var = pd.concat([var_rna, var_atac], axis=0)
    var.index = var.index.astype(str)
    for column in var.select_dtypes(include=["object"]).columns:
        var[column] = var[column].fillna("").astype(str)

    joint_x = sparse.hstack(
        [to_csr_float32(adata_rna.X), to_csr_float32(adata_atac.X)],
        format="csr",
        dtype=np.float32,
    )
    joint_adata = sc.AnnData(X=joint_x.copy(), obs=adata_rna.obs.copy(), var=var)
    joint_adata.obsm["spatial"] = rna_spatial
    joint_adata.layers["normalized"] = joint_x.copy()
    joint_adata.uns["joint_input_modalities"] = {
        "st_modality": "RNA",
        "sm_modality": "ATAC",
        "shared_obs_count": int(joint_adata.n_obs),
        "st_feature_count": int(adata_rna.n_vars),
        "sm_feature_count": int(adata_atac.n_vars),
        "st_top_genes": int(top_rna_genes),
        "max_spatial_diff": max_spatial_diff,
        "source": "SpaDTA_718 tutorial.preprocess_atac RNA raw HVG + SMART-style ATAC preprocessing",
        "rna_representation": "raw_counts_hvg_for_zinb",
        "atac_representation": "smart_tfidf_normalize_per_cell_1e4_log1p_pca",
        "atac_pca_components": int(atac_pca_components),
        "atac_pca_random_seed": int(pca_random_seed),
        "trainer_expected_input": "feature_input_mode_false_rna_raw_hvg_atac_smart_pca",
    }
    return joint_adata


def run_sample(sample_name, target_root=output_root):
    rna_path, atac_path, output_path = resolve_job(sample_name, target_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[job] sample={sample_name} rna_path={rna_path} atac_path={atac_path} output_path={output_path}",
        flush=True,
    )
    adata = prepare_spadta_model_input(
        build_joint_rna_atac(rna_path, atac_path),
        modality="atac",
        expression_graph_k=4,
        spatial_context_k=12,
    )
    adata.write_h5ad(output_path)
    print(f"[done] sample={sample_name} shape={adata.shape} output_path={output_path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess one RNA+ATAC sample for SpaDTA.")
    parser.add_argument("--sample-name", default=sample)
    parser.add_argument("--output-root", type=Path, default=output_root)
    return parser.parse_args()


def main():
    args = parse_args()
    run_sample(args.sample_name, args.output_root)


if __name__ == "__main__":
    main()
