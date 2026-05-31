__all__ = [
    "DecAlignSpatialMetaLinear",
    "normalize_total_joint_adata_sm_st",
    "TrainingResult",
    "SingleRunResult",
    "train_spatial_model",
    "evaluate_clustering",
    "run_single_sample",
    "run_all_samples",
    "run_parallel_jobs",
    "compare_run_outputs",
]


def __getattr__(name):
    if name == "DecAlignSpatialMetaLinear":
        from .model import DecAlignSpatialMetaLinear

        return DecAlignSpatialMetaLinear
    if name == "normalize_total_joint_adata_sm_st":
        from .preprocess import normalize_total_joint_adata_sm_st

        return normalize_total_joint_adata_sm_st
    if name in {
        "train_spatial_model",
        "evaluate_clustering",
        "run_single_sample",
        "run_all_samples",
        "run_parallel_jobs",
        "compare_run_outputs",
        "TrainingResult",
        "SingleRunResult",
    }:
        from . import workflow

        return getattr(workflow, name)
    raise AttributeError(name)
