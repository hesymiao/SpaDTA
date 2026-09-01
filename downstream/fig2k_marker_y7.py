from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path("/data/user/hesy/projects/SpatialMETA")
SOURCE = PROJECT_ROOT / "spaDTA" / "downstream" / "fig2k_2.py"

OLD_MARKER_GROUPS = '''MARKER_GROUPS = {
    "vascular_endothelial": ["PLVAP", "VWF", "PECAM1", "RGS5"],
    "perivascular_stromal": ["MGP", "ACTA2", "COL1A1", "THBS1"],
    "tumor_hypoxia": ["KRT8", "EPCAM", "CA9", "NDUFA4L2"],
}'''

NEW_MARKER_GROUPS = '''MARKER_GROUPS = {
    "myeloid_apc_axis": ["APOC1", "APOE", "HLA-DQA1", "GPNMB"],
    "stromal_axis": ["IGFBP4"],
}'''

OLD_TOP_RECOVERY = '''TOP_RECOVERY_GENES = [
    "PLVAP",
    "ATP5F1E",
    "STAT1",
    "SLC7A7",
    "MGP",
    "THBS1",
]'''

NEW_TOP_RECOVERY = '''TOP_RECOVERY_GENES = [
    "APOC1",
    "APOE",
    "HLA-DQA1",
    "GPNMB",
    "IGFBP4",
]'''


def main() -> None:
    code = SOURCE.read_text(encoding="utf-8")
    replacements = {
        "from spaDTA.model.preprocess import normalize_total_joint_adata_sm_st": (
            "from SpaDTA_718.model.preprocess import normalize_total_joint_adata_sm_st"
        ),
        'sample_name = "X49_T"': 'sample_name = "Y7_T"',
        'fig2k_root = project_root / "spaDTA" / "runs" / "downstream_runs" / "fig2k"': (
            'fig2k_root = project_root / "SpaDTA_718" / "runs" / "sm_downstream" / "fig2k_Y7_T"'
        ),
        OLD_MARKER_GROUPS: NEW_MARKER_GROUPS,
        OLD_TOP_RECOVERY: NEW_TOP_RECOVERY,
    }
    for old, new in replacements.items():
        if old not in code:
            raise RuntimeError(f"Expected snippet not found in {SOURCE}: {old[:80]}")
        code = code.replace(old, new, 1)

    exec_globals = {"__name__": "__main__", "__file__": str(Path(__file__).resolve())}
    exec(compile(code, str(SOURCE), "exec"), exec_globals)


if __name__ == "__main__":
    main()
