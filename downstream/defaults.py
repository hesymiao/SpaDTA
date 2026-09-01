from pathlib import Path
import sys


DOWNSTREAM_CODE_DIR = Path(__file__).resolve().parent
OURS_ROOT = DOWNSTREAM_CODE_DIR.parent
RUNS_DIR = OURS_ROOT / "runs"
MODEL_RUNS_DIR = RUNS_DIR / "model_runs"
DOWNSTREAM_RUNS_DIR = RUNS_DIR / "downstream_runs"

DEFAULT_PYTHON = Path("/data/user/hesy/miniconda3/envs/spatialmeta/bin/python")
PYTHON_EXECUTABLE = DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable)
PROCESSED_DIR = Path("/bigdat2/user/hesy/spatialmeta/SpatialMETA/processed")
GT_DIR = Path("/bigdat2/user/hesy/spatialmeta/SpatialMETA/06_spatialmeta_groundtruth/06_spatialmeta_groundtruth")

DEFAULT_RUNTIME_LABEL = "current_localcontext_contrastive_w002_fixed_targetcount_20260427_localized"
DEFAULT_CONFIG_ID = "ablate_no_hete_homo_splitrecon_sharedhalf_balance"

SAMPLE_ORDER = [
    "248_T",
    "R114_T",
    "S15_T",
    "X49_T",
    "Y27_T",
    "Y7_T",
    "m1_FMP",
    "m3_FMP",
    "m4_FMP",
]
