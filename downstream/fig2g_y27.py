from __future__ import annotations

from pathlib import Path


project_root = Path("/data/user/hesy/projects/SpatialMETA")
source_script = project_root / "spaDTA" / "downstream" / "fig2g.py"
source = source_script.read_text(encoding="utf-8")
source = source.replace(
    'input_h5ad = project_root / "spaDTA" / "runs" / "first" / sample / f"{sample}_output.h5ad"',
    'input_h5ad = project_root / "SpaDTA_718" / "runs" / "sm_downstream" / "inputs" / sample / f"{sample}_output.h5ad"',
)
source = source.replace(
    'output_dir = project_root / "spaDTA" / "runs" / "downstream_runs" / "fig2g"',
    'output_dir = project_root / "SpaDTA_718" / "runs" / "sm_downstream" / "fig2g_Y27_T"',
)
source = source.replace(
    'de_input_table_dir = project_root / "spaDTA" / "runs" / "downstream_runs" / "fig2f" / "tables"',
    'de_input_table_dir = project_root / "SpaDTA_718" / "runs" / "sm_downstream" / "fig2f_Y27_T" / "tables"',
)
source = source.replace('subtype_value = "Stro_1"', 'subtype_value = "Stro_2"')
exec(compile(source, str(source_script), "exec"), {"__name__": __name__, "__file__": str(source_script)})
