from __future__ import annotations

from pathlib import Path


source_script = Path(__file__).with_name("fig2h.py")
source = source_script.read_text(encoding="utf-8")
source = source.replace(
    "'    left_edge = 2500.0\\n',",
    "'    left_edge = max(0.0, float(go_plot[\"combined_score\"].min()) - 4.0)\\n',",
)
source = source.replace('sample_name = "X49_T"', 'sample_name = "Y27_T"')
source = source.replace('major_name = "Imm"', 'major_name = "Stro"')
source = source.replace('subtype_name = "Imm_1"', 'subtype_name = "Stro_2"')
source = source.replace('prerequisite_script = project_root / "SpaDTA_718" / "downstream" / "fig2f.py"', 'prerequisite_script = project_root / "SpaDTA_718" / "downstream" / "fig2f_y27.py"')
source = source.replace('prerequisite_root = run_root / "fig2f"', 'prerequisite_root = run_root / "fig2f_Y27_T"')
source = source.replace('root_dir = run_root / "fig2h"', 'root_dir = run_root / "fig2h_Y27_T"')
exec(compile(source, str(source_script), "exec"), {"__name__": __name__, "__file__": str(source_script)})
