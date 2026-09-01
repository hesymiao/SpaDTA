from __future__ import annotations

from pathlib import Path


source_script = Path(__file__).with_name("fig2e_undergraph.py")
source = source_script.read_text(encoding="utf-8")
source = source.replace('sample = "X49_T"', 'sample = "Y27_T"')
source = source.replace('output_dir = run_root / "fig2e"', 'output_dir = run_root / "fig2e_Y27_T"')
source = source.replace('major_value = "Imm"', 'major_value = "Stro"', 1)
exec(compile(source, str(source_script), "exec"), {"__name__": __name__, "__file__": str(source_script)})
