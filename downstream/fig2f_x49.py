from __future__ import annotations
from pathlib import Path
source_script = Path(__file__).with_name("fig2f.py")
source = source_script.read_text(encoding="utf-8").replace('output_dir = run_root / "fig2f"', 'output_dir = run_root / "fig2f_X49_T"')
exec(compile(source, str(source_script), "exec"), {"__name__": __name__, "__file__": str(source_script)})
