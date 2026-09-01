from __future__ import annotations

from pathlib import Path

source_script = Path(__file__).with_name("fig2_b.py")
source = source_script.read_text(encoding="utf-8")
source = source.replace('sample_name = "m3_FMP"', 'sample_name = "X49_T"')
exec(compile(source, str(source_script), "exec"), {"__name__": __name__, "__file__": str(source_script)})
