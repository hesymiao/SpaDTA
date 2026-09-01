from __future__ import annotations

from pathlib import Path


source_script = Path(__file__).with_name("fig2k.py")
source = source_script.read_text(encoding="utf-8")
source = source.replace('sample_name = "X49_T"', 'sample_name = "Y27_T"')
source = source.replace(
    'train_input_h5ad = processed_root / f"{sample_name}.h5ad"',
    'train_input_h5ad = Path("/bigdat2/user/hesy/spatialmeta/SpatialMETA/SpaDTA_718_model_input_preselect800_20260719/SM/Y27_T.h5ad")',
)
source = source.replace('/ "fig2k"', '/ "fig2k_Y27_T"')
exec(compile(source, str(source_script), "exec"), {"__name__": __name__, "__file__": str(source_script)})
