from __future__ import annotations

from pathlib import Path


SOURCE = Path(__file__).with_name("fig2k.py")


def main() -> None:
    code = SOURCE.read_text(encoding="utf-8")
    replacements = {
        'sample_name = "X49_T"': 'sample_name = "Y7_T"',
        'root_dir = project_root / "SpaDTA_718" / "runs" / "sm_downstream" / "fig2k"': (
            'root_dir = project_root / "SpaDTA_718" / "runs" / "sm_downstream" / "fig2k_Y7_T"'
        ),
    }
    for old, new in replacements.items():
        if old not in code:
            raise RuntimeError(f"Expected snippet not found in {SOURCE}: {old}")
        code = code.replace(old, new, 1)

    exec_globals = {"__name__": "__main__", "__file__": str(Path(__file__).resolve())}
    exec(compile(code, str(SOURCE), "exec"), exec_globals)


if __name__ == "__main__":
    main()
