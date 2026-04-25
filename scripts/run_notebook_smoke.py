from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
import traceback
from pathlib import Path

from IPython.display import display


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOKS = [
    "notebooks/01_data_understanding.ipynb",
    "notebooks/02_smartwatch_cleaning.ipynb",
    "notebooks/03_windowing_and_features.ipynb",
    "notebooks/04_label_preparation.ipynb",
    "notebooks/05_baseline_model_local_or_colab.ipynb",
    "notebooks/06_improved_baselines_tier1.ipynb",
]


def is_shell_or_magic_cell(source: str) -> bool:
    return any(line.lstrip().startswith(("!", "%")) for line in source.splitlines())


def run_notebook(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {
        "__name__": "__notebook__",
        "__file__": str(path),
        "display": display,
    }
    os.chdir(REPO_ROOT)
    print(f"\n=== {path.relative_to(REPO_ROOT)} ===")

    code_cell_number = 0
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        code_cell_number += 1
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        if is_shell_or_magic_cell(source):
            print(f"SKIP cell {code_cell_number}: shell/magic cell")
            continue

        print(f"RUN cell {code_cell_number}")
        try:
            exec(compile(source, f"{path}:{code_cell_number}", "exec"), namespace)
        except Exception:
            print(f"FAILED {path.relative_to(REPO_ROOT)} cell {code_cell_number}")
            traceback.print_exc()
            raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-run project notebooks.")
    parser.add_argument("notebooks", nargs="*", default=DEFAULT_NOTEBOOKS)
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT))
    for notebook in args.notebooks:
        run_notebook(REPO_ROOT / notebook)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
