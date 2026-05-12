"""End-to-end pipeline runner.

This is the single source of truth for the Tier-1 / Tier-2 / Tier-3 numbers
in the report.  Run it from the project root:

    .venv\\Scripts\\python.exe scripts/run_full_pipeline.py            # smoke (1 seed, 5 ep)
    .venv\\Scripts\\python.exe scripts/run_full_pipeline.py --full     # full Colab protocol

The full protocol (T4 GPU, ~25 min):
    - regenerates the event-onset labelled parquet
    - runs the leakage probe (gates the rest of the pipeline)
    - trains Tier-2 GRU/CNN across 5 seeds with patient-cluster bootstrap CIs
    - runs the 4-config Tier-3 multimodal ablation with Platt scaling and CIs
    - aggregates everything into outputs/tables/headline_results.csv

The smoke mode is intended for end-to-end testability on CPU in < 5 minutes;
it does NOT produce defensible numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import DATA_PROCESSED, OUTPUT_TABLES
from src.event_labels import run_event_labeling
from src.labels import run_labeling


def step(idx: int, total: int, msg: str) -> None:
    print(f"\n[{idx}/{total}] {msg}")
    print("=" * 70)


def run_pipeline(full: bool = False) -> int:
    set_seed_env()

    seeds = [42, 43, 44, 45, 46] if full else [42]
    epochs = 20 if full else 5
    patience = 5 if full else 2

    n_steps = 6
    t0 = time.time()

    step(1, n_steps, "Regenerate canonical (weekly-questionnaire) labels")
    # Canonical task: weekly-questionnaire OR-of-flags target (30% positive
    # rate, matches the progress-report task).  Event-onset labelling is run
    # too so the ablation parquet is up to date, but does NOT overwrite the
    # canonical labelled parquet.
    run_event_labeling()
    labeled = run_labeling()
    pos_rate = float(labeled["target_binary"].mean())
    print(f"  rows={len(labeled)}  users={labeled['user_key'].nunique()}  pos_rate={pos_rate:.3f}")

    step(2, n_steps, "Leakage probe (gate)")
    # Importing inside the function avoids running probe code at module load.
    from scripts.leakage_probe import main as run_leakage_probe  # noqa: PLC0415

    rc = run_leakage_probe()
    if rc != 0:
        print("Leakage probe FAILED — aborting pipeline.")
        return rc

    step(3, n_steps, f"Tier-2 sequence models ({len(seeds)} seed{'s' if len(seeds) > 1 else ''})")
    from src.deep_learning import run_deep_learning_experiment, run_deep_learning_multi_seed

    if len(seeds) == 1:
        run_deep_learning_experiment(
            epochs=epochs, batch_size=128, patience=patience, seed=seeds[0]
        )
    else:
        agg = run_deep_learning_multi_seed(
            seeds=seeds, epochs=epochs, batch_size=128, patience=patience
        )
        print(agg.to_string(index=False))

    step(4, n_steps, f"Tier-3 multimodal ablation ({len(seeds)} seed{'s' if len(seeds) > 1 else ''})")
    from src.deep_learning import run_multimodal_experiment, run_multimodal_multi_seed

    if len(seeds) == 1:
        mm_results = run_multimodal_experiment(
            epochs=epochs, batch_size=64, patience=patience, random_state=seeds[0]
        )
        print(mm_results.to_string(index=False))
    else:
        mm_agg = run_multimodal_multi_seed(
            seeds=seeds, epochs=epochs, batch_size=64, patience=patience
        )
        print(mm_agg.to_string(index=False))

    step(5, n_steps, "Aggregate headline results")
    headline_path = OUTPUT_TABLES / "headline_results.csv"
    aggregate_headline(seeds, headline_path)

    step(6, n_steps, "Environment fingerprint")
    write_env_info(OUTPUT_TABLES / "env_info.json")

    print(f"\nTotal pipeline time: {time.time() - t0:.1f}s")
    print(f"Headline results → {headline_path}")
    return 0


def set_seed_env() -> None:
    import os

    os.environ.setdefault("PYTHONHASHSEED", "42")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def aggregate_headline(seeds: list[int], out_path: Path) -> None:
    """Collect Tier-2 (multi-seed if available) and Tier-3 results into one CSV."""
    tables = []

    # Tier-2: prefer multi-seed summary, fall back to single-seed.
    multi = OUTPUT_TABLES / "tier2_multi_seed_results_summary.csv"
    single = OUTPUT_TABLES / "deep_learning_reproduced_results.csv"
    if multi.exists():
        df = pd.read_csv(multi)
        df["tier"] = "Tier-2"
        df["source"] = "multi-seed"
        tables.append(df)
    elif single.exists():
        df = pd.read_csv(single)
        df["tier"] = "Tier-2"
        df["source"] = "single-seed"
        tables.append(df)

    # Tier-3: prefer multi-seed summary, fall back to single-seed.
    mm_multi = OUTPUT_TABLES / "multimodal_multi_seed_results_summary.csv"
    mm_single = OUTPUT_TABLES / "multimodal_results.csv"
    if mm_multi.exists():
        df = pd.read_csv(mm_multi)
        df["tier"] = "Tier-3"
        df["source"] = "multimodal-ablation-multi-seed"
        tables.append(df)
    elif mm_single.exists():
        df = pd.read_csv(mm_single)
        df["tier"] = "Tier-3"
        df["source"] = "multimodal-ablation-single-seed"
        tables.append(df)

    if not tables:
        print("  No result tables found — nothing to aggregate.")
        return
    headline = pd.concat(tables, ignore_index=True, sort=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headline.to_csv(out_path, index=False)
    print(f"  wrote {out_path} ({len(headline)} rows)")


def write_env_info(out_path: Path) -> None:
    import platform

    info: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for pkg in ("numpy", "pandas", "sklearn", "xgboost", "torch", "pyarrow"):
        try:
            mod = __import__(pkg)
            info[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            info[pkg] = "not-installed"
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cudnn_version"] = torch.backends.cudnn.version()
    except Exception:  # noqa: BLE001
        pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"  wrote {out_path}")
    print(json.dumps(info, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the full Colab protocol (5 seeds, 20 epochs). Default is smoke mode.",
    )
    args = parser.parse_args()
    return run_pipeline(full=args.full)


if __name__ == "__main__":
    raise SystemExit(main())
