"""Launch the v2 metric-selection protocol from one entry point.

This script does not add new modeling behavior.  It only runs the existing
v2 phases in the order needed for the PR-AUC-first selection plan.

Examples
--------
Local checks only::

    python scripts/v2/run_metric_protocol.py check

Local tabular sweep::

    python scripts/v2/run_metric_protocol.py local-tabular

Heavy Colab/GPU deep-learning sweep::

    python scripts/v2/run_metric_protocol.py colab-dl

Full v2 protocol, usually on Colab::

    python scripts/v2/run_metric_protocol.py full-v2
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable


def _module_cmd(module: str, *args: str) -> list[str]:
    return [PYTHON, "-m", module, *args]


def _compile_cmd() -> list[str]:
    return [PYTHON, "-m", "compileall", "src/event_v2", "scripts/v2"]


def _display_cmd(cmd: Iterable[str]) -> str:
    return subprocess.list2cmdline(list(cmd))


def _csv_override(value: str | None, quick_value: str, full_value: str, quick: bool) -> str:
    if value:
        return value
    return quick_value if quick else full_value


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    quick = args.quick or args.profile == "smoke"
    thresholds = _csv_override(args.thresholds, "3", "2,3,4", quick)
    lengths = _csv_override(args.lengths, "7", "3,7,14", quick)
    washouts = _csv_override(args.washouts, "7", "0,7,14", quick)
    algos = _csv_override(args.algos, "lr", "lr,rf,xgb", quick)
    archs = _csv_override(args.archs, "gru", "gru,lstm,rnn,cnn", quick)
    multi_seeds = _csv_override(args.multi_seeds, "42,43", "42,43,44,45,46", quick)
    epochs = str(args.epochs if args.epochs is not None else (3 if quick else 30))
    patience = str(args.patience if args.patience is not None else (2 if quick else 5))
    n_shuffles = str(args.n_shuffles if args.n_shuffles is not None else (2 if quick else 5))

    tabular = _module_cmd(
        "scripts.v2.tune_tabular_v2",
        "--thresholds",
        thresholds,
        "--lengths",
        lengths,
        "--washouts",
        washouts,
        "--algos",
        algos,
        "--seed",
        str(args.seed),
    )
    if args.reuse_features:
        tabular.append("--reuse-features")

    dl = _module_cmd(
        "scripts.v2.tune_dl_v2",
        "--thresholds",
        thresholds,
        "--lengths",
        lengths,
        "--washouts",
        washouts,
        "--archs",
        archs,
        "--seed",
        str(args.seed),
        "--multi-seeds",
        multi_seeds,
        "--epochs",
        epochs,
        "--batch-size",
        str(args.batch_size),
        "--patience",
        patience,
        "--learning-rate",
        str(args.learning_rate),
    )
    if args.skip_multi_seed:
        dl.append("--skip-multi-seed")

    leakage = _module_cmd(
        "scripts.v2.leakage_probe_v2",
        "--n-shuffles",
        n_shuffles,
        "--seed",
        str(args.seed),
    )

    sensor_ablation = _module_cmd("scripts.v2.sensor_ablation_v2", "--seed", str(args.seed))

    final = _module_cmd(
        "scripts.v2.final_test_eval_v2",
        "--seed",
        str(args.seed),
        "--dl-batch-size",
        str(args.batch_size),
        "--dl-patience",
        patience,
        "--dl-learning-rate",
        str(args.learning_rate),
    )
    if args.skip_leakage_gate:
        final.append("--skip-leakage-gate")

    analysis = _module_cmd("scripts.v2.analyze_v2", "--n-boot", str(args.analysis_n_boot))

    profiles = {
        "check": [_compile_cmd()],
        "smoke": [_compile_cmd(), tabular, leakage, sensor_ablation, final, analysis],
        "local-tabular": [_compile_cmd(), tabular],
        "local-gates": [leakage, sensor_ablation],
        "colab-dl": [_compile_cmd(), dl],
        "finalize": [final, analysis],
        "full-v2": [_compile_cmd(), tabular, dl, leakage, sensor_ablation, final, analysis],
    }
    return profiles[args.profile]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "profile",
        choices=[
            "check",
            "smoke",
            "local-tabular",
            "local-gates",
            "colab-dl",
            "finalize",
            "full-v2",
        ],
        help="Which protocol phase to run.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use tiny smoke-test defaults: T=3, L=7, W=7, LR only, GRU only.",
    )
    parser.add_argument("--thresholds", help="Comma-separated event thresholds. Default: 2,3,4.")
    parser.add_argument("--lengths", help="Comma-separated history lengths in days. Default: 3,7,14.")
    parser.add_argument("--washouts", help="Comma-separated washout lengths in days. Default: 0,7,14.")
    parser.add_argument("--algos", help="Comma-separated tabular algos from {lr,rf,xgb}.")
    parser.add_argument("--archs", help="Comma-separated DL archs from {gru,lstm,rnn,cnn}.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi-seeds", help="Comma-separated seeds for DL winner refits.")
    parser.add_argument("--epochs", type=int, help="DL epochs. Default: 30, or 3 with --quick.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, help="DL early-stopping patience. Default: 5, or 2 with --quick.")
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--n-shuffles", type=int, help="Leakage-probe shuffle count. Default: 5, or 2 with --quick.")
    parser.add_argument("--analysis-n-boot", type=int, default=2000)
    parser.add_argument(
        "--reuse-features",
        action="store_true",
        help="Pass --reuse-features to the tabular sweep.",
    )
    parser.add_argument(
        "--skip-multi-seed",
        action="store_true",
        help="Pass --skip-multi-seed to DL HPO. Do not use before final reporting.",
    )
    parser.add_argument(
        "--skip-leakage-gate",
        action="store_true",
        help="Let final_test_eval_v2 run even if the leakage summary is missing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    commands = build_commands(args)

    quick = args.quick or args.profile == "smoke"
    print(f"[v2-protocol] profile={args.profile} quick={quick} dry_run={args.dry_run}")
    print(f"[v2-protocol] cwd={PROJECT_ROOT}")
    for index, cmd in enumerate(commands, start=1):
        print(f"\n[{index}/{len(commands)}] {_display_cmd(cmd)}")
        if not args.dry_run:
            subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
