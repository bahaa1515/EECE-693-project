from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import DATA_PROCESSED, DATASET_DIR, OUTPUT_TABLES
from src.event_features import build_and_write_feature_tables
from src.event_labels import DEFAULT_THRESHOLDS, run_event_labeling
from src.event_modeling import train_selected_feature_tables
from src.event_samples import (
    DEFAULT_INPUT_LENGTHS,
    DEFAULT_WASHOUT_DAYS,
    build_all_sample_indexes,
    write_sample_indexes,
)


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run questionnaire event-episode labeling and sensor training pipeline."
    )
    parser.add_argument(
        "--thresholds",
        default=",".join(str(v) for v in DEFAULT_THRESHOLDS),
        help="Comma-separated daily score thresholds.",
    )
    parser.add_argument(
        "--input-lengths",
        default=",".join(str(v) for v in DEFAULT_INPUT_LENGTHS),
        help="Comma-separated input window lengths in days.",
    )
    parser.add_argument(
        "--washouts",
        default=",".join(str(v) for v in DEFAULT_WASHOUT_DAYS),
        help="Comma-separated washout lengths in days.",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Only write label and sample-index outputs.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Write labels/features but do not train tabular models.",
    )
    parser.add_argument(
        "--train-all",
        action="store_true",
        help="Train every threshold/input-length/washout combination.",
    )
    parser.add_argument("--train-threshold", type=int, default=3)
    parser.add_argument("--train-length", type=int, default=7)
    parser.add_argument("--train-washout", type=int, default=7)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    thresholds = parse_int_list(args.thresholds)
    input_lengths = parse_int_list(args.input_lengths)
    washouts = parse_int_list(args.washouts)

    print(f"Dataset directory: {DATASET_DIR}")
    print(f"Processed data directory: {DATA_PROCESSED}")
    print(f"Output tables directory: {OUTPUT_TABLES}")

    print("\n[1/4] Building questionnaire-only event labels...")
    label_artifacts = run_event_labeling(thresholds=thresholds)
    print(label_artifacts.summary.to_string(index=False))

    print("\n[2/4] Building positive/negative sample indexes...")
    sample_indexes, sample_counts = build_all_sample_indexes(
        weekly_events=label_artifacts.weekly_events,
        probable_days_by_threshold=label_artifacts.probable_event_days,
        episodes_by_threshold=label_artifacts.event_episodes,
        input_lengths=input_lengths,
        washout_values=washouts,
    )
    write_sample_indexes(sample_indexes, sample_counts)
    print(sample_counts.to_string(index=False))

    if args.skip_features:
        print("\nSkipped feature extraction.")
        return 0

    print("\n[3/4] Cleaning sensors and building merged feature tables...")
    _, feature_paths = build_and_write_feature_tables(sample_indexes)
    print(f"Wrote {len(feature_paths)} sensor feature tables.")

    if args.skip_training:
        print("\nSkipped training.")
        return 0

    if args.train_all:
        selected_keys = sorted(feature_paths)
    else:
        selected_keys = [(args.train_threshold, args.train_length, args.train_washout)]
        missing = [key for key in selected_keys if key not in feature_paths]
        if missing:
            raise KeyError(f"Selected training configuration was not generated: {missing}")

    print("\n[4/4] Training tabular models...")
    results = train_selected_feature_tables(
        feature_paths=feature_paths,
        selected_keys=selected_keys,
        random_state=args.random_state,
    )
    if results.empty:
        print("No model results were produced.")
    else:
        print(results.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
