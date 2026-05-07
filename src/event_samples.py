from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DATA_PROCESSED, OUTPUT_TABLES
from .event_labels import USER_COL


DEFAULT_INPUT_LENGTHS = [1, 3, 5, 7, 14, 28]
DEFAULT_WASHOUT_DAYS = [0, 7, 14]
LABEL_STRATEGY = "questionnaire_event_episode_labeling"


def _week_day_set(weekly_events: pd.DataFrame, positive: bool) -> set[tuple[int, int]]:
    selected = weekly_events[
        weekly_events["broad_event_week"].eq(positive)
        & weekly_events["has_valid_covered_days"]
    ]
    days: set[tuple[int, int]] = set()
    for row in selected.itertuples(index=False):
        user_key = int(getattr(row, USER_COL))
        for day in range(int(row.week_start_day), int(row.week_end_day) + 1):
            days.add((user_key, day))
    return days


def _event_day_set(probable_days: pd.DataFrame) -> set[tuple[int, int]]:
    return {
        (int(row.user_key), int(row.event_day))
        for row in probable_days.itertuples(index=False)
    }


def _episode_spans(episodes: pd.DataFrame) -> list[tuple[int, int, int]]:
    return [
        (int(row.user_key), int(row.event_onset_day), int(row.event_end_day))
        for row in episodes.itertuples(index=False)
    ]


def _washout_day_set(
    episodes: pd.DataFrame, washout_days: int
) -> set[tuple[int, int]]:
    if washout_days <= 0 or episodes.empty:
        return set()
    days: set[tuple[int, int]] = set()
    for row in episodes.itertuples(index=False):
        user_key = int(row.user_key)
        start = int(row.event_end_day) + 1
        end = int(row.event_end_day) + washout_days
        for day in range(start, end + 1):
            days.add((user_key, day))
    return days


def _window_days(user_key: int, start_day: int, end_day: int) -> set[tuple[int, int]]:
    return {(user_key, day) for day in range(start_day, end_day + 1)}


def build_sample_index(
    weekly_events: pd.DataFrame,
    probable_days: pd.DataFrame,
    episodes: pd.DataFrame,
    input_length_days: int,
    washout_days: int,
    threshold: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Create positive onset samples and stable negative samples."""
    positive_week_days = _week_day_set(weekly_events, positive=True)
    negative_week_days = _week_day_set(weekly_events, positive=False)
    stable_negative_anchor_days = sorted(negative_week_days - positive_week_days)
    event_days = _event_day_set(probable_days)
    event_spans = _episode_spans(episodes)
    washout_days_set = _washout_day_set(episodes, washout_days)

    rows: list[dict[str, object]] = []
    positive_lost_history = 0

    for episode in episodes.itertuples(index=False):
        user_key = int(episode.user_key)
        onset_day = int(episode.event_onset_day)
        window_start = onset_day - input_length_days
        window_end = onset_day - 1
        if window_start < 0:
            positive_lost_history += 1
            continue
        rows.append(
            {
                USER_COL: user_key,
                "sample_anchor_day": onset_day,
                "window_start_day": window_start,
                "window_end_day": window_end,
                "event_onset_day": onset_day,
                "event_end_day": int(episode.event_end_day),
                "target": 1,
                "sample_type": "positive_onset",
                "label_strategy": LABEL_STRATEGY,
                "threshold": threshold,
                "input_length_days": input_length_days,
                "washout_days": washout_days,
            }
        )

    negative_candidates = 0
    negative_excluded_positive_overlap = 0
    negative_excluded_event_overlap = 0
    negative_excluded_washout = 0

    for user_key, anchor_day in stable_negative_anchor_days:
        window_start = anchor_day - input_length_days
        window_end = anchor_day - 1
        if window_start < 0:
            continue

        negative_candidates += 1
        input_days = _window_days(user_key, window_start, window_end)
        input_and_target_days = set(input_days)
        input_and_target_days.add((user_key, anchor_day))

        if input_and_target_days & positive_week_days:
            negative_excluded_positive_overlap += 1
            continue
        if input_and_target_days & event_days:
            negative_excluded_event_overlap += 1
            continue
        overlaps_episode = any(
            episode_user == user_key
            and not (window_end < episode_start or window_start > episode_end)
            for episode_user, episode_start, episode_end in event_spans
        )
        if overlaps_episode:
            negative_excluded_event_overlap += 1
            continue
        if input_and_target_days & washout_days_set:
            negative_excluded_washout += 1
            continue

        rows.append(
            {
                USER_COL: user_key,
                "sample_anchor_day": anchor_day,
                "window_start_day": window_start,
                "window_end_day": window_end,
                "event_onset_day": pd.NA,
                "event_end_day": pd.NA,
                "target": 0,
                "sample_type": "stable_negative",
                "label_strategy": LABEL_STRATEGY,
                "threshold": threshold,
                "input_length_days": input_length_days,
                "washout_days": washout_days,
            }
        )

    sample_index = pd.DataFrame(rows)
    if not sample_index.empty:
        sample_index = sample_index.sort_values(
            [USER_COL, "sample_anchor_day", "target"]
        ).reset_index(drop=True)

    counts = {
        "threshold": threshold,
        "input_length_days": input_length_days,
        "washout_days": washout_days,
        "positive_candidates": len(episodes),
        "positive_samples": int(sample_index["target"].eq(1).sum())
        if not sample_index.empty
        else 0,
        "positive_excluded_not_enough_history": positive_lost_history,
        "negative_candidates": negative_candidates,
        "negative_samples": int(sample_index["target"].eq(0).sum())
        if not sample_index.empty
        else 0,
        "negative_excluded_positive_week_overlap": negative_excluded_positive_overlap,
        "negative_excluded_event_overlap": negative_excluded_event_overlap,
        "negative_excluded_washout": negative_excluded_washout,
        "total_samples": len(sample_index),
    }
    counts["negative_excluded_total"] = (
        counts["negative_excluded_positive_week_overlap"]
        + counts["negative_excluded_event_overlap"]
        + counts["negative_excluded_washout"]
    )
    return sample_index, counts


def build_all_sample_indexes(
    weekly_events: pd.DataFrame,
    probable_days_by_threshold: dict[int, pd.DataFrame],
    episodes_by_threshold: dict[int, pd.DataFrame],
    input_lengths: list[int] | None = None,
    washout_values: list[int] | None = None,
) -> tuple[dict[tuple[int, int, int], pd.DataFrame], pd.DataFrame]:
    input_lengths = input_lengths or DEFAULT_INPUT_LENGTHS
    washout_values = washout_values or DEFAULT_WASHOUT_DAYS
    sample_indexes: dict[tuple[int, int, int], pd.DataFrame] = {}
    count_rows: list[dict[str, int]] = []

    for threshold, probable_days in probable_days_by_threshold.items():
        episodes = episodes_by_threshold[threshold]
        for input_length_days in input_lengths:
            for washout_days in washout_values:
                key = (threshold, input_length_days, washout_days)
                sample_index, counts = build_sample_index(
                    weekly_events=weekly_events,
                    probable_days=probable_days,
                    episodes=episodes,
                    input_length_days=input_length_days,
                    washout_days=washout_days,
                    threshold=threshold,
                )
                sample_indexes[key] = sample_index
                count_rows.append(counts)

    return sample_indexes, pd.DataFrame(count_rows)


def sample_index_filename(threshold: int, input_length_days: int, washout_days: int) -> str:
    return (
        "questionnaire_event_samples_"
        f"L{input_length_days}_threshold{threshold}_washout{washout_days}.parquet"
    )


def write_sample_indexes(
    sample_indexes: dict[tuple[int, int, int], pd.DataFrame],
    sample_counts: pd.DataFrame,
    output_dir: Path = DATA_PROCESSED,
    table_dir: Path = OUTPUT_TABLES,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    for (threshold, input_length_days, washout_days), sample_index in sample_indexes.items():
        sample_index.to_parquet(
            output_dir
            / sample_index_filename(threshold, input_length_days, washout_days),
            index=False,
        )
    sample_counts.to_csv(table_dir / "sample_counts_by_strategy.csv", index=False)
