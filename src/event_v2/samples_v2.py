"""Event-episode sample builder (v2).

Copy of ``src.event_samples`` with the same algorithm and an explicit
contract check that confirms:

* Every positive sample window is contained in ``[E - L, E - 1]`` and never
  includes day ``E`` itself.
* Every stable-negative window contains no event day, no positive event-week
  day, and respects the configured washout.

The contract is checked programmatically in :func:`verify_sample_index`,
which is also called inline at the end of :func:`build_sample_index` so any
violation aborts loudly before features are built.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import DATA_PROCESSED
from src.event_labels import USER_COL
from . import DATA_PROCESSED_V2, OUTPUT_TABLES_V2

DEFAULT_INPUT_LENGTHS = [1, 3, 5, 7, 14, 28]
DEFAULT_WASHOUT_DAYS = [0, 7, 14]
LABEL_STRATEGY = "questionnaire_event_episode_labeling_v2"


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
    """Create positive onset samples and stable negative samples.

    Positives end at ``E-1`` and start at ``E-input_length_days``.
    Negatives are stable non-event weeks, with all in-window days
    free of event/positive-week/washout days.
    """
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

    # Contract check (raises on violation)
    if not sample_index.empty:
        verify_sample_index(sample_index, event_days, positive_week_days, washout_days_set)

    return sample_index, counts


def verify_sample_index(
    sample_index: pd.DataFrame,
    event_days: set[tuple[int, int]],
    positive_week_days: set[tuple[int, int]],
    washout_days_set: set[tuple[int, int]],
) -> None:
    """Raise if any sample violates the v2 contract.

    * Positives:  window in [event_onset_day - L, event_onset_day - 1];
      must NOT include event_onset_day itself.
    * Negatives:  no in-window or anchor day in event_days,
      positive_week_days, or washout_days_set.
    """
    positives = sample_index[sample_index["target"].eq(1)]
    for row in positives.itertuples(index=False):
        onset = int(row.event_onset_day)
        end = int(row.window_end_day)
        start = int(row.window_start_day)
        if end >= onset:
            raise AssertionError(
                f"Positive window includes event day E={onset} "
                f"(start={start}, end={end}, user={int(getattr(row, USER_COL))})"
            )
        if end != onset - 1:
            raise AssertionError(
                f"Positive window does not end at E-1: "
                f"E={onset}, end={end}, user={int(getattr(row, USER_COL))}"
            )

    negatives = sample_index[sample_index["target"].eq(0)]
    for row in negatives.itertuples(index=False):
        user_key = int(getattr(row, USER_COL))
        for day in range(int(row.window_start_day), int(row.window_end_day) + 1):
            key = (user_key, day)
            if key in event_days:
                raise AssertionError(
                    f"Negative window contains event day for user={user_key}, day={day}"
                )
            if key in positive_week_days:
                raise AssertionError(
                    f"Negative window contains positive-week day for "
                    f"user={user_key}, day={day}"
                )
            if key in washout_days_set:
                raise AssertionError(
                    f"Negative window contains washout day for "
                    f"user={user_key}, day={day}"
                )


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

    for threshold, probable in probable_days_by_threshold.items():
        episodes = episodes_by_threshold[threshold]
        for length in input_lengths:
            for washout in washout_values:
                key = (int(threshold), int(length), int(washout))
                sample_index, counts = build_sample_index(
                    weekly_events=weekly_events,
                    probable_days=probable,
                    episodes=episodes,
                    input_length_days=length,
                    washout_days=washout,
                    threshold=int(threshold),
                )
                sample_indexes[key] = sample_index
                count_rows.append(counts)
    return sample_indexes, pd.DataFrame(count_rows)


def sample_index_filename(threshold: int, input_length_days: int, washout_days: int) -> str:
    return f"sample_index_v2_T{threshold}_L{input_length_days}_W{washout_days}.parquet"


def write_sample_indexes(
    sample_indexes: dict[tuple[int, int, int], pd.DataFrame],
    counts_summary: pd.DataFrame,
    output_dir: Path = DATA_PROCESSED_V2,
) -> dict[tuple[int, int, int], Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[tuple[int, int, int], Path] = {}
    for key, sample_index in sample_indexes.items():
        threshold, length, washout = key
        path = output_dir / sample_index_filename(threshold, length, washout)
        sample_index.to_parquet(path, index=False)
        paths[key] = path

    counts_summary.to_csv(OUTPUT_TABLES_V2 / "sample_counts_v2.csv", index=False)
    return paths


__all__ = [
    "DEFAULT_INPUT_LENGTHS",
    "DEFAULT_WASHOUT_DAYS",
    "LABEL_STRATEGY",
    "build_sample_index",
    "build_all_sample_indexes",
    "verify_sample_index",
    "sample_index_filename",
    "write_sample_indexes",
]
