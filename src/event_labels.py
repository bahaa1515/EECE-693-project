from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import AAMOS_DAILY_FILE, AAMOS_WEEKLY_FILE, OUTPUT_TABLES


USER_COL = "user_key"
DATE_COL = "date"
DIRECT_EVENT_FIELDS = ["weekly_doc", "weekly_hospital", "weekly_er"]
IMPORTANT_TRIGGER_CODES = {"2", "9", "11", "12", "13", "14"}
DEFAULT_THRESHOLDS = [2, 3, 4]


@dataclass(frozen=True)
class LabelArtifacts:
    weekly_events: pd.DataFrame
    daily_scores: pd.DataFrame
    probable_event_days: dict[int, pd.DataFrame]
    event_episodes: dict[int, pd.DataFrame]
    summary: pd.DataFrame


def load_weekly_questionnaire(path: Path = AAMOS_WEEKLY_FILE) -> pd.DataFrame:
    return pd.read_csv(path)


def load_daily_questionnaire(path: Path = AAMOS_DAILY_FILE) -> pd.DataFrame:
    return pd.read_csv(path)


def parse_weekly_offsets(value: object) -> list[int]:
    """Parse weekly offset fields such as -1, -3, or '-1,-3'."""
    if pd.isna(value):
        return []

    offsets: list[int] = []
    for part in str(value).split(","):
        part = part.strip()
        if part in {"", "0", "0.0", "nan", "NaN"}:
            continue
        try:
            offset = int(float(part))
        except ValueError:
            continue
        if -7 <= offset <= -1:
            offsets.append(offset)
    return offsets


def has_weekly_offset(value: object) -> bool:
    return bool(parse_weekly_offsets(value))


def prepare_weekly_event_labels(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """Build broad event-week labels from weekly questionnaire event fields only."""
    required = [USER_COL, DATE_COL, *DIRECT_EVENT_FIELDS, "weekly_oral"]
    missing = [col for col in required if col not in weekly_df.columns]
    if missing:
        raise KeyError(f"Missing weekly questionnaire columns: {missing}")

    weekly = weekly_df[required].copy()
    weekly = weekly.dropna(subset=[USER_COL, DATE_COL]).drop_duplicates().copy()
    weekly[USER_COL] = weekly[USER_COL].astype(int)
    weekly[DATE_COL] = weekly[DATE_COL].astype(int)

    weekly["doc_flag"] = weekly["weekly_doc"].apply(has_weekly_offset)
    weekly["hospital_flag"] = weekly["weekly_hospital"].apply(has_weekly_offset)
    weekly["er_flag"] = weekly["weekly_er"].apply(has_weekly_offset)
    weekly["oral_flag"] = pd.to_numeric(weekly["weekly_oral"], errors="coerce").eq(3)
    weekly["broad_event_week"] = weekly[
        ["doc_flag", "hospital_flag", "er_flag", "oral_flag"]
    ].any(axis=1)
    weekly["week_start_day"] = (weekly[DATE_COL] - 7).clip(lower=0)
    weekly["week_end_day"] = weekly[DATE_COL] - 1
    weekly["has_valid_covered_days"] = weekly["week_end_day"].ge(
        weekly["week_start_day"]
    )
    weekly["label_strategy"] = "questionnaire_event_episode_labeling"
    return weekly.sort_values([USER_COL, DATE_COL]).reset_index(drop=True)


def extract_direct_event_days(weekly_events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    positive = weekly_events[
        weekly_events["broad_event_week"] & weekly_events["has_valid_covered_days"]
    ]
    for row in positive.itertuples(index=False):
        weekly_date = int(getattr(row, DATE_COL))
        user_key = int(getattr(row, USER_COL))
        for field in DIRECT_EVENT_FIELDS:
            for offset in parse_weekly_offsets(getattr(row, field)):
                event_day = weekly_date + offset
                if event_day < 0:
                    continue
                rows.append(
                    {
                        USER_COL: user_key,
                        "event_day": int(event_day),
                        "weekly_date": weekly_date,
                        "event_day_source": field,
                        "event_day_confidence": "high",
                    }
                )
    if not rows:
        return pd.DataFrame(
            columns=[
                USER_COL,
                "event_day",
                "weekly_date",
                "event_day_source",
                "event_day_confidence",
            ]
        )
    return pd.DataFrame(rows).drop_duplicates().sort_values([USER_COL, "event_day"])


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype("string").fillna("").str.lower().eq("true")


def trigger_present(value: object) -> bool:
    if pd.isna(value):
        return False
    codes = {part.strip() for part in str(value).split(",") if part.strip()}
    return bool(codes & IMPORTANT_TRIGGER_CODES)


def compute_daily_questionnaire_scores(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Compute questionnaire-only worsening scores for each daily row."""
    required = [
        USER_COL,
        DATE_COL,
        "daily_night_symp",
        "daily_day_symp",
        "daily_limit_activity",
        "daily_prev_inhaler",
        "daily_relief_inhaler",
        "daily_triggers",
    ]
    missing = [col for col in required if col not in daily_df.columns]
    if missing:
        raise KeyError(f"Missing daily questionnaire columns: {missing}")

    daily = daily_df[required].copy()
    daily = daily.dropna(subset=[USER_COL, DATE_COL]).drop_duplicates().copy()
    daily[USER_COL] = daily[USER_COL].astype(int)
    daily[DATE_COL] = daily[DATE_COL].astype(int)

    relief = pd.to_numeric(daily["daily_relief_inhaler"], errors="coerce")
    daily["relief_inhaler_points"] = np.select(
        [relief.eq(0), relief.isin([1, 3]), relief.eq(5), relief.isin([9, 12])],
        [0, 1, 2, 3],
        default=0,
    ).astype(int)
    daily["night_symptom_points"] = _bool_series(daily["daily_night_symp"]).astype(int)
    daily["day_symptom_points"] = _bool_series(daily["daily_day_symp"]).astype(int)
    daily["activity_limit_points"] = _bool_series(
        daily["daily_limit_activity"]
    ).astype(int)
    daily["preventer_more_than_usual_points"] = pd.to_numeric(
        daily["daily_prev_inhaler"], errors="coerce"
    ).eq(4).astype(int)
    daily["trigger_points"] = daily["daily_triggers"].apply(trigger_present).astype(int)
    score_cols = [
        "relief_inhaler_points",
        "night_symptom_points",
        "day_symptom_points",
        "activity_limit_points",
        "preventer_more_than_usual_points",
        "trigger_points",
    ]
    daily["daily_questionnaire_worsening_score"] = daily[score_cols].sum(axis=1)
    return daily.sort_values([USER_COL, DATE_COL]).reset_index(drop=True)


def daily_scores_inside_positive_weeks(
    daily_scores: pd.DataFrame, weekly_events: pd.DataFrame
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    positive = weekly_events[
        weekly_events["broad_event_week"] & weekly_events["has_valid_covered_days"]
    ]
    for week in positive.itertuples(index=False):
        user_key = int(getattr(week, USER_COL))
        start = int(week.week_start_day)
        end = int(week.week_end_day)
        mask = (
            daily_scores[USER_COL].eq(user_key)
            & daily_scores[DATE_COL].between(start, end)
        )
        part = daily_scores.loc[mask].copy()
        if part.empty:
            continue
        part["weekly_date"] = int(getattr(week, DATE_COL))
        part["week_start_day"] = start
        part["week_end_day"] = end
        rows.append(part)
    if not rows:
        return pd.DataFrame(columns=[*daily_scores.columns, "weekly_date"])
    return pd.concat(rows, ignore_index=True).drop_duplicates(
        [USER_COL, DATE_COL, "weekly_date"]
    )


def build_probable_event_days(
    weekly_events: pd.DataFrame,
    positive_week_daily_scores: pd.DataFrame,
    threshold: int = 3,
) -> pd.DataFrame:
    direct = extract_direct_event_days(weekly_events)
    direct_days = direct.rename(columns={"event_day": DATE_COL})

    score_days = positive_week_daily_scores[
        positive_week_daily_scores["daily_questionnaire_worsening_score"].ge(threshold)
    ].copy()
    if not score_days.empty:
        score_days = score_days[
            [
                USER_COL,
                DATE_COL,
                "weekly_date",
                "daily_questionnaire_worsening_score",
            ]
        ].copy()
        score_days["event_day_source"] = "daily_score"
        score_days["event_day_confidence"] = "probable"
    else:
        score_days = pd.DataFrame(
            columns=[
                USER_COL,
                DATE_COL,
                "weekly_date",
                "daily_questionnaire_worsening_score",
                "event_day_source",
                "event_day_confidence",
            ]
        )

    if not direct_days.empty:
        direct_days = direct_days[
            [USER_COL, DATE_COL, "weekly_date", "event_day_source", "event_day_confidence"]
        ].copy()
        direct_days["daily_questionnaire_worsening_score"] = np.nan

    combined = pd.concat([direct_days, score_days], ignore_index=True)
    if combined.empty:
        return pd.DataFrame(
            columns=[
                USER_COL,
                "event_day",
                "threshold",
                "probable_event_day",
                "event_day_source",
                "event_day_confidence",
                "daily_questionnaire_worsening_score",
                "weekly_dates",
            ]
        )

    rows = []
    for (user_key, event_day), group in combined.groupby([USER_COL, DATE_COL]):
        sources = sorted(set(group["event_day_source"].dropna().astype(str)))
        confidence = "high" if any(src in DIRECT_EVENT_FIELDS for src in sources) else "probable"
        scores = pd.to_numeric(
            group["daily_questionnaire_worsening_score"], errors="coerce"
        ).dropna()
        rows.append(
            {
                USER_COL: int(user_key),
                "event_day": int(event_day),
                "threshold": int(threshold),
                "probable_event_day": 1,
                "event_day_source": "+".join(sources),
                "event_day_confidence": confidence,
                "daily_questionnaire_worsening_score": (
                    float(scores.max()) if len(scores) else np.nan
                ),
                "weekly_dates": ",".join(
                    str(int(v)) for v in sorted(group["weekly_date"].dropna().unique())
                ),
            }
        )
    return pd.DataFrame(rows).sort_values([USER_COL, "event_day"]).reset_index(drop=True)


def build_event_episodes(probable_days: pd.DataFrame, threshold: int = 3) -> pd.DataFrame:
    if probable_days.empty:
        return pd.DataFrame(
            columns=[
                USER_COL,
                "episode_id",
                "threshold",
                "event_onset_day",
                "event_end_day",
                "event_duration_days",
                "event_day_count",
                "has_high_confidence_day",
                "event_day_sources",
            ]
        )

    rows: list[dict[str, object]] = []
    for user_key, group in probable_days.sort_values([USER_COL, "event_day"]).groupby(
        USER_COL
    ):
        episode_id = 0
        current_days: list[int] = []
        current_sources: list[str] = []
        current_high = False

        def flush_episode() -> None:
            nonlocal episode_id, current_days, current_sources, current_high
            if not current_days:
                return
            episode_id += 1
            rows.append(
                {
                    USER_COL: int(user_key),
                    "episode_id": episode_id,
                    "threshold": int(threshold),
                    "event_onset_day": int(current_days[0]),
                    "event_end_day": int(current_days[-1]),
                    "event_duration_days": int(current_days[-1] - current_days[0] + 1),
                    "event_day_count": len(current_days),
                    "has_high_confidence_day": bool(current_high),
                    "event_day_sources": "+".join(sorted(set(current_sources))),
                }
            )
            current_days = []
            current_sources = []
            current_high = False

        previous_day: int | None = None
        for day_row in group.itertuples(index=False):
            day = int(day_row.event_day)
            if previous_day is not None and day != previous_day + 1:
                flush_episode()
            current_days.append(day)
            current_sources.extend(str(day_row.event_day_source).split("+"))
            current_high = current_high or day_row.event_day_confidence == "high"
            previous_day = day
        flush_episode()

    return pd.DataFrame(rows).sort_values([USER_COL, "event_onset_day"]).reset_index(
        drop=True
    )


def summarize_label_artifacts(
    weekly_events: pd.DataFrame,
    positive_week_daily_scores: pd.DataFrame,
    probable_days_by_threshold: dict[int, pd.DataFrame],
    episodes_by_threshold: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    rows = [
        {"metric": "weekly_rows", "threshold": np.nan, "value": len(weekly_events)},
        {
            "metric": "broad_positive_weeks",
            "threshold": np.nan,
            "value": int(weekly_events["broad_event_week"].sum()),
        },
        {
            "metric": "broad_negative_weeks",
            "threshold": np.nan,
            "value": int((~weekly_events["broad_event_week"]).sum()),
        },
        {
            "metric": "positive_week_daily_score_rows",
            "threshold": np.nan,
            "value": len(positive_week_daily_scores),
        },
    ]
    direct = extract_direct_event_days(weekly_events)
    rows.append(
        {
            "metric": "direct_offset_event_user_days",
            "threshold": np.nan,
            "value": direct[[USER_COL, "event_day"]].drop_duplicates().shape[0],
        }
    )
    for threshold, probable_days in probable_days_by_threshold.items():
        episodes = episodes_by_threshold[threshold]
        rows.extend(
            [
                {
                    "metric": "probable_event_user_days",
                    "threshold": threshold,
                    "value": probable_days[[USER_COL, "event_day"]]
                    .drop_duplicates()
                    .shape[0],
                },
                {
                    "metric": "event_episodes",
                    "threshold": threshold,
                    "value": len(episodes),
                },
                {
                    "metric": "episode_users",
                    "threshold": threshold,
                    "value": episodes[USER_COL].nunique() if len(episodes) else 0,
                },
            ]
        )
    return pd.DataFrame(rows)


def build_label_artifacts(
    weekly_path: Path = AAMOS_WEEKLY_FILE,
    daily_path: Path = AAMOS_DAILY_FILE,
    thresholds: list[int] | None = None,
) -> LabelArtifacts:
    thresholds = thresholds or DEFAULT_THRESHOLDS
    weekly_events = prepare_weekly_event_labels(load_weekly_questionnaire(weekly_path))
    daily_scores = compute_daily_questionnaire_scores(load_daily_questionnaire(daily_path))
    positive_week_daily_scores = daily_scores_inside_positive_weeks(
        daily_scores, weekly_events
    )
    probable_days = {
        threshold: build_probable_event_days(
            weekly_events, positive_week_daily_scores, threshold
        )
        for threshold in thresholds
    }
    episodes = {
        threshold: build_event_episodes(probable_days[threshold], threshold)
        for threshold in thresholds
    }
    summary = summarize_label_artifacts(
        weekly_events, positive_week_daily_scores, probable_days, episodes
    )
    return LabelArtifacts(
        weekly_events=weekly_events,
        daily_scores=positive_week_daily_scores,
        probable_event_days=probable_days,
        event_episodes=episodes,
        summary=summary,
    )


def write_label_artifacts(
    artifacts: LabelArtifacts,
    output_dir: Path = OUTPUT_TABLES,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts.weekly_events.to_csv(output_dir / "weekly_event_labels.csv", index=False)
    artifacts.daily_scores.to_csv(
        output_dir / "daily_questionnaire_worsening_scores.csv", index=False
    )
    artifacts.summary.to_csv(output_dir / "event_label_summary.csv", index=False)
    for threshold, probable_days in artifacts.probable_event_days.items():
        probable_days.to_csv(
            output_dir / f"probable_event_days_threshold_{threshold}.csv",
            index=False,
        )
    for threshold, episodes in artifacts.event_episodes.items():
        episodes.to_csv(
            output_dir / f"event_episodes_threshold_{threshold}.csv",
            index=False,
        )


def run_event_labeling(
    weekly_path: Path = AAMOS_WEEKLY_FILE,
    daily_path: Path = AAMOS_DAILY_FILE,
    output_dir: Path = OUTPUT_TABLES,
    thresholds: list[int] | None = None,
) -> LabelArtifacts:
    artifacts = build_label_artifacts(weekly_path, daily_path, thresholds)
    write_label_artifacts(artifacts, output_dir)
    return artifacts


if __name__ == "__main__":
    result = run_event_labeling()
    print(result.summary.to_string(index=False))
