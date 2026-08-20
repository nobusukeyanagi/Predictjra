#!/usr/bin/env python3
"""Rebuild Predictjra historical predictions/results for an arbitrary date range.

Goals
-----
* Reconstruct predictions from archived PRE-RACE snapshots.
* Never use current-race odds, actual popularity, horse bodyweight/change, or race result
  as inputs to the performance/selection score. Previous-race popularity is allowed only
  in the separate market-popularity estimator.
* Use actual popularity only as a teacher label for the estimated-popularity model.
  Each race is estimated with a model trained while excluding that race itself.
* Re-fetch archived final results and trifecta payouts and recalculate hit/miss, return,
  stake, and daily totals.
* Refuse to write partial data: every selected race and all required validation checks must pass.

Historical snapshot source:
  https://github.com/sugaimo15/keibayosoku
  ref: claude/horse-racing-predictor-ak6crm

The archived prediction CSVs are used only to prove that a clean pre-race snapshot
existed and to verify the runner set. Archived model scores/ranks are never reused.
All derived indices and selections are recalculated through scripts/prediction_logic.py.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from prediction_logic import (
    FEATURE_COLS,
    MODEL_VERSION,
    SELECTION_RULE_TEXT,
    build_index_core,
    build_market_profile,
    market_context_adjustment,
    parse_class_level,
    prediction_target_count,
    rank_strengths,
    select_prediction,
)

JST = ZoneInfo("Asia/Tokyo")

TRACKS = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
}
TRACK_ORDER = {name: i for i, name in enumerate(TRACKS.values(), start=1)}

# Kept only for popularity-model case-study diagnostics.  No race-specific
# prediction score override is allowed; all races use prediction_logic.py.
SAPPORO11_ID = "202601010811"

REBUILD_VERSION = "predictjra-history-generic-v7-unified-logic"
SOURCE_REPO = "sugaimo15/keibayosoku"
SOURCE_REF = "claude/horse-racing-predictor-ak6crm"

# Known race-condition metadata gaps in the archived source.
# Values here come from authoritative pre-race race-program information.
RACE_METADATA_OVERRIDES = {
    "202604020709": {"surface": "障害", "distance_m": 3250.0},  # 新潟ジャンプS
}

DIAGNOSTIC_PATH_DEFAULT = Path("data/rebuild_diagnostics.json")


def clean_str(v) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip()


def numeric_or_nan(v) -> float:
    return pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]


def parse_age(sex_age) -> float:
    m = re.search(r"(\d+)", clean_str(sex_age))
    return float(m.group(1)) if m else 5.0


def normalize_entity_id(value) -> str:
    s = clean_str(value)
    digits = re.sub(r"\D", "", s)
    return digits.zfill(5) if digits else ""


def clipped01(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.5
    return float(np.clip(value, 0.0, 1.0))


def shrunk_market_mean(values: pd.Series | list[float], prior: float = 0.5, prior_weight: float = 12.0) -> float:
    vals = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().astype(float)
    if vals.empty:
        return float(prior)
    return clipped01((float(vals.sum()) + prior * prior_weight) / (len(vals) + prior_weight))


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def source_commit(source_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return ""


def parse_time_seconds(value) -> float:
    s = clean_str(value)
    if not s:
        return math.nan
    try:
        if ":" in s:
            mins, sec = s.split(":", 1)
            return float(mins) * 60.0 + float(sec)
        return float(s)
    except (TypeError, ValueError):
        return math.nan


def parse_distance_m(value) -> float:
    """Parse race distance from 1700 / 1700m / 1,700 / 芝1700m style values."""
    s = clean_str(value).replace(",", "")
    if not s:
        return math.nan
    direct = pd.to_numeric(pd.Series([s]), errors="coerce").iloc[0]
    if pd.notna(direct) and float(direct) > 0:
        return float(direct)
    m = re.search(r"(\d{3,4})", s)
    if not m:
        return math.nan
    value = float(m.group(1))
    return value if value > 0 else math.nan


def first_race_distance(*frames: pd.DataFrame) -> float:
    """Return the first valid distance from pre-race metadata, then result metadata.

    Result data is used only as a fallback for the already-known race condition
    (distance), never for performance scoring or horse evaluation.
    """
    for df in frames:
        if df is None or df.empty or "distance_m" not in df.columns:
            continue
        for raw in df["distance_m"].tolist():
            distance = parse_distance_m(raw)
            if math.isfinite(distance) and distance > 0:
                return float(distance)
    return math.nan


def first_race_surface(*frames: pd.DataFrame) -> str:
    for df in frames:
        if df is None or df.empty or "surface" not in df.columns:
            continue
        for raw in df["surface"].tolist():
            value = clean_str(raw)
            if value:
                return value
    return ""


def passing_positions(value) -> list[int]:
    s = clean_str(value)
    if not s:
        return []
    return [int(x) for x in re.findall(r"\d+", s)]


def load_target_history(source_root: Path, target_horse_ids: set[str]) -> pd.DataFrame:
    """Load full race files containing at least one target horse.

    The scan reads race-result archives only. Target-day rows may be present in the
    loaded frame, but every index calculation later applies date < target_date, so
    the race being predicted and all same-day results are excluded.
    """
    ids = sorted({clean_str(x) for x in target_horse_ids if clean_str(x)})
    if not ids:
        return pd.DataFrame()

    pattern = re.compile(r",(?:" + "|".join(re.escape(x) for x in ids) + r"),")
    frames = []
    result_root = source_root / "data" / "race_results"

    for path in sorted(result_root.glob("*/*.csv")):
        # The oldest archived seasons are irrelevant to most last-5-run lookups but
        # retaining every available season also handles long layoffs correctly.
        try:
            text = path.read_text(encoding="utf-8-sig", errors="ignore")
        except Exception:
            continue
        if not pattern.search(text):
            continue
        try:
            df = read_csv(path)
        except Exception:
            continue
        if "horse_id" not in df.columns:
            continue
        mask = df["horse_id"].apply(clean_str).isin(ids)
        if mask.any():
            # Keep the full race, not only target horses, because relative time and
            # last-3F indices need the other runners in that historical race.
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    hist = pd.concat(frames, ignore_index=True)
    hist["race_id"] = hist["race_id"].astype(str)
    hist["horse_id"] = hist["horse_id"].apply(clean_str)
    hist["_date"] = pd.to_datetime(hist.get("date"), errors="coerce")
    hist["_finish"] = pd.to_numeric(hist.get("finish_position"), errors="coerce")
    hist["_time_sec"] = hist.get("time", pd.Series(index=hist.index, dtype=object)).apply(parse_time_seconds)

    numeric_finish = hist["_finish"].notna()
    hist["_field_size"] = (
        numeric_finish.astype(int)
        .groupby(hist["race_id"])
        .transform("sum")
        .replace(0, np.nan)
    )
    hist["_winner_time"] = hist["_time_sec"].groupby(hist["race_id"]).transform("min")

    # Historical market-memory fields. These are all known before a future race:
    # previous-race popularity, assigned weight, jockey and trainer.
    hist["_popularity"] = pd.to_numeric(
        hist.get("popularity", pd.Series(index=hist.index, dtype=float)),
        errors="coerce",
    )
    pop_denom = (hist["_field_size"] - 1).replace(0, np.nan)
    hist["_market_strength"] = (
        1 - (hist["_popularity"] - 1) / pop_denom
    ).clip(0, 1)
    hist["_weight_carried_num"] = pd.to_numeric(
        hist.get("weight_carried", pd.Series(index=hist.index, dtype=float)),
        errors="coerce",
    )
    hist["_jockey_id_norm"] = hist.get(
        "jockey_id", pd.Series(index=hist.index, dtype=object)
    ).apply(normalize_entity_id)
    hist["_trainer_id_norm"] = hist.get(
        "trainer_id", pd.Series(index=hist.index, dtype=object)
    ).apply(normalize_entity_id)
    hist["_jockey_name_norm"] = hist.get(
        "jockey", pd.Series(index=hist.index, dtype=object)
    ).apply(clean_str)
    hist["_trainer_name_norm"] = hist.get(
        "trainer", pd.Series(index=hist.index, dtype=object)
    ).apply(clean_str)


    return hist


def normalize_historical_run(row: pd.Series) -> dict:
    """Convert one archived result row to the same normalized run shape as live parsing."""
    race_id = clean_str(row.get("race_id"))
    finish_raw = row.get("_finish")
    field_raw = row.get("_field_size")
    pop_raw = row.get("_popularity")
    carried_raw = row.get("_weight_carried_num")
    time_raw = row.get("_time_sec")
    winner_raw = row.get("_winner_time")

    finish = int(finish_raw) if pd.notna(finish_raw) else None
    field = int(field_raw) if pd.notna(field_raw) else None
    popularity = int(pop_raw) if pd.notna(pop_raw) else None
    carried = float(carried_raw) if pd.notna(carried_raw) else math.nan
    if pd.notna(time_raw) and pd.notna(winner_raw):
        margin = max(0.0, float(time_raw) - float(winner_raw))
    else:
        margin = math.nan

    surface_raw = clean_str(row.get("surface"))
    if "障" in surface_raw:
        surface = "障害"
    elif "芝" in surface_raw:
        surface = "芝"
    elif "ダ" in surface_raw:
        surface = "ダート"
    else:
        surface = surface_raw

    distance = parse_distance_m(row.get("distance_m"))
    race_name = clean_str(row.get("race_name")) or clean_str(row.get("race_title"))

    return {
        "date": row.get("_date").strftime("%Y-%m-%d") if pd.notna(row.get("_date")) else "",
        "venue": TRACKS.get(race_id[4:6], race_id[4:6]) if len(race_id) >= 6 else "",
        "finish": finish,
        "field": field,
        "popularity": popularity,
        "classLevel": parse_class_level(race_name),
        "carriedWeight": carried,
        "surface": surface,
        "distance": distance,
        "positions": passing_positions(row.get("passing_order")),
        "margin": margin,
    }


def build_index_detail(
    race_id: str,
    date_s: str,
    card: pd.DataFrame,
    pred: pd.DataFrame,
    result: pd.DataFrame,
    history: pd.DataFrame,
) -> tuple[dict, dict[int, float], dict[int, int], dict[int, list[dict]]]:
    """Reconstruct indices with the exact same shared core used for live prediction."""
    target_date = pd.Timestamp(date_s)
    card2 = card.copy()
    card2["horse_number"] = pd.to_numeric(card2["horse_number"], errors="raise").astype(int)
    card2["horse_id"] = card2["horse_id"].apply(clean_str)

    quality_warnings: list[str] = []
    race_distance = first_race_distance(card2, pred, result)
    race_surface = first_race_surface(card2, pred, result)

    override = RACE_METADATA_OVERRIDES.get(race_id, {})
    if (not math.isfinite(race_distance) or race_distance <= 0) and override.get("distance_m"):
        race_distance = float(override["distance_m"])
        quality_warnings.append("distance_m supplemented from authoritative race-program override")
    if not race_surface and override.get("surface"):
        race_surface = clean_str(override["surface"])
        quality_warnings.append("surface supplemented from authoritative race-program override")

    if not math.isfinite(race_distance) or race_distance <= 0:
        race_distance = math.nan
        quality_warnings.append("distance_m unavailable; course index used neutral fallback")
    if not race_surface:
        quality_warnings.append("surface unavailable; course index used neutral fallback")

    venue = TRACKS.get(race_id[4:6], race_id[4:6])
    entries: list[dict] = []
    for _, row in card2.iterrows():
        no = int(row["horse_number"])
        horse_id = clean_str(row.get("horse_id"))
        h = history[
            (history["horse_id"] == horse_id)
            & history["_date"].notna()
            & (history["_date"] < target_date)
            & history["_finish"].notna()
        ].sort_values("_date")
        recent5 = h.tail(5).iloc[::-1]
        histories = [normalize_historical_run(rr) for _, rr in recent5.iterrows()]
        entries.append({
            "no": no,
            "name": clean_str(row.get("horse_name")),
            "histories": histories,
            "age": parse_age(row.get("sex_age")),
        })

    core = build_index_core(
        entries,
        venue=venue,
        surface=race_surface,
        distance_m=float(race_distance) if math.isfinite(race_distance) else None,
    )
    horses = core["horses"]
    internal_totals = core["totals"]

    race_no = int(race_id[-2:])
    race_name = clean_str(card2.get("race_name", pd.Series([""])).iloc[0])
    title = f"{venue}{race_no}R" + (f" {race_name}" if race_name else "")
    detail = {
        "title": title,
        "horseCount": len(horses),
        "paceRegime": core["paceRegime"],
        "raceConditions": {
            "surface": race_surface or None,
            "distanceM": int(race_distance) if math.isfinite(race_distance) else None,
        },
        "qualityWarnings": quality_warnings,
        "horses": horses,
    }
    total_display_map = {h["no"]: int(h["total"]) for h in horses}
    return detail, internal_totals, total_display_map, core["runsByNo"]


@dataclass
class RaceModel:
    race_id: str
    date: str
    card: pd.DataFrame
    pred: pd.DataFrame
    result: pd.DataFrame
    payout: pd.DataFrame
    total_internal: dict[int, float]
    total_display: dict[int, int]
    popularity_features: pd.DataFrame
    actual_popularity: dict[int, int]
    index_detail: dict



def prediction_snapshot_issues(pred: pd.DataFrame) -> list[str]:
    """Return reasons why a prediction CSV is not a clean pre-race snapshot."""
    issues: list[str] = []

    if "win_odds" in pred.columns:
        odds = pd.to_numeric(pred["win_odds"], errors="coerce")
        if (odds > 0).any():
            issues.append("current odds populated")

    if "horse_weight" in pred.columns:
        nonblank = pred["horse_weight"].apply(clean_str)
        if nonblank.ne("").any():
            issues.append("current horse bodyweight populated")

    # Historical prediction files from some dates were saved after final popularity
    # had been populated. Those snapshots are not valid backtest inputs.
    if "popularity" in pred.columns:
        popularity = pd.to_numeric(pred["popularity"], errors="coerce")
        if popularity.notna().any():
            issues.append("current actual popularity populated")

    return issues


def validate_prediction_snapshot(pred: pd.DataFrame, race_id: str) -> None:
    """Fail if the prediction snapshot contains current-race market/bodyweight data."""
    issues = prediction_snapshot_issues(pred)
    if issues:
        raise ValueError(
            f"{race_id}: contaminated prediction snapshot ({', '.join(issues)})"
        )


def sanitize_card(card: pd.DataFrame) -> pd.DataFrame:
    """Remove post-snapshot fields that are never allowed as prediction features.

    race_cards in the archive may have been refreshed after a race, even when the
    prediction CSV itself is a clean pre-race snapshot. The model only needs stable
    program fields (horse id/name, assigned weight, jockey/trainer, surface/distance).
    """
    return card.drop(
        columns=["win_odds", "horse_weight", "popularity"],
        errors="ignore",
    ).copy()


def build_race_model(source_root: Path, date_s: str, pred_path: Path, history: pd.DataFrame) -> RaceModel:
    race_id = pred_path.stem
    card_path = source_root / "data" / "race_cards" / date_s.replace("-", "") / f"{race_id}.csv"
    result_path = source_root / "data" / "race_results" / "2026" / f"{race_id}.csv"
    payout_path = source_root / "data" / "race_payouts" / f"{race_id}.csv"

    card = sanitize_card(read_csv(card_path))
    pred = read_csv(pred_path)
    result = read_csv(result_path)
    payout = read_csv(payout_path)

    validate_prediction_snapshot(pred, race_id)

    for df_name, df in (("card", card), ("pred", pred), ("result", result)):
        if df.empty:
            raise ValueError(f"{race_id}: empty {df_name}")
        if "race_id" not in df.columns or set(df["race_id"].astype(str)) != {race_id}:
            raise ValueError(f"{race_id}: invalid race_id in {df_name}")

    card_nums = set(pd.to_numeric(card["horse_number"], errors="raise").astype(int))
    pred_nums = set(pd.to_numeric(pred["horse_number"], errors="raise").astype(int))
    result_nums = set(pd.to_numeric(result["horse_number"], errors="raise").astype(int))
    if card_nums != pred_nums:
        raise ValueError(f"{race_id}: card/pred horse mismatch {sorted(card_nums ^ pred_nums)}")
    if not result_nums.issubset(card_nums):
        raise ValueError(f"{race_id}: result contains horse not in card {sorted(result_nums-card_nums)}")
    if len(card_nums) < 5:
        raise ValueError(f"{race_id}: only {len(card_nums)} horses")

    # The archived prediction file is now used only as proof of a clean pre-race
    # snapshot and to verify the runner set.  Old model scores / predicted ranks are
    # deliberately ignored: all derived values come from prediction_logic.py.
    if "horse_number" not in pred.columns:
        raise ValueError(f"{race_id}: prediction missing horse_number")
    p = pred.copy()
    p["horse_number"] = pd.to_numeric(p["horse_number"], errors="raise").astype(int)

    index_detail, total_internal, total_display, runs_by_no = build_index_detail(
        race_id, date_s, card, p, result, history
    )

    detail_horses = index_detail.get("horses", [])
    total_rank_strength, recent_rank_strength = rank_strengths(
        detail_horses, total_internal
    )
    recent_map = {
        int(h["no"]): float(h.get("recentIndex", 72)) for h in detail_horses
    }

    target_date = pd.Timestamp(date_s)
    eligible_hist = history[
        history["_date"].notna()
        & (history["_date"] < target_date)
        & history["_market_strength"].notna()
    ]
    card2 = card.copy()
    card2["horse_number"] = pd.to_numeric(
        card2["horse_number"], errors="raise"
    ).astype(int)

    race_name = clean_str(card2.get("race_name", pd.Series([""])).iloc[0])
    current_class_level = parse_class_level(race_name)
    market_rows = []
    for _, row in card2.iterrows():
        no = int(row["horse_number"])

        jockey_id = normalize_entity_id(row.get("jockey_id"))
        trainer_id = normalize_entity_id(row.get("trainer_id"))
        jockey_name = clean_str(row.get("jockey"))
        trainer_name = clean_str(row.get("trainer"))

        if jockey_id:
            jockey_values = eligible_hist.loc[
                eligible_hist["_jockey_id_norm"] == jockey_id, "_market_strength"
            ]
        elif jockey_name:
            jockey_values = eligible_hist.loc[
                eligible_hist["_jockey_name_norm"] == jockey_name, "_market_strength"
            ]
        else:
            jockey_values = pd.Series(dtype=float)

        if trainer_id:
            trainer_values = eligible_hist.loc[
                eligible_hist["_trainer_id_norm"] == trainer_id, "_market_strength"
            ]
        elif trainer_name:
            trainer_values = eligible_hist.loc[
                eligible_hist["_trainer_name_norm"] == trainer_name, "_market_strength"
            ]
        else:
            trainer_values = pd.Series(dtype=float)

        factors, context = build_market_profile(
            runs_by_no.get(no, []),
            total_rank_strength=total_rank_strength.get(no, 0.5),
            recent_rank_strength=recent_rank_strength.get(no, 0.5),
            current_carried_weight=numeric_or_nan(row.get("weight_carried")),
            jockey_market_strength=shrunk_market_mean(
                jockey_values, prior=0.5, prior_weight=20.0
            ),
            trainer_market_strength=shrunk_market_mean(
                trainer_values, prior=0.5, prior_weight=30.0
            ),
            age=parse_age(row.get("sex_age")),
            current_class_level=current_class_level,
        )
        market_rows.append({
            "horse_number": no,
            **factors,
            "_current_class_level": int(context["classLevel"]),
            "_last_class_level": int(context["lastClassLevel"]),
            "_max_recent_class_level": int(context["maxRecentClassLevel"]),
            "_assigned_weight_delta": float(context["assignedWeightDelta"]),
            "_recent_index": float(recent_map.get(no, 72.0)),
            "_total_display": int(total_display[no]),
        })

    features = pd.DataFrame(market_rows)
    features["race_id"] = race_id

    actual_popularity = {}
    if "popularity" not in result.columns:
        raise ValueError(f"{race_id}: result popularity missing")
    for _, row in result.iterrows():
        no = int(row["horse_number"])
        pop = pd.to_numeric(pd.Series([row["popularity"]]), errors="coerce").iloc[0]
        if pd.notna(pop):
            actual_popularity[no] = int(pop)

    # 取消・除外などの非出走馬は結果CSVに馬番自体は残る一方、
    # 確定人気は付かない。これは正常なので教師データからだけ除外する。
    # 実際に走った馬について人気が欠けている場合だけエラーにする。
    finish_num = pd.to_numeric(result["finish_position"], errors="coerce")
    starter_nums = set(
        pd.to_numeric(
            result.loc[finish_num.notna(), "horse_number"], errors="raise"
        ).astype(int)
    )
    missing_starter_pop = starter_nums - set(actual_popularity)
    if missing_starter_pop:
        raise ValueError(
            f"{race_id}: starters missing actual popularity: "
            f"{sorted(missing_starter_pop)}"
        )

    return RaceModel(
        race_id=race_id,
        date=date_s,
        card=card,
        pred=p,
        result=result,
        payout=payout,
        total_internal=total_internal,
        total_display=total_display,
        popularity_features=features,
        actual_popularity=actual_popularity,
        index_detail=index_detail,
    )


def target_market_strength(pop: int, field_size: int) -> float:
    return 1.0 - (float(pop) - 1.0) / max(field_size - 1, 1)


def fit_ridge(rows: pd.DataFrame, ridge: float = 3.0) -> np.ndarray:
    X = rows[FEATURE_COLS].astype(float).to_numpy()
    X = np.column_stack([np.ones(len(X)), X])
    y = rows["market_strength"].astype(float).to_numpy()
    reg = np.eye(X.shape[1]) * ridge
    reg[0, 0] = 0.0  # do not regularize intercept
    return np.linalg.solve(X.T @ X + reg, X.T @ y)


def predict_strength(features: pd.DataFrame, beta: np.ndarray) -> np.ndarray:
    X = features[FEATURE_COLS].astype(float).to_numpy()
    X = np.column_stack([np.ones(len(X)), X])
    return X @ beta


def estimate_popularities(races: dict[str, RaceModel]) -> tuple[dict[str, dict[int, int]], dict]:
    # Build teacher table from final popularity. It is never an input feature.
    teacher_parts = []
    for rid, race in races.items():
        f = race.popularity_features.copy()
        actual_starter_count = len(race.actual_popularity)
        if actual_starter_count < 2:
            raise ValueError(f"{rid}: too few actual starters for popularity calibration")

        f["field_size"] = actual_starter_count
        f["actual_popularity"] = f["horse_number"].map(race.actual_popularity)
        # 取消・除外などはレース前予想候補としては保持するが、
        # 最終人気の教師ラベルを持たないため学習行から除外する。
        f = f[f["actual_popularity"].notna()].copy()
        f["market_strength"] = [
            target_market_strength(int(p), actual_starter_count)
            for p in f["actual_popularity"]
        ]
        teacher_parts.append(f)
    teacher = pd.concat(teacher_parts, ignore_index=True)

    estimates: dict[str, dict[int, int]] = {}
    abs_errors = []
    top3_overlaps = []
    top1_hits = []
    large_errors = []
    sapporo_case = {}

    # Leave-one-race-out: a race's own final popularity never trains its estimator.
    for rid, race in races.items():
        train = teacher[teacher["race_id"] != rid]
        beta = fit_ridge(train)
        feats = race.popularity_features.copy()
        feats["_pred_market"] = predict_strength(feats, beta)
        # The live engine adds the same class/handicap context adjustment after
        # the calibrated linear market score.  Apply it here as well.
        adjustments = []
        for _, row in feats.iterrows():
            factors = {name: float(row[name]) for name in FEATURE_COLS}
            context = {
                "classLevel": int(row["_current_class_level"]),
                "lastClassLevel": int(row["_last_class_level"]),
                "maxRecentClassLevel": int(row["_max_recent_class_level"]),
                "assignedWeightDelta": float(row["_assigned_weight_delta"]),
            }
            adjustments.append(market_context_adjustment(factors, context))
        feats["_pred_market"] = feats["_pred_market"] + np.asarray(adjustments)

        # Identical tie break to prediction_logic.expected_popularity_from_scores.
        feats = feats.sort_values(
            ["_pred_market", "_recent_index", "_total_display", "horse_number"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
        feats["estimated_popularity"] = np.arange(1, len(feats) + 1)
        est = dict(zip(
            feats["horse_number"].astype(int),
            feats["estimated_popularity"].astype(int),
        ))
        estimates[rid] = est

        actual = race.actual_popularity

        # 検証時だけ非出走馬を除き、実際に走った馬の中で想定人気を振り直す。
        # これにより、除外馬が想定順位の途中にいた場合でも実人気1〜Nと
        # 公平に比較できる。ページに保存する est 自体はレース前全頭順位のまま。
        starter_order = [
            int(no) for no in feats["horse_number"].tolist()
            if int(no) in actual
        ]
        est_starter = {no: i + 1 for i, no in enumerate(starter_order)}

        for no, ep in est_starter.items():
            abs_errors.append(abs(ep - actual[no]))

        est_top3 = {no for no, p in est_starter.items() if p <= 3}
        act_top3 = {no for no, p in actual.items() if p <= 3}
        top_n = min(3, len(actual))
        top3_overlaps.append(
            len(est_top3 & act_top3) / top_n if top_n else 0.0
        )
        actual_top1 = next((no for no, p in actual.items() if p == 1), None)
        estimated_top1 = next((no for no, p in est_starter.items() if p == 1), None)
        top1_hits.append(float(actual_top1 == estimated_top1))
        race_errors = [
            abs(est_starter[no] - actual[no])
            for no in est_starter
        ]
        large_errors.extend(float(err >= 5) for err in race_errors)

        if rid == SAPPORO11_ID:
            sapporo_case = {
                "zendanHayabusaHorseNumber": 12,
                "zendanHayabusaEstimatedPopularity": int(est_starter.get(12, -1)),
                "zendanHayabusaActualPopularity": int(actual.get(12, -1)),
                "estimatedTop3": sorted(
                    [int(no) for no, p in est_starter.items() if p <= 3],
                    key=lambda no: est_starter[no],
                ),
                "actualTop3": sorted(
                    [int(no) for no, p in actual.items() if p <= 3],
                    key=lambda no: actual[no],
                ),
            }

    # Final coefficients for future generalization (trained on all selected races).
    final_beta = fit_ridge(teacher)
    metrics = {
        "method": "leave-one-race-out for historical estimates",
        "ridge": 3.0,
        "features": FEATURE_COLS,
        "meanAbsolutePopularityRankError": round(float(np.mean(abs_errors)), 4),
        "medianAbsolutePopularityRankError": round(float(np.median(abs_errors)), 4),
        "maxAbsolutePopularityRankError": int(np.max(abs_errors)) if abs_errors else 0,
        "largeError5PlusRate": round(float(np.mean(large_errors)), 4) if large_errors else 0.0,
        "meanTop3OverlapRate": round(float(np.mean(top3_overlaps)), 4),
        "top1Accuracy": round(float(np.mean(top1_hits)), 4) if top1_hits else 0.0,
        "sapporo11CaseStudy": sapporo_case,
        "futureModelCoefficients": {
            "intercept": float(final_beta[0]),
            **{name: float(v) for name, v in zip(FEATURE_COLS, final_beta[1:])},
        },
        "teacherRows": int(len(teacher)),
        "teacherRaces": int(teacher["race_id"].nunique()),
    }
    return estimates, metrics


def build_future_entity_priors(history: pd.DataFrame, target_dates: list[str]) -> dict:
    """Market priors for current jockey/trainer, trained only on past race popularity."""
    h = history[
        history["_market_strength"].notna()
        & history["_date"].notna()
        & (history["_date"] <= pd.Timestamp(max(target_dates)))
    ].copy()

    def make_map(col: str, prior_weight: float) -> dict[str, float]:
        out = {}
        if h.empty or col not in h.columns:
            return out
        for entity, group in h[h[col].ne("")].groupby(col):
            score = shrunk_market_mean(
                group["_market_strength"], prior=0.5, prior_weight=prior_weight
            )
            out[str(entity)] = round(float(score), 6)
        return out

    return {
        "jockey": make_map("_jockey_id_norm", 20.0),
        "trainer": make_map("_trainer_id_norm", 30.0),
        "jockeyName": make_map("_jockey_name_norm", 20.0),
        "trainerName": make_map("_trainer_name_norm", 30.0),
    }



def build_result(race: RaceModel) -> tuple[dict, list[int]]:
    r = race.result.copy()
    r["finish_num"] = pd.to_numeric(r["finish_position"], errors="coerce")
    r["horse_num"] = pd.to_numeric(r["horse_number"], errors="coerce")
    top = r[r["finish_num"].notna() & (r["finish_num"] <= 3)].copy()
    if top.empty:
        raise ValueError(f"{race.race_id}: no top-3 result")

    places = []
    for pos in sorted(top["finish_num"].astype(int).unique()):
        group = top[top["finish_num"].astype(int) == pos]["horse_num"].astype(int).tolist()
        if group:
            places.append(group)

    p = race.payout.copy()
    tri = p[p["bet_type"].astype(str).isin(["三連単", "3連単"])].copy()
    if tri.empty:
        raise ValueError(f"{race.race_id}: no trifecta payout row")

    trifectas = []
    for _, row in tri.iterrows():
        nums = [int(x) for x in re.findall(r"\d+", clean_str(row["combination"]))]
        if len(nums) != 3:
            raise ValueError(f"{race.race_id}: bad trifecta combination {row['combination']}")
        amount = int(pd.to_numeric(pd.Series([row["amount"]]), errors="raise").iloc[0])
        trifectas.append({"horses": nums, "payout": amount})

    return {"places": places, "trifectas": trifectas}, [t["payout"] for t in trifectas]


def covered(prediction: dict, horses: list[int]) -> bool:
    combo = set(horses)
    axes = set(prediction.get("axes", []))
    opponents = set(prediction.get("opponents", []))
    return axes.issubset(combo) and bool(opponents & combo)


def normalize_old_result(race: dict) -> tuple:
    result = race.get("result") or {}
    places = tuple(tuple(int(x) for x in g) for g in result.get("places", []))
    tris = tuple(sorted(
        (tuple(int(x) for x in t.get("horses", [])), int(t.get("payout", 0)))
        for t in result.get("trifectas", [])
    ))
    return places, tris


def normalize_new_result(result: dict) -> tuple:
    places = tuple(tuple(int(x) for x in g) for g in result.get("places", []))
    tris = tuple(sorted(
        (tuple(int(x) for x in t["horses"]), int(t["payout"]))
        for t in result.get("trifectas", [])
    ))
    return places, tris


def inspect_prediction_date(pred_dir: Path) -> dict:
    """Classify one archived date before any race is rebuilt."""
    valid_files: list[str] = []
    ignored_files: list[str] = []
    contaminated_files: list[dict] = []
    schema_errors: list[dict] = []

    for path in sorted(pred_dir.glob("*.csv")):
        # Ignore scratch/test files that are not a real 12-digit JRA race id.
        if not re.fullmatch(r"\d{12}\.csv", path.name):
            ignored_files.append(path.name)
            continue

        try:
            pred = read_csv(path)
        except Exception as exc:
            schema_errors.append({
                "file": path.name,
                "reason": f"{type(exc).__name__}: {exc}",
            })
            continue

        missing = {"horse_number"} - set(pred.columns)
        if missing:
            schema_errors.append({
                "file": path.name,
                "reason": "missing core columns: " + ", ".join(sorted(missing)),
            })
            continue

        issues = prediction_snapshot_issues(pred)
        if issues:
            contaminated_files.append({
                "file": path.name,
                "issues": issues,
            })
            continue

        valid_files.append(path.name)

    if contaminated_files:
        reason = (
            f"{len(contaminated_files)} prediction files contain current-race "
            "odds/bodyweight/popularity"
        )
        safe = False
    elif schema_errors:
        reason = f"{len(schema_errors)} prediction files have unsupported schema/read errors"
        safe = False
    elif not valid_files:
        reason = "no valid 12-digit race prediction files"
        safe = False
    else:
        reason = ""
        safe = True

    return {
        "safe": safe,
        "validRaceFiles": valid_files,
        "ignoredFiles": ignored_files,
        "contaminatedFiles": contaminated_files,
        "schemaErrors": schema_errors,
        "reason": reason,
    }


def discover_target_dates(
    source_root: Path,
    scope: str,
    start_date: str,
    end_date: str,
) -> tuple[list[str], dict]:
    pred_root = source_root / "data" / "predictions"
    available: list[str] = []
    date_dirs: dict[str, Path] = {}

    for path in sorted(pred_root.iterdir() if pred_root.exists() else []):
        if not path.is_dir() or not re.fullmatch(r"20\d{6}", path.name):
            continue
        if not any(path.glob("*.csv")):
            continue
        try:
            dt = datetime.strptime(path.name, "%Y%m%d").date()
        except ValueError:
            continue
        date_s = dt.isoformat()
        available.append(date_s)
        date_dirs[date_s] = path

    if not available:
        raise RuntimeError("No archived prediction dates are available")

    if scope == "all":
        requested = list(available)
    else:
        if not start_date or not end_date:
            raise ValueError("range scope requires both --start-date and --end-date")
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        if start > end:
            raise ValueError("start-date must be on or before end-date")
        requested = [d for d in available if start.isoformat() <= d <= end.isoformat()]
        if not requested:
            raise RuntimeError(
                f"No archived prediction dates found in {start_date}..{end_date}. "
                f"Available dates: {', '.join(available)}"
            )

    selected: list[str] = []
    skipped: list[dict] = []
    ignored_files: list[dict] = []

    for date_s in requested:
        inspection = inspect_prediction_date(date_dirs[date_s])
        if inspection["ignoredFiles"]:
            ignored_files.append({
                "date": date_s,
                "files": inspection["ignoredFiles"],
            })
        if inspection["safe"]:
            selected.append(date_s)
        else:
            skipped.append({
                "date": date_s,
                "reason": inspection["reason"],
                "contaminatedFiles": inspection["contaminatedFiles"],
                "schemaErrors": inspection["schemaErrors"],
            })

    if not selected:
        detail = "; ".join(f"{x['date']}: {x['reason']}" for x in skipped)
        raise RuntimeError(
            "No safe pre-race prediction dates remain after snapshot validation. "
            + detail
        )

    return selected, {
        "availableDates": available,
        "requestedDates": requested,
        "skippedDates": skipped,
        "ignoredFiles": ignored_files,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-commit", default="")
    parser.add_argument("--cache-manifest", default="", type=Path)
    parser.add_argument("--scope", choices=("all", "range"), default="range")
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--data-path", default="data/races.json", type=Path)
    parser.add_argument("--audit-path", default="data/rebuild_audit.json", type=Path)
    parser.add_argument("--pop-model-path", default="data/popularity_model.json", type=Path)
    parser.add_argument("--diagnostic-path", default=DIAGNOSTIC_PATH_DEFAULT, type=Path)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    if not source_root.exists():
        raise FileNotFoundError(source_root)
    if not args.data_path.exists():
        raise FileNotFoundError(args.data_path)

    target_dates, discovery = discover_target_dates(
        source_root, args.scope, args.start_date.strip(), args.end_date.strip()
    )
    print("Target dates:", ", ".join(target_dates))
    if discovery["skippedDates"]:
        print("Skipped unsafe dates:")
        for item in discovery["skippedDates"]:
            print(f"  - {item['date']}: {item['reason']}")
    if discovery["ignoredFiles"]:
        print("Ignored non-race CSV files:")
        for item in discovery["ignoredFiles"]:
            print(f"  - {item['date']}: {', '.join(item['files'])}")

    cache_manifest = {}
    if args.cache_manifest and args.cache_manifest.is_file():
        try:
            cache_manifest = json.loads(args.cache_manifest.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Failed to read cache manifest: {exc}") from exc

    source_sha = args.source_commit.strip() or cache_manifest.get("sourceCommit", "") or source_commit(source_root)
    races: dict[str, RaceModel] = {}
    target_files: list[tuple[str, Path]] = []
    target_horse_ids: set[str] = set()
    expected_by_date: dict[str, int] = {}

    for date_s in target_dates:
        pred_dir = source_root / "data" / "predictions" / date_s.replace("-", "")
        files = sorted(
            f for f in pred_dir.glob("*.csv")
            if re.fullmatch(r"\d{12}\.csv", f.name)
        )
        if not files:
            raise RuntimeError(f"{date_s}: no valid 12-digit race prediction files")
        expected_by_date[date_s] = len(files)
        for f in files:
            target_files.append((date_s, f))
            card_path = (
                source_root / "data" / "race_cards" / date_s.replace("-", "")
                / f"{f.stem}.csv"
            )
            card = read_csv(card_path)
            if "horse_id" in card.columns:
                target_horse_ids.update(card["horse_id"].apply(clean_str))

    expected_total_races = sum(expected_by_date.values())
    print(
        f"Selected {len(target_dates)} dates / {expected_total_races} races: "
        + ", ".join(f"{d}={expected_by_date[d]}" for d in target_dates)
    )

    print(f"Loading historical races for {len(target_horse_ids)} target horses...")
    history = load_target_history(source_root, target_horse_ids)
    if history.empty:
        raise RuntimeError("No pre-race historical results could be loaded")
    print(
        f"Historical rows={len(history)} races={history['race_id'].nunique()} "
        f"horses={history['horse_id'].nunique()}"
    )

    build_errors = []
    for date_s, f in target_files:
        print(f"Rebuilding {f.stem} ({date_s})...")
        try:
            race = build_race_model(source_root, date_s, f, history)
            if race.race_id in races:
                raise RuntimeError(f"duplicate race {race.race_id}")
            races[race.race_id] = race
        except Exception as exc:
            build_errors.append({
                "date": date_s,
                "raceId": f.stem,
                "errorType": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            })
            print(f"  ERROR {f.stem}: {type(exc).__name__}: {exc}")

    if build_errors or len(races) != expected_total_races:
        diagnostic = {
            "generatedAt": datetime.now(JST).isoformat(timespec="seconds"),
            "stage": "race-preflight",
            "targetDates": target_dates,
            "availableDates": discovery["availableDates"],
            "requestedDates": discovery["requestedDates"],
            "skippedDates": discovery["skippedDates"],
            "ignoredFiles": discovery["ignoredFiles"],
            "expectedRaces": expected_total_races,
            "builtRaces": len(races),
            "errorCount": len(build_errors),
            "errors": build_errors,
        }
        args.diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        args.diagnostic_path.write_text(
            json.dumps(diagnostic, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summary = "\n".join(
            f"- {e['raceId']}: {e['errorType']}: {e['message']}"
            for e in build_errors
        )
        raise RuntimeError(
            f"Preflight found {len(build_errors)} critical race errors "
            f"({len(races)}/{expected_total_races} built).\n{summary}"
        )

    estimated_pops, pop_metrics = estimate_popularities(races)
    future_entity_priors = build_future_entity_priors(history, target_dates)

    data = json.loads(args.data_path.read_text(encoding="utf-8"))
    existing_by_date = {d.get("date"): d for d in data.get("days", [])}

    audit_races = []
    daily = {}
    old_result_mismatches = []

    for date_s in target_dates:
        day = existing_by_date.get(date_s)
        if day is None:
            day = {"date": date_s, "races": []}
            data.setdefault("days", []).append(day)
            existing_by_date[date_s] = day

        old_by_id = {r.get("raceId"): r for r in day.get("races", [])}
        rebuilt = []

        day_races = [r for r in races.values() if r.date == date_s]
        day_races.sort(key=lambda x: (int(x.race_id[4:6]), int(x.race_id[-2:])))

        day_hits = 0
        day_payout = 0
        day_stake = 0

        for race in day_races:
            rid = race.race_id
            est_pop = estimated_pops[rid]
            prediction, danger, _target_count = select_prediction(
                race.index_detail.get("horses", []),
                race.total_internal,
                est_pop,
            )
            result, tri_payouts = build_result(race)

            old = old_by_id.get(rid, {})
            if old.get("result") and normalize_old_result(old) != normalize_new_result(result):
                old_result_mismatches.append(rid)

            winners = [t for t in result["trifectas"] if covered(prediction, t["horses"])]
            payout_return = sum(int(t["payout"]) for t in winners)
            stake = len(prediction["opponents"]) * 6 * 100
            status = "hit" if winners else "miss"

            if status == "hit":
                day_hits += 1
            day_payout += payout_return
            day_stake += stake

            card = race.card.copy()
            card["horse_number"] = pd.to_numeric(card["horse_number"], errors="raise").astype(int)
            frames = {
                str(int(row["horse_number"])): int(row["waku"])
                for _, row in card.iterrows()
                if pd.notna(row.get("waku"))
            }
            names = {
                str(int(row["horse_number"])): clean_str(row["horse_name"])
                for _, row in card.iterrows()
            }

            index_detail = json.loads(json.dumps(race.index_detail, ensure_ascii=False))
            index_horses = {int(h["no"]): h for h in index_detail.get("horses", [])}
            for no, h in index_horses.items():
                h["expectedPopularity"] = int(est_pop[no])
                h["excluded"] = int(no) == int(danger)
            index_detail["prediction"] = {
                "axes": list(prediction["axes"]),
                "opponents": list(prediction["opponents"]),
                "excluded": [int(danger)],
            }

            updated = dict(old)
            updated.pop("seedNote", None)
            updated.update({
                "raceId": rid,
                "venue": TRACKS.get(rid[4:6], rid[4:6]),
                "raceNo": int(rid[-2:]),
                "horseCount": int(len(card)),
                "horseFrames": frames,
                "horseNames": names,
                "prediction": prediction,
                "danger": [int(danger)],
                "result": result,
                "status": status,
                "payout": int(payout_return),
                "trifectaPayouts": [int(x) for x in tri_payouts],
                "stake": int(stake),
                "modelMeta": {
                    "version": MODEL_VERSION,
                    "rebuildVersion": REBUILD_VERSION,
                    "estimatedPopularity": {str(k): int(v) for k, v in est_pop.items()},
                    "totalIndex": {str(k): int(v) for k, v in race.total_display.items()},
                    "indexDetail": index_detail,
                    "performanceSource": "pre-race historical reconstruction via shared prediction core",
                    "popularityMethod": "market-memory v2 leave-one-race-out calibrated model",
                    "selectionRule": SELECTION_RULE_TEXT,
                    "logicSource": "scripts/prediction_logic.py",
                    "nonStarters": sorted(
                        set(int(x) for x in race.card["horse_number"])
                        - set(race.actual_popularity)
                    ),
                },
                "dataSources": {
                    "preRaceSnapshot": f"{SOURCE_REPO}@{source_sha or SOURCE_REF}",
                    "resultArchive": f"{SOURCE_REPO}@{source_sha or SOURCE_REF}",
                    "payoutArchive": f"{SOURCE_REPO}@{source_sha or SOURCE_REF}",
                },
            })
            rebuilt.append(updated)

            actual_pop = race.actual_popularity
            est_order = [
                h for h, _ in sorted(est_pop.items(), key=lambda kv: kv[1]) if h in actual_pop
            ]
            est_pop_starters = {h: i + 1 for i, h in enumerate(est_order)}
            mae = float(np.mean([
                abs(est_pop_starters[h] - actual_pop[h]) for h in est_pop_starters
            ]))
            audit_races.append({
                "date": date_s,
                "raceId": rid,
                "venue": updated["venue"],
                "raceNo": updated["raceNo"],
                "horseCount": updated["horseCount"],
                "prediction": prediction,
                "danger": danger,
                "result": result["places"],
                "trifectas": result["trifectas"],
                "status": status,
                "return": payout_return,
                "stake": stake,
                "recoveryRate": round(payout_return / stake * 100, 1) if stake else 0.0,
                "popularityMAE": round(mae, 3),
                "indexDetailGenerated": True,
                "indexHorseCount": len(index_detail.get("horses", [])),
                "indexQualityWarnings": index_detail.get("qualityWarnings", []),
                "nonStarters": sorted(
                    set(int(x) for x in race.card["horse_number"])
                    - set(race.actual_popularity)
                ),
                "oldResultMatchedArchive": rid not in old_result_mismatches,
            })

        expected_for_day = expected_by_date[date_s]
        if len(rebuilt) != expected_for_day:
            raise RuntimeError(f"{date_s}: rebuilt {len(rebuilt)} != {expected_for_day} races")

        day["races"] = rebuilt
        daily[date_s] = {
            "races": len(rebuilt),
            "hits": day_hits,
            "return": day_payout,
            "stake": day_stake,
            "recoveryRate": round(day_payout / day_stake * 100, 1) if day_stake else 0.0,
        }

    data["updatedAt"] = datetime.now(JST).isoformat(timespec="seconds")
    data["days"] = sorted(data.get("days", []), key=lambda d: d.get("date", ""), reverse=True)
    for day in data["days"]:
        day["races"] = sorted(
            day.get("races", []),
            key=lambda r: (TRACK_ORDER.get(r.get("venue", ""), 99), int(r.get("raceNo", 99))),
        )

    for date_s in target_dates:
        day = next(d for d in data["days"] if d.get("date") == date_s)
        expected_for_day = expected_by_date[date_s]
        if len(day["races"]) != expected_for_day:
            raise RuntimeError(f"{date_s}: final race count invalid")
        ids = [r["raceId"] for r in day["races"]]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"{date_s}: duplicate race IDs")
        for r in day["races"]:
            p = r["prediction"]
            target_n = prediction_target_count(r["horseCount"])
            if 2 + len(p["opponents"]) != target_n:
                raise RuntimeError(f"{r['raceId']}: selected count invalid")
            if set(p["axes"]) & set(p["opponents"]):
                raise RuntimeError(f"{r['raceId']}: axes/opponents overlap")
            if r.get("danger", [None])[0] in set(p["axes"] + p["opponents"]):
                raise RuntimeError(f"{r['raceId']}: danger horse selected")
            expected_stake = len(p["opponents"]) * 600
            if r["stake"] != expected_stake:
                raise RuntimeError(f"{r['raceId']}: stake invalid")

            detail = r.get("modelMeta", {}).get("indexDetail")
            if not detail:
                raise RuntimeError(f"{r['raceId']}: indexDetail missing")
            horses = detail.get("horses", [])
            if len(horses) != r["horseCount"]:
                raise RuntimeError(
                    f"{r['raceId']}: indexDetail horse count {len(horses)} != {r['horseCount']}"
                )
            for h in horses:
                if len(h.get("recent", [])) != 5:
                    raise RuntimeError(f"{r['raceId']}: recent index count invalid")
                for key in ("recentIndex", "pace", "course", "today", "total", "rank"):
                    if key not in h:
                        raise RuntimeError(f"{r['raceId']}: index field {key} missing")

    summary_totals = {
        "dates": len(target_dates),
        "races": sum(v["races"] for v in daily.values()),
        "hits": sum(v["hits"] for v in daily.values()),
        "return": sum(v["return"] for v in daily.values()),
        "stake": sum(v["stake"] for v in daily.values()),
    }
    summary_totals["recoveryRate"] = round(
        summary_totals["return"] / summary_totals["stake"] * 100, 1
    ) if summary_totals["stake"] else 0.0

    audit = {
        "version": REBUILD_VERSION,
        "generatedAt": datetime.now(JST).isoformat(timespec="seconds"),
        "targetDates": target_dates,
        "availableDates": discovery["availableDates"],
        "requestedDates": discovery["requestedDates"],
        "skippedDates": discovery["skippedDates"],
        "ignoredFiles": discovery["ignoredFiles"],
        "sourceRepository": SOURCE_REPO,
        "sourceRef": SOURCE_REF,
        "sourceCommit": source_sha,
        "historicalFactsCache": {
            "enabled": bool(cache_manifest),
            "cacheVersion": cache_manifest.get("cacheVersion"),
            "generatedAt": cache_manifest.get("generatedAt"),
            "archiveSha256": (cache_manifest.get("archive") or {}).get("sha256"),
            "cachedSafeDates": cache_manifest.get("safeDates", []),
            "cachedFileCount": cache_manifest.get("cachedFileCount"),
            "historicalResultFiles": cache_manifest.get("historicalResultFiles"),
        },
        "sharedPredictionLogic": {
            "modelVersion": MODEL_VERSION,
            "source": "scripts/prediction_logic.py",
            "selectionRule": SELECTION_RULE_TEXT,
        },
        "strictInputRules": {
            "currentOddsUsed": False,
            "currentPopularityUsedAsPredictionFeature": False,
            "horseWeightOrChangeUsed": False,
            "predictionSnapshotPolicy": (
                "dates whose prediction CSVs contain current odds, current actual "
                "popularity, or horse bodyweight are excluded as unsafe; post-race "
                "market/bodyweight columns in race_cards are dropped and never used"
            ),
            "legacySnapshotPolicy": (
                "clean early snapshots without optional ml_win_prob/ml_rank remain eligible; "
                "archived model scores/ranks are ignored and all derived indices are rebuilt "
                "from immutable pre-race facts through scripts/prediction_logic.py"
            ),
            "raceResultUsedAsPerformanceFeature": False,
            "actualPopularityUse": (
                "teacher label only; each race excluded from its own estimator training"
            ),
            "nonStarterHandling": (
                "cancelled/excluded horses are retained in pre-race prediction candidates "
                "but omitted from final-popularity teacher rows and validation metrics"
            ),
            "indexDetailHistoryCutoff": (
                "previous-run/detail indices use only cached archived race results with date < target race date"
            ),
            "historicalFactsCachePolicy": (
                "immutable source facts are cached once; derived indices and prediction selections "
                "are recalculated on every rebuild so logic changes take effect immediately"
            ),
            "raceConditionMetadataFallback": (
                "surface/distance may fall back to target result metadata or an authoritative "
                "race-program override; if still missing, course index degrades neutrally "
                "instead of aborting. Target result performance is never used"
            ),
            "indexDetailFailurePolicy": (
                "no legacy score fallback is allowed; every race must be derivable through the "
                "shared prediction core, otherwise the rebuild reports the input error"
            ),
        },
        "summary": summary_totals,
        "raceCount": len(audit_races),
        "oldResultMismatchCount": len(old_result_mismatches),
        "oldResultMismatches": old_result_mismatches,
        "popularityModel": pop_metrics,
        "daily": daily,
        "races": audit_races,
    }

    pop_model = {
        "version": "predictjra-popularity-calibration-generic-v4-unified-market-memory",
        "logicSource": "scripts/prediction_logic.py",
        "modelVersion": MODEL_VERSION,
        "trainedAt": datetime.now(JST).isoformat(timespec="seconds"),
        "teacherDates": target_dates,
        "teacherRaces": pop_metrics["teacherRaces"],
        "teacherRows": pop_metrics["teacherRows"],
        "features": pop_metrics["features"],
        "ridge": pop_metrics["ridge"],
        "coefficients": pop_metrics["futureModelCoefficients"],
        "entityPriors": future_entity_priors,
        "validation": {
            "method": pop_metrics["method"],
            "meanAbsolutePopularityRankError": pop_metrics["meanAbsolutePopularityRankError"],
            "medianAbsolutePopularityRankError": pop_metrics["medianAbsolutePopularityRankError"],
            "maxAbsolutePopularityRankError": pop_metrics["maxAbsolutePopularityRankError"],
            "largeError5PlusRate": pop_metrics["largeError5PlusRate"],
            "meanTop3OverlapRate": pop_metrics["meanTop3OverlapRate"],
            "top1Accuracy": pop_metrics["top1Accuracy"],
            "sapporo11CaseStudy": pop_metrics["sapporo11CaseStudy"],
        },
        "allowedHistoricalMarketInputs": [
            "previous-race popularity",
            "previous assigned weight",
            "historical jockey market tendency",
            "historical trainer market tendency",
        ],
        "prohibitedInputs": [
            "current odds", "current actual popularity", "horse bodyweight", "horse bodyweight change"
        ],
    }

    args.data_path.parent.mkdir(parents=True, exist_ok=True)
    args.audit_path.parent.mkdir(parents=True, exist_ok=True)
    args.pop_model_path.parent.mkdir(parents=True, exist_ok=True)
    args.data_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.pop_model_path.write_text(
        json.dumps(pop_model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    args.diagnostic_path.write_text(
        json.dumps({
            "generatedAt": datetime.now(JST).isoformat(timespec="seconds"),
            "stage": "complete",
            "targetDates": target_dates,
            "availableDates": discovery["availableDates"],
            "requestedDates": discovery["requestedDates"],
            "skippedDates": discovery["skippedDates"],
            "ignoredFiles": discovery["ignoredFiles"],
            "expectedRaces": expected_total_races,
            "builtRaces": len(audit_races),
            "errorCount": 0,
            "warnings": [
                {
                    "raceId": r["raceId"],
                    "qualityWarnings": r.get("modelMeta", {}).get("indexDetail", {}).get("qualityWarnings", []),
                }
                for d in data.get("days", [])
                if d.get("date") in target_dates
                for r in d.get("races", [])
                if r.get("modelMeta", {}).get("indexDetail", {}).get("qualityWarnings")
            ],
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "targetDates": target_dates,
        "skippedDates": discovery["skippedDates"],
        "ignoredFiles": discovery["ignoredFiles"],
        "summary": summary_totals,
        "daily": daily,
        "oldResultMismatchCount": len(old_result_mismatches),
        "popularityMAE": pop_metrics["meanAbsolutePopularityRankError"],
        "top3Overlap": pop_metrics["meanTop3OverlapRate"],
        "historicalFactsCache": {
            "enabled": bool(cache_manifest),
            "cacheVersion": cache_manifest.get("cacheVersion"),
            "sourceCommit": source_sha,
        },
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
