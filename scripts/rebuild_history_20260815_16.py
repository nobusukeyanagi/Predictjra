#!/usr/bin/env python3
"""Rebuild Predictjra predictions/results for 2026-08-15 and 2026-08-16.

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
* Refuse to write partial data: 72 races and all required validation checks must pass.

Historical snapshot source:
  https://github.com/sugaimo15/keibayosoku
  ref: claude/horse-racing-predictor-ak6crm

The archived prediction CSVs were created before odds/bodyweight were populated. This
script verifies that condition again before using their non-market pre-race score fields.
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

JST = ZoneInfo("Asia/Tokyo")
TARGET_DATES = ("2026-08-15", "2026-08-16")
EXPECTED_RACES_PER_DAY = 36
EXPECTED_TOTAL_RACES = 72

TRACKS = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
}
TRACK_ORDER = {name: i for i, name in enumerate(TRACKS.values(), start=1)}

# This race already has the fully hand-reviewed Predictjra detailed index.
# Keep its "総合" ordering from the modal; only 想人 is replaced by the
# cross-race calibrated popularity estimator.
SAPPORO11_ID = "202601010811"
SAPPORO11_TOTAL = {
    1: 65, 2: 65, 3: 64, 4: 80, 5: 75, 6: 74, 7: 79, 8: 87,
    9: 75, 10: 83, 11: 75, 12: 75, 13: 81, 14: 79, 15: 78, 16: 74,
}
# Secondary ordering used by the current detailed index when displayed totals tie.
SAPPORO11_RANK = {
    8: 1, 10: 2, 13: 3, 4: 4, 7: 5, 14: 6, 15: 7, 12: 8,
    9: 9, 11: 10, 5: 11, 16: 12, 6: 13, 1: 14, 2: 15, 3: 16,
}

REBUILD_VERSION = "predictjra-history-20260815-16-v3-market-memory"
SOURCE_REPO = "sugaimo15/keibayosoku"
SOURCE_REF = "claude/horse-racing-predictor-ak6crm"

# Known race-condition metadata gaps in the archived source.
# Values here come from authoritative pre-race race-program information.
RACE_METADATA_OVERRIDES = {
    "202604020709": {"surface": "障害", "distance_m": 3250.0},  # 新潟ジャンプS
}

DIAGNOSTIC_PATH_DEFAULT = Path("data/rebuild_diagnostics_20260815_16.json")


def clean_str(v) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip()


def numeric_or_nan(v) -> float:
    return pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]


def minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    if s.notna().sum() == 0:
        return pd.Series(0.5, index=series.index, dtype=float)
    s = s.fillna(s.mean())
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-12:
        return pd.Series(0.5, index=series.index, dtype=float)
    return (s - lo) / (hi - lo)


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


def market_recency(values: list[float], limit: int = 5) -> float:
    vals = [float(v) for v in values[:limit] if math.isfinite(float(v))]
    if not vals:
        return 0.5
    weights = [0.40, 0.25, 0.16, 0.11, 0.08][:len(vals)]
    sw = sum(weights)
    return clipped01(sum(v * w for v, w in zip(vals, weights)) / sw)


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


def clamp_index(value: float, lo: int = 45, hi: int = 98) -> float:
    if not math.isfinite(float(value)):
        return float((lo + hi) / 2)
    return float(np.clip(value, lo, hi))


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
    hist["_last3f"] = pd.to_numeric(hist.get("last_3f"), errors="coerce")

    numeric_finish = hist["_finish"].notna()
    hist["_field_size"] = (
        numeric_finish.astype(int)
        .groupby(hist["race_id"])
        .transform("sum")
        .replace(0, np.nan)
    )
    hist["_winner_time"] = hist["_time_sec"].groupby(hist["race_id"]).transform("min")
    hist["_last3f_min"] = hist["_last3f"].groupby(hist["race_id"]).transform("min")
    hist["_last3f_max"] = hist["_last3f"].groupby(hist["race_id"]).transform("max")

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

    denom = (hist["_field_size"] - 1).replace(0, 1)
    hist["_finish_strength"] = (1 - (hist["_finish"] - 1) / denom).clip(0, 1)

    # 成績指数: 着順を頭数で補正。
    hist["_result_index"] = (48 + 50 * hist["_finish_strength"]).clip(45, 98)

    # タイム指数: 勝ち馬とのタイム差を中心に、上がり3Fの相対値を補助。
    gap = (hist["_time_sec"] - hist["_winner_time"]).clip(lower=0)
    gap_score = (96 - 6.0 * gap).clip(45, 98)
    last_span = (hist["_last3f_max"] - hist["_last3f_min"]).replace(0, np.nan)
    last_strength = 1 - (hist["_last3f"] - hist["_last3f_min"]) / last_span
    last_score = (55 + 40 * last_strength).clip(50, 96)
    hist["_time_index"] = (
        0.75 * gap_score + 0.25 * last_score.fillna(gap_score)
    ).clip(45, 98)

    # 各過去走の展開指数: 位置取りからの前進度と上がりを加味。
    pace_values = []
    front_values = []
    for _, row in hist.iterrows():
        finish = row.get("_finish")
        field = row.get("_field_size")
        pos = passing_positions(row.get("passing_order"))
        finish_strength = row.get("_finish_strength")
        l3min, l3max, l3 = row.get("_last3f_min"), row.get("_last3f_max"), row.get("_last3f")

        if pd.notna(field) and field > 1 and pos:
            first = pos[0]
            last = pos[-1]
            front = 1 - (first - 1) / max(field - 1, 1)
            improvement = (last - finish) / max(field, 1) if pd.notna(finish) else 0.0
        else:
            front = math.nan
            improvement = 0.0

        if pd.notna(l3) and pd.notna(l3min) and pd.notna(l3max) and l3max > l3min:
            l3s = 1 - (l3 - l3min) / (l3max - l3min)
        else:
            l3s = 0.5

        fs = float(finish_strength) if pd.notna(finish_strength) else 0.5
        if math.isfinite(float(front)):
            score = 64 + 18 * fs + 10 * np.clip(improvement, -1, 1) + 8 * l3s
        else:
            score = 62 + 26 * fs + 7 * l3s
        pace_values.append(clamp_index(score))
        front_values.append(front)

    hist["_pace_index"] = pace_values
    hist["_front_strength"] = front_values
    hist["_run_composite"] = (
        0.25 * hist["_pace_index"]
        + 0.35 * hist["_time_index"]
        + 0.40 * hist["_result_index"]
    )

    return hist


def recency_weighted(values: list[float]) -> float:
    if not values:
        return 70.0
    # 近走指数は「直近5走」が定義。呼び出し側から5件超が渡っても
    # 6要素×5重みの形状不一致を起こさないよう、この関数自身でも防御する。
    values = list(values)[:5]
    base_weights = [0.34, 0.25, 0.18, 0.13, 0.10]
    weights = np.array(base_weights[:len(values)], dtype=float)
    weights = weights / weights.sum()
    return float(np.dot(np.array(values, dtype=float), weights))


def course_index_for_horse(
    horse_hist: pd.DataFrame,
    current_race_id: str,
    surface: str,
    distance_m: float,
    proxy: float,
) -> float:
    """Course suitability with graceful degradation for incomplete metadata.

    Missing distance/surface is not grounds to abort an otherwise valid race.
    Exact-course evidence is used when available; otherwise venue/distance history
    is used, and finally the pre-race proxy is returned as a neutral fallback.
    """
    if horse_hist.empty:
        return clamp_index(min(78, 0.82 * proxy + 0.18 * 72))

    track_code = current_race_id[4:6]
    h = horse_hist.copy()
    h["_track_code"] = h["race_id"].astype(str).str[4:6]
    h["_distance"] = pd.to_numeric(h.get("distance_m"), errors="coerce")
    h["_surface"] = h.get(
        "surface", pd.Series(index=h.index, dtype=object)
    ).apply(clean_str)

    valid_distance = math.isfinite(float(distance_m)) and float(distance_m) > 0
    valid_surface = bool(clean_str(surface))

    def score(frame: pd.DataFrame, cap: float) -> float | None:
        if frame.empty:
            return None
        vals = (
            pd.to_numeric(frame["_run_composite"], errors="coerce")
            .dropna()
            .tail(5)
            .tolist()
        )
        if not vals:
            return None
        value = 0.72 * recency_weighted(list(reversed(vals))) + 0.28 * max(vals)
        return min(cap, value)

    # Best case: same venue + same surface + nearly same distance.
    if valid_distance and valid_surface:
        exact = h[
            (h["_track_code"] == track_code)
            & (h["_surface"] == surface)
            & ((h["_distance"] - float(distance_m)).abs() <= 100)
        ]
        exact_score = score(exact, 98)
        if exact_score is not None:
            return clamp_index(0.84 * exact_score + 0.16 * proxy)

    # Same venue; retain surface filter when known.
    venue = h[h["_track_code"] == track_code]
    if valid_surface:
        venue = venue[venue["_surface"] == surface]
    venue_score = score(venue, 86 if valid_surface else 82)
    if venue_score is not None:
        return clamp_index(0.78 * venue_score + 0.22 * proxy)

    # Same/similar distance elsewhere; retain surface filter when known.
    if valid_distance:
        distance = h[(h["_distance"] - float(distance_m)).abs() <= 200]
        if valid_surface:
            distance = distance[distance["_surface"] == surface]
        distance_score = score(distance, 82 if valid_surface else 79)
        if distance_score is not None:
            return clamp_index(0.76 * distance_score + 0.24 * proxy)

    # No trustworthy course metadata/history: neutral pre-race fallback.
    return clamp_index(min(78, 0.88 * proxy + 0.12 * 72))


def build_index_detail(
    race_id: str,
    date_s: str,
    card: pd.DataFrame,
    pred: pd.DataFrame,
    result: pd.DataFrame,
    history: pd.DataFrame,
) -> tuple[dict, dict[int, float], dict[int, int], dict[int, float]]:
    target_date = pd.Timestamp(date_s)
    card2 = card.copy()
    card2["horse_number"] = pd.to_numeric(card2["horse_number"], errors="raise").astype(int)
    card2["horse_id"] = card2["horse_id"].apply(clean_str)

    pmap = pred.set_index("horse_number")

    # Race conditions are known before the race. Prefer archived pre-race metadata,
    # then result metadata strictly for condition fields, then an authoritative
    # per-race override. If still missing, continue with a neutral course fallback.
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

    horse_context = {}
    front_type_count = 0

    for _, row in card2.iterrows():
        no = int(row["horse_number"])
        horse_id = clean_str(row.get("horse_id"))
        h = history[
            (history["horse_id"] == horse_id)
            & (history["_date"].notna())
            & (history["_date"] < target_date)
            & (history["_finish"].notna())
        ].sort_values("_date")
        recent5 = h.tail(5).iloc[::-1].copy()

        fronts = pd.to_numeric(recent5.get("_front_strength"), errors="coerce").dropna()
        front_ratio = float(fronts.mean()) if len(fronts) else math.nan
        if math.isfinite(front_ratio) and front_ratio >= 0.58:
            front_type_count += 1

        proxy = 55 + 40 * float(pmap.loc[no, "rule_norm"])
        horse_context[no] = {
            "row": row,
            "history": h,
            "recent5": recent5,
            "front_ratio": front_ratio,
            "proxy": proxy,
        }

    if front_type_count >= 3:
        pace_regime = "fast"
    elif front_type_count <= 1:
        pace_regime = "slow"
    else:
        pace_regime = "medium"

    horses = []
    internal_totals = {}

    for no in sorted(horse_context):
        ctx = horse_context[no]
        row = ctx["row"]
        recent5 = ctx["recent5"]
        proxy = float(ctx["proxy"])
        front_ratio = ctx["front_ratio"]

        recent_strings = []
        composites = []
        for _, rr in recent5.iterrows():
            pace = int(round(clamp_index(rr["_pace_index"])))
            timing = int(round(clamp_index(rr["_time_index"])))
            result_i = int(round(clamp_index(rr["_result_index"])))
            recent_strings.append(f"{pace}/{timing}/{result_i}")
            composites.append(0.25 * pace + 0.35 * timing + 0.40 * result_i)
        while len(recent_strings) < 5:
            recent_strings.append("評価外")

        if composites:
            base = recency_weighted(composites)
            ceiling = max(composites)
            consistency = clamp_index(96 - 1.8 * float(np.std(composites)), 55, 96)
            hist_recent = 0.70 * base + 0.20 * ceiling + 0.10 * consistency
            history_weight = min(len(composites) / 5.0, 1.0) * 0.84
            recent_index = history_weight * hist_recent + (1 - history_weight) * proxy
        else:
            recent_index = proxy
        recent_index = clamp_index(recent_index)

        if math.isfinite(float(front_ratio)):
            if pace_regime == "fast":
                style = 70 + 20 * (1 - front_ratio)
            elif pace_regime == "slow":
                style = 70 + 20 * front_ratio
            else:
                style = 76 + 10 * (1 - abs(front_ratio - 0.5) * 2)
            pace_index = clamp_index(0.78 * style + 0.22 * proxy)
        else:
            pace_index = clamp_index(0.78 * proxy + 0.22 * 75)

        surface = clean_str(row.get("surface")) or race_surface
        row_distance = parse_distance_m(row.get("distance_m"))
        distance = (
            float(row_distance)
            if math.isfinite(row_distance) and row_distance > 0
            else float(race_distance)
        )

        course_index = course_index_for_horse(
            ctx["history"], race_id, surface, distance, proxy
        )
        today_index = clamp_index(0.50 * pace_index + 0.50 * course_index)
        total_internal = 0.60 * recent_index + 0.40 * today_index
        total_display = int(round(clamp_index(total_internal)))

        internal_totals[no] = float(total_internal)
        horses.append({
            "no": no,
            "name": clean_str(row.get("horse_name")),
            "recent": recent_strings,
            "recentIndex": int(round(recent_index)),
            "pace": int(round(pace_index)),
            "course": int(round(course_index)),
            "today": int(round(today_index)),
            "total": total_display,
        })

    # Rank uses unrounded internal totals, then current pre-race proxy, then horse number.
    ordered = sorted(
        horses,
        key=lambda h: (
            -internal_totals[h["no"]],
            -horse_context[h["no"]]["proxy"],
            h["no"],
        ),
    )
    rank_map = {h["no"]: i + 1 for i, h in enumerate(ordered)}
    for h in horses:
        h["rank"] = rank_map[h["no"]]

    venue = TRACKS.get(race_id[4:6], race_id[4:6])
    race_no = int(race_id[-2:])
    race_name = clean_str(card2.get("race_name", pd.Series([""])).iloc[0])
    title = f"{venue}{race_no}R" + (f" {race_name}" if race_name else "")

    detail = {
        "title": title,
        "horseCount": len(horses),
        "paceRegime": pace_regime,
        "raceConditions": {
            "surface": race_surface or None,
            "distanceM": int(race_distance) if math.isfinite(race_distance) else None,
        },
        "qualityWarnings": quality_warnings,
        "horses": horses,
    }
    total_display_map = {h["no"]: int(h["total"]) for h in horses}
    tie_order = {h["no"]: float(h["rank"]) for h in horses}
    return detail, internal_totals, total_display_map, tie_order


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
    tie_order: dict[int, float]
    popularity_features: pd.DataFrame
    actual_popularity: dict[int, int]
    index_detail: dict



def build_fallback_index_detail(
    race_id: str,
    card: pd.DataFrame,
    pred: pd.DataFrame,
    reason: str,
) -> tuple[dict, dict[int, float], dict[int, int], dict[int, float]]:
    """Last-resort modal data from validated pre-race model scores only.

    This prevents a non-critical display-index issue from blocking all 72 races.
    The fallback is explicitly marked in qualityWarnings and never uses target results.
    """
    card2 = card.copy()
    card2["horse_number"] = pd.to_numeric(
        card2["horse_number"], errors="coerce"
    )
    card2 = card2[card2["horse_number"].notna()].copy()
    card2["horse_number"] = card2["horse_number"].astype(int)

    p = pred.copy()
    p["horse_number"] = pd.to_numeric(p["horse_number"], errors="coerce")
    p = p[p["horse_number"].notna()].copy()
    p["horse_number"] = p["horse_number"].astype(int)
    if "rule_norm" not in p.columns:
        p["rule_norm"] = minmax(p.get("score", pd.Series(0.5, index=p.index)))
    if "ml_norm" not in p.columns:
        p["ml_norm"] = minmax(p.get("ml_win_prob", pd.Series(0.5, index=p.index)))
    p["fallback_raw"] = 0.65 * p["rule_norm"] + 0.35 * p["ml_norm"]
    pmap = p.set_index("horse_number")

    horses = []
    totals = {}
    for _, row in card2.iterrows():
        no = int(row["horse_number"])
        if no in pmap.index:
            raw = float(pmap.loc[no, "fallback_raw"])
        else:
            raw = 0.5
        total = clamp_index(55 + 40 * raw)
        recent_i = clamp_index(53 + 42 * raw)
        pace_i = clamp_index(58 + 34 * raw)
        course_i = clamp_index(0.88 * total + 0.12 * 72)
        today_i = clamp_index(0.50 * pace_i + 0.50 * course_i)
        internal = 0.60 * recent_i + 0.40 * today_i
        totals[no] = float(internal)
        horses.append({
            "no": no,
            "name": clean_str(row.get("horse_name")),
            "recent": ["評価外"] * 5,
            "recentIndex": int(round(recent_i)),
            "pace": int(round(pace_i)),
            "course": int(round(course_i)),
            "today": int(round(today_i)),
            "total": int(round(clamp_index(internal))),
        })

    ordered = sorted(horses, key=lambda h: (-totals[h["no"]], h["no"]))
    rank_map = {h["no"]: i + 1 for i, h in enumerate(ordered)}
    for h in horses:
        h["rank"] = rank_map[h["no"]]

    venue = TRACKS.get(race_id[4:6], race_id[4:6])
    race_no = int(race_id[-2:])
    race_name = clean_str(card2.get("race_name", pd.Series([""])).iloc[0]) if len(card2) else ""
    detail = {
        "title": f"{venue}{race_no}R" + (f" {race_name}" if race_name else ""),
        "horseCount": len(horses),
        "paceRegime": "unknown",
        "raceConditions": {"surface": None, "distanceM": None},
        "qualityWarnings": [f"fallback index detail used: {reason}"],
        "horses": horses,
    }
    total_display = {h["no"]: h["total"] for h in horses}
    tie_order = {h["no"]: float(h["rank"]) for h in horses}
    return detail, totals, total_display, tie_order


def validate_no_market_input(card: pd.DataFrame, pred: pd.DataFrame, race_id: str) -> None:
    """Fail if the archived prediction snapshot contains current odds/bodyweight.

    We deliberately do not merely drop these values because the precomputed score could
    already have incorporated them. A contaminated snapshot must not be used.
    """
    for label, df in (("card", card), ("prediction", pred)):
        if "win_odds" in df.columns:
            odds = pd.to_numeric(df["win_odds"], errors="coerce")
            if (odds > 0).any():
                bad = df.loc[odds > 0, ["horse_number", "win_odds"]].to_dict("records")
                raise ValueError(f"{race_id} {label}: numeric current odds found: {bad[:4]}")
        if "horse_weight" in df.columns:
            nonblank = df["horse_weight"].apply(clean_str)
            if nonblank.ne("").any():
                bad = df.loc[nonblank.ne(""), ["horse_number", "horse_weight"]].to_dict("records")
                raise ValueError(f"{race_id} {label}: current horse weight found: {bad[:4]}")


def build_race_model(source_root: Path, date_s: str, pred_path: Path, history: pd.DataFrame) -> RaceModel:
    race_id = pred_path.stem
    card_path = source_root / "data" / "race_cards" / date_s.replace("-", "") / f"{race_id}.csv"
    result_path = source_root / "data" / "race_results" / "2026" / f"{race_id}.csv"
    payout_path = source_root / "data" / "race_payouts" / f"{race_id}.csv"

    card = read_csv(card_path)
    pred = read_csv(pred_path)
    result = read_csv(result_path)
    payout = read_csv(payout_path)

    validate_no_market_input(card, pred, race_id)

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

    required_pred = {"horse_number", "score", "ml_win_prob", "predicted_rank", "ml_rank"}
    missing = required_pred - set(pred.columns)
    if missing:
        raise ValueError(f"{race_id}: prediction missing columns {sorted(missing)}")

    p = pred.copy()
    p["horse_number"] = pd.to_numeric(p["horse_number"], errors="raise").astype(int)
    p["rule_norm"] = minmax(p["score"])
    p["ml_norm"] = minmax(p["ml_win_prob"])

    try:
        index_detail, total_internal, total_display, tie_order = build_index_detail(
            race_id, date_s, card, p, result, history
        )
    except Exception as exc:
        index_detail, total_internal, total_display, tie_order = build_fallback_index_detail(
            race_id, card, p, f"{type(exc).__name__}: {exc}"
        )

    # 札幌記念は既に個別検証済みの詳細指数があるため、選定用の総合値は
    # その手作業レビュー値を優先する。画面も app.js の詳細表を優先表示する。
    if race_id == SAPPORO11_ID and set(card_nums) == set(SAPPORO11_TOTAL):
        total_internal = {n: float(v) for n, v in SAPPORO11_TOTAL.items()}
        total_display = dict(SAPPORO11_TOTAL)
        tie_order = {n: float(v) for n, v in SAPPORO11_RANK.items()}

    # Popularity model v2: separate "ability" from "how the market tends to buy".
    # Target-race actual popularity/odds/bodyweight never enter X.
    n = len(p)
    denom = max(n - 1, 1)

    # Current total/recent rank strengths are reproducible by the live engine.
    total_order = sorted(
        total_internal,
        key=lambda no: (-total_internal[no], tie_order.get(no, 999), no),
    )
    total_rank_strength = {
        no: 1.0 - (rank - 1) / denom
        for rank, no in enumerate(total_order, start=1)
    }
    recent_map = {
        int(h["no"]): float(h.get("recentIndex", 72))
        for h in index_detail.get("horses", [])
    }
    recent_order = sorted(
        total_internal,
        key=lambda no: (-recent_map.get(no, 72.0), -total_internal[no], no),
    )
    recent_rank_strength = {
        no: 1.0 - (rank - 1) / denom
        for rank, no in enumerate(recent_order, start=1)
    }

    target_date = pd.Timestamp(date_s)
    market_rows = []
    for _, row in p.iterrows():
        no = int(row["horse_number"])
        horse_id = clean_str(row.get("horse_id"))
        h = history[
            (history["horse_id"] == horse_id)
            & history["_date"].notna()
            & (history["_date"] < target_date)
        ].sort_values("_date", ascending=False)

        hp = h[h["_market_strength"].notna()].head(5)
        market_values = hp["_market_strength"].astype(float).tolist()
        last_market = market_values[0] if market_values else 0.5
        recent3_market = market_recency(market_values, 3)
        recent5_market = market_recency(market_values, 5)

        if not hp.empty:
            last_hist = hp.iloc[0]
            last_finish = (
                float(last_hist["_finish_strength"])
                if pd.notna(last_hist.get("_finish_strength"))
                else 0.5
            )
            last_weight = (
                float(last_hist["_weight_carried_num"])
                if pd.notna(last_hist.get("_weight_carried_num"))
                else math.nan
            )
            last_pop = (
                float(last_hist["_popularity"])
                if pd.notna(last_hist.get("_popularity"))
                else math.nan
            )
            last_field = (
                float(last_hist["_field_size"])
                if pd.notna(last_hist.get("_field_size"))
                else math.nan
            )
            last_finish_pos = (
                float(last_hist["_finish"])
                if pd.notna(last_hist.get("_finish"))
                else math.nan
            )
        else:
            last_finish = 0.5
            last_weight = math.nan
            last_pop = math.nan
            last_field = math.nan
            last_finish_pos = math.nan

        surprise_strength = clipped01(0.5 + (last_finish - last_market) / 2.0)

        lowpop_threshold = (
            max(6.0, math.ceil(last_field / 2.0))
            if math.isfinite(last_field)
            else 7.0
        )
        last_lowpop_win = float(
            math.isfinite(last_finish_pos)
            and int(last_finish_pos) == 1
            and math.isfinite(last_pop)
            and last_pop >= lowpop_threshold
        )

        current_weight = numeric_or_nan(row.get("weight_carried"))
        if math.isfinite(current_weight) and math.isfinite(last_weight):
            weight_delta = current_weight - last_weight
            carried_change_strength = clipped01(0.5 - weight_delta / 12.0)
            handicap_rebound_risk = clipped01(
                last_lowpop_win * max(weight_delta - 1.0, 0.0) / 5.0
            )
        else:
            carried_change_strength = 0.5
            handicap_rebound_risk = 0.0

        jockey_id = normalize_entity_id(row.get("jockey_id"))
        trainer_id = normalize_entity_id(row.get("trainer_id"))
        jockey_name = clean_str(row.get("jockey"))
        trainer_name = clean_str(row.get("trainer"))
        eligible_hist = history[
            history["_date"].notna()
            & (history["_date"] < target_date)
            & history["_market_strength"].notna()
        ]
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

        age = parse_age(row.get("sex_age"))
        age_strength = clipped01((10.0 - age) / 8.0)

        market_rows.append({
            "horse_number": no,
            "total_rank_strength": clipped01(total_rank_strength.get(no, 0.5)),
            "recent_rank_strength": clipped01(recent_rank_strength.get(no, 0.5)),
            "last_market_strength": clipped01(last_market),
            "recent3_market_strength": clipped01(recent3_market),
            "recent5_market_strength": clipped01(recent5_market),
            "last_finish_strength": clipped01(last_finish),
            "surprise_strength": clipped01(surprise_strength),
            "jockey_market_strength": shrunk_market_mean(
                jockey_values, prior=0.5, prior_weight=20.0
            ),
            "trainer_market_strength": shrunk_market_mean(
                trainer_values, prior=0.5, prior_weight=30.0
            ),
            "age_strength": age_strength,
            "carried_change_strength": clipped01(carried_change_strength),
            "last_lowpop_win": last_lowpop_win,
            "handicap_rebound_risk": clipped01(handicap_rebound_risk),
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
        tie_order=tie_order,
        popularity_features=features,
        actual_popularity=actual_popularity,
        index_detail=index_detail,
    )


FEATURE_COLS = [
    "total_rank_strength",
    "recent_rank_strength",
    "last_market_strength",
    "recent3_market_strength",
    "recent5_market_strength",
    "last_finish_strength",
    "surprise_strength",
    "jockey_market_strength",
    "trainer_market_strength",
    "age_strength",
    "carried_change_strength",
    "last_lowpop_win",
    "handicap_rebound_risk",
]


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
        # Stable deterministic tie break: performance proxy, then horse number.
        feats["_total"] = feats["horse_number"].map(race.total_internal)
        feats = feats.sort_values(
            ["_pred_market", "_total", "horse_number"],
            ascending=[False, False, True],
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

    # Final coefficients for future generalization (trained on all 72 races).
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


def build_future_entity_priors(history: pd.DataFrame) -> dict:
    """Market priors for current jockey/trainer, trained only on past race popularity."""
    h = history[
        history["_market_strength"].notna()
        & history["_date"].notna()
        & (history["_date"] <= pd.Timestamp(max(TARGET_DATES)))
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


def prediction_target_count(field_size: int) -> int:
    return min((field_size + 1) // 2, 7)


def build_prediction(race: RaceModel, estimated_popularity: dict[int, int]) -> tuple[dict, int]:
    horses = sorted(race.total_internal)

    # 危険: 想定1〜3番人気のうち総合評価最下位.
    top3 = [h for h in horses if estimated_popularity[h] <= 3]
    if len(top3) != 3:
        raise ValueError(f"{race.race_id}: estimated top3 count={len(top3)}")
    danger = min(
        top3,
        key=lambda h: (
            race.total_internal[h],
            -race.tie_order.get(h, 999),
            -h,
        ),
    )

    target_n = prediction_target_count(len(horses))
    eligible = [h for h in horses if h != danger]
    selected = sorted(
        eligible,
        key=lambda h: (
            -race.total_internal[h],
            race.tie_order.get(h, 999),
            h,
        ),
    )[:target_n]
    if len(selected) != target_n:
        raise ValueError(f"{race.race_id}: selected {len(selected)} != {target_n}")

    main = selected[0]
    # 対抗: selection target with the lowest estimated popularity (= largest rank).
    second = max(
        (h for h in selected if h != main),
        key=lambda h: (estimated_popularity[h], race.total_internal[h], -h),
    )
    opponents = [
        h for h in selected if h not in {main, second}
    ]
    opponents.sort(
        key=lambda h: (-race.total_internal[h], race.tie_order.get(h, 999), h)
    )

    return {"axes": [main, second], "opponents": opponents}, danger


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--data-path", default="data/races.json", type=Path)
    parser.add_argument("--audit-path", default="data/rebuild_audit_20260815_16.json", type=Path)
    parser.add_argument("--pop-model-path", default="data/popularity_model_20260815_16.json", type=Path)
    parser.add_argument("--diagnostic-path", default=DIAGNOSTIC_PATH_DEFAULT, type=Path)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    if not source_root.exists():
        raise FileNotFoundError(source_root)
    if not args.data_path.exists():
        raise FileNotFoundError(args.data_path)

    source_sha = source_commit(source_root)
    races: dict[str, RaceModel] = {}
    target_files: list[tuple[str, Path]] = []
    target_horse_ids: set[str] = set()

    for date_s in TARGET_DATES:
        pred_dir = source_root / "data" / "predictions" / date_s.replace("-", "")
        files = sorted(pred_dir.glob("*.csv"))
        if len(files) != EXPECTED_RACES_PER_DAY:
            raise RuntimeError(
                f"{date_s}: expected {EXPECTED_RACES_PER_DAY} prediction files, got {len(files)}"
            )
        for f in files:
            target_files.append((date_s, f))
            card_path = (
                source_root / "data" / "race_cards" / date_s.replace("-", "")
                / f"{f.stem}.csv"
            )
            card = read_csv(card_path)
            if "horse_id" in card.columns:
                target_horse_ids.update(card["horse_id"].apply(clean_str))

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

    if build_errors or len(races) != EXPECTED_TOTAL_RACES:
        diagnostic = {
            "generatedAt": datetime.now(JST).isoformat(timespec="seconds"),
            "stage": "race-preflight",
            "expectedRaces": EXPECTED_TOTAL_RACES,
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
            f"({len(races)}/{EXPECTED_TOTAL_RACES} built).\n{summary}"
        )

    estimated_pops, pop_metrics = estimate_popularities(races)
    future_entity_priors = build_future_entity_priors(history)

    data = json.loads(args.data_path.read_text(encoding="utf-8"))
    existing_by_date = {
        d.get("date"): d for d in data.get("days", [])
    }

    audit_races = []
    daily = {}
    old_result_mismatches = []

    for date_s in TARGET_DATES:
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
            prediction, danger = build_prediction(race, est_pop)
            result, tri_payouts = build_result(race)

            old = old_by_id.get(rid, {})
            if old.get("result") and normalize_old_result(old) != normalize_new_result(result):
                old_result_mismatches.append(rid)

            winners = [
                t for t in result["trifectas"]
                if covered(prediction, t["horses"])
            ]
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
                # For Sapporo11 the frontend uses the separately reviewed hardcoded detail.
                if (
                    rid == SAPPORO11_ID
                    and set(index_horses) == set(SAPPORO11_TOTAL)
                    and no in SAPPORO11_TOTAL
                ):
                    h["total"] = int(SAPPORO11_TOTAL[no])
                    h["rank"] = int(SAPPORO11_RANK[no])
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
                    "version": REBUILD_VERSION,
                    "estimatedPopularity": {str(k): int(v) for k, v in est_pop.items()},
                    "totalIndex": {str(k): int(v) for k, v in race.total_display.items()},
                    "indexDetail": index_detail,
                    "performanceSource": (
                        "detailed-index-pilot"
                        if rid == SAPPORO11_ID
                        else "pre-race historical reconstruction"
                    ),
                    "popularityMethod": "market-memory v2 leave-one-race-out calibrated model",
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
                h for h, _ in sorted(est_pop.items(), key=lambda kv: kv[1])
                if h in actual_pop
            ]
            est_pop_starters = {h: i + 1 for i, h in enumerate(est_order)}
            mae = float(np.mean([
                abs(est_pop_starters[h] - actual_pop[h])
                for h in est_pop_starters
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

        if len(rebuilt) != EXPECTED_RACES_PER_DAY:
            raise RuntimeError(f"{date_s}: rebuilt {len(rebuilt)} races")

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
            key=lambda r: (
                TRACK_ORDER.get(r.get("venue", ""), 99),
                int(r.get("raceNo", 99)),
            ),
        )

    # Final hard validations before writing.
    for date_s in TARGET_DATES:
        day = next(d for d in data["days"] if d.get("date") == date_s)
        if len(day["races"]) != EXPECTED_RACES_PER_DAY:
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
                    f"{r['raceId']}: indexDetail horse count "
                    f"{len(horses)} != {r['horseCount']}"
                )
            for h in horses:
                if len(h.get("recent", [])) != 5:
                    raise RuntimeError(f"{r['raceId']}: recent index count invalid")
                for key in ("recentIndex", "pace", "course", "today", "total", "rank"):
                    if key not in h:
                        raise RuntimeError(f"{r['raceId']}: index field {key} missing")

    audit = {
        "version": REBUILD_VERSION,
        "generatedAt": datetime.now(JST).isoformat(timespec="seconds"),
        "sourceRepository": SOURCE_REPO,
        "sourceRef": SOURCE_REF,
        "sourceCommit": source_sha,
        "strictInputRules": {
            "currentOddsUsed": False,
            "currentPopularityUsedAsPredictionFeature": False,
            "horseWeightOrChangeUsed": False,
            "raceResultUsedAsPerformanceFeature": False,
            "actualPopularityUse": (
                "teacher label only; each race excluded from its own estimator training"
            ),
            "nonStarterHandling": (
                "cancelled/excluded horses are retained in pre-race prediction candidates "
                "but omitted from final-popularity teacher rows and validation metrics"
            ),
            "indexDetailHistoryCutoff": (
                "previous-run/detail indices use only archived race results with date < target race date"
            ),
            "raceConditionMetadataFallback": (
                "surface/distance may fall back to target result metadata or an authoritative "
                "race-program override; if still missing, course index degrades neutrally "
                "instead of aborting. Target result performance is never used"
            ),
            "indexDetailFailurePolicy": (
                "non-critical modal-index exceptions fall back to validated pre-race scores; "
                "critical race input errors are collected across all races and reported together"
            ),
        },
        "raceCount": len(audit_races),
        "oldResultMismatchCount": len(old_result_mismatches),
        "oldResultMismatches": old_result_mismatches,
        "popularityModel": pop_metrics,
        "daily": daily,
        "races": audit_races,
    }

    pop_model = {
        "version": "predictjra-popularity-calibration-20260815-16-v2-market-memory",
        "trainedAt": datetime.now(JST).isoformat(timespec="seconds"),
        "teacherDates": list(TARGET_DATES),
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
            "historical trainer market tendency"
        ],
        "prohibitedInputs": [
            "current odds", "current actual popularity", "horse bodyweight", "horse bodyweight change"
        ],
    }

    args.data_path.parent.mkdir(parents=True, exist_ok=True)
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
            "expectedRaces": EXPECTED_TOTAL_RACES,
            "builtRaces": len(audit_races),
            "errorCount": 0,
            "warnings": [
                {
                    "raceId": r["raceId"],
                    "qualityWarnings": r.get("modelMeta", {}).get("indexDetail", {}).get("qualityWarnings", []),
                }
                for d in data.get("days", [])
                if d.get("date") in TARGET_DATES
                for r in d.get("races", [])
                if r.get("modelMeta", {}).get("indexDetail", {}).get("qualityWarnings")
            ],
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "raceCount": len(audit_races),
        "daily": daily,
        "oldResultMismatchCount": len(old_result_mismatches),
        "popularityMAE": pop_metrics["meanAbsolutePopularityRankError"],
        "top3Overlap": pop_metrics["meanTop3OverlapRate"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
