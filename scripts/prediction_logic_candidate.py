#!/usr/bin/env python3
"""Predictjra v3 candidate prediction rules: 走・展・力.

Candidate/production isolation remains unchanged:
- prediction_logic_candidate.py: editable experiment used by Rebuild/validate
- prediction_logic_production.py: applied snapshot used by live Update race data

A successful Rebuild/apply copies this candidate byte-for-byte over production.
All performance scores are 0-100 and use only facts available before the target race.
"""
from __future__ import annotations

import math
from statistics import mean
from typing import Iterable

MODEL_VERSION = "predictjra-live-index-v3-run-flow-power-v54-prior-odds-top3"
POPULARITY_MODEL_VERSION = "predictjra-popularity-v54-hgb-prior-odds"

# Leakage-safe Top3 classifier inputs. Every field is available when the draw is fixed.
# Current-race odds / actual popularity / bodyweight are intentionally absent.
FEATURE_COLS = [
    "recent_index",
    "total_index",
    "current_run",
    "current_flow",
    "current_power",
    "today_index",
    "last_market_strength",
    "recent3_market_strength",
    "recent5_market_strength",
    "market_trend_strength",
    "market_stability_strength",
    "surface_market_strength",
    "surface_distance_market_strength",
    "last_finish_strength",
    "recent_finish_strength",
    "last_odds_strength",
    "recent3_odds_strength",
    "recent5_odds_strength",
    "best5_odds_strength",
    "odds_trend_strength",
    "surface_odds_strength",
    "surface_distance_odds_strength",
    "history_count",
    "age_strength",
    "carried_change_strength",
    "class_fit_strength",
    "layoff_strength",
    "jockey_market_strength",
    "trainer_market_strength",
    "jockey_surface_market_strength",
    "trainer_surface_market_strength",
]


SELECTION_RULE_TEXT = (
    "danger=lowest total among estimated-popularity top3; "
    "exclude danger; select top min(ceil(field/2),7) total; "
    "main=top total; second=lowest estimated popularity among selected"
)

# Public scoring contract. These exact weights are also documented in the ! logic modal.
PER_RUN_WEIGHTS = {"run": 0.40, "flow": 0.25, "power": 0.35}
RECENCY_WEIGHTS = [0.35, 0.25, 0.18, 0.13, 0.09]
RECENT_TOTAL_WEIGHT = 0.55
CURRENT_TOTAL_WEIGHT = 0.45
CLASS_SCORES = {0: 45.0, 1: 55.0, 2: 64.0, 3: 73.0, 4: 82.0, 5: 88.0, 6: 94.0, 7: 100.0}


def clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return (lo + hi) / 2
    if not math.isfinite(x):
        return (lo + hi) / 2
    return max(lo, min(hi, x))


def clamp01(value: float, default: float = 0.5) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(x):
        return default
    return max(0.0, min(1.0, x))


def _float_or_nan(value) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return math.nan
    return x if math.isfinite(x) else math.nan


def _time_seconds(value) -> float:
    if value is None:
        return math.nan
    if isinstance(value, (int, float)):
        return _float_or_nan(value)
    s = str(value).strip()
    if not s:
        return math.nan
    try:
        if ":" in s:
            mins, secs = s.split(":", 1)
            return float(mins) * 60.0 + float(secs)
        return float(s)
    except (TypeError, ValueError):
        return math.nan


def parse_class_level(text: str) -> int:
    s = "".join(str(text or "").split()).upper()
    # Longer roman-numeral grade labels must be checked first.
    if any(x in s for x in ("GIII", "G3", "ＧⅢ", "Ｇ３")):
        return 5
    if any(x in s for x in ("GII", "G2", "ＧⅡ", "Ｇ２")):
        return 6
    if any(x in s for x in ("GI", "G1", "ＧⅠ", "Ｇ１")):
        return 7
    if "リステッド" in s or "(L)" in s or "（L）" in s or "OP" in s or "オープン" in s:
        return 4
    if "3勝" in s or "３勝" in s:
        return 3
    if "2勝" in s or "２勝" in s:
        return 2
    if "1勝" in s or "１勝" in s:
        return 1
    if "未勝利" in s or "新馬" in s:
        return 0
    return 3


def recency_weighted(values: Iterable[float]) -> float:
    vals = [float(x) for x in list(values)[:5] if math.isfinite(_float_or_nan(x))]
    if not vals:
        return 50.0
    weights = RECENCY_WEIGHTS[: len(vals)]
    sw = sum(weights)
    return sum(v * w for v, w in zip(vals, weights)) / sw


def _weighted_available(values: list[float], weights: list[float]) -> float:
    pairs = [
        (float(v), float(w)) for v, w in zip(values, weights)
        if math.isfinite(_float_or_nan(v)) and w > 0
    ]
    if not pairs:
        return 50.0
    sw = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / sw


def market_strength(popularity: int | None, field: int | None) -> float:
    if not isinstance(popularity, int) or not isinstance(field, int) or field <= 1:
        return 0.5
    return clamp01(1 - (popularity - 1) / (field - 1))


def odds_strength(odds) -> float:
    """Convert a historical win-odds price into a bounded support strength.

    1.0 is extremely short support, 0.0 is roughly 100x or longer.  This uses only
    odds from races strictly before the target date.
    """
    try:
        value = float(odds)
    except (TypeError, ValueError):
        return 0.5
    if not math.isfinite(value) or value <= 0:
        return 0.5
    return clamp01(1.0 - math.log10(max(value, 1.0)) / 2.0)


def market_recency(values: Iterable[float], limit: int = 5) -> float:
    vals = [clamp01(v) for v in list(values)[:limit]]
    if not vals:
        return 0.5
    weights = [0.40, 0.25, 0.16, 0.11, 0.08][: len(vals)]
    sw = sum(weights)
    return clamp01(sum(v * w for v, w in zip(vals, weights)) / sw)


def _surface_kind(surface: str) -> str:
    s = str(surface or "")
    if "障" in s:
        return "jump"
    if "ダ" in s:
        return "dirt"
    return "turf"


def _benchmark_sec_per_1000(surface: str, distance: float, track_condition: str = "") -> float:
    """Transparent distance/surface benchmark used by 走.

    This is intentionally a simple public formula rather than an opaque learned constant.
    Wet-track adjustments only normalize the clock; they never use the target race's
    unknown same-day result.
    """
    kind = _surface_kind(surface)
    d = distance if math.isfinite(_float_or_nan(distance)) and distance > 0 else 1600.0
    if kind == "dirt":
        base = 60.6 + 0.0018 * max(d - 1000.0, 0.0)
        adj = {"稍重": -0.2, "重": -0.4, "不良": -0.2}.get(str(track_condition or ""), 0.0)
    elif kind == "jump":
        base = 64.0 + 0.0007 * max(d - 2500.0, 0.0)
        adj = {"稍重": 0.3, "重": 0.7, "不良": 1.0}.get(str(track_condition or ""), 0.0)
    else:
        base = 58.4 + 0.0015 * max(d - 1000.0, 0.0)
        adj = {"稍重": 0.6, "重": 1.4, "不良": 2.4}.get(str(track_condition or ""), 0.0)
    return base + adj


def _time_baseline_key(venue: str, surface: str, distance: float, condition: str) -> str:
    d = int(round(distance)) if math.isfinite(_float_or_nan(distance)) and distance > 0 else 0
    return f"{venue or '*'}|{surface or '*'}|{d if d else '*'}|{condition or '*'}"


def _lookup_time_baseline(history: dict, time_baselines: dict | None) -> tuple[float, float, str]:
    """Return (median third-place sec/1000, robust sigma, source key).

    Exact/fallback groups are learned from completed races that were already known at the
    target prediction date.  When the table has too little history, use the fully public
    fixed benchmark as a deterministic fallback.
    """
    distance = _float_or_nan(history.get("distance"))
    venue = str(history.get("venue") or "")
    surface = str(history.get("surface") or "")
    condition = str(history.get("trackCondition") or "")
    groups = (time_baselines or {}).get("groups", time_baselines or {})
    if math.isfinite(distance) and distance > 0 and isinstance(groups, dict):
        keys = [
            _time_baseline_key(venue, surface, distance, condition),
            _time_baseline_key(venue, surface, distance, ""),
            _time_baseline_key("", surface, distance, condition),
            _time_baseline_key("", surface, distance, ""),
        ]
        # Prefer statistically thicker groups, but allow n>=3 before falling back.
        for minimum_n in (5, 3):
            for key in keys:
                row = groups.get(key) or {}
                n = int(row.get("n") or 0)
                median = _float_or_nan(row.get("medianSecPer1000"))
                sigma = _float_or_nan(row.get("sigmaSecPer1000"))
                if n >= minimum_n and math.isfinite(median) and math.isfinite(sigma) and sigma > 0:
                    return median, max(0.20, sigma), key

    benchmark = _benchmark_sec_per_1000(surface, distance, condition)
    # 走 = 50 + 15*z; sigma 1.25 gives the legacy-transparent fallback slope of 12/second.
    return benchmark, 1.25, "fixed-public-fallback"


def _finish_strength(finish: int | None, field: int | None) -> float:
    if not isinstance(finish, int) or not isinstance(field, int) or field <= 1:
        return 0.5
    return clamp01(1 - (finish - 1) / (field - 1))


def _front_strength(positions: list[int], field: int | None) -> float:
    if not isinstance(field, int) or field <= 1 or not positions:
        return math.nan
    valid = [int(x) for x in positions if isinstance(x, int) and 1 <= int(x) <= field]
    if not valid:
        return math.nan
    # Average the first two recorded corners where available: stable enough to describe style.
    sample = valid[:2]
    return clamp01(mean(1 - (p - 1) / max(field - 1, 1) for p in sample))


def run_indices(history: dict, time_baselines: dict | None = None) -> dict:
    """Score one historical race as 走・展・力, each 0-100."""
    finish = history.get("finish")
    field = history.get("field")
    finish_strength = _finish_strength(finish, field)

    # ---- 走: clock itself, normalized by surface/distance. ----
    raw_time = _float_or_nan(history.get("timeSeconds"))
    if not math.isfinite(raw_time):
        raw_time = _time_seconds(history.get("time"))
    third_time = _float_or_nan(history.get("thirdTimeSeconds"))
    margin = _float_or_nan(history.get("margin"))
    effective_time = raw_time
    capped_after_third = False
    if math.isfinite(raw_time) and isinstance(finish, int) and finish > 3:
        if math.isfinite(third_time):
            cap_time = third_time + 1.0
            if raw_time > cap_time:
                effective_time = cap_time
                capped_after_third = True
        elif math.isfinite(margin) and margin > 1.0:
            # Live past-5 cells do not always expose the 3rd-place clock. In that case,
            # remove loss beyond one second from the winner-gap proxy. Rebuild uses exact 3rd.
            effective_time = max(0.1, raw_time - (margin - 1.0))
            capped_after_third = True

    distance = _float_or_nan(history.get("distance"))
    baseline_median, baseline_sigma, baseline_source = _lookup_time_baseline(
        history, time_baselines
    )
    sec_per_1000 = math.nan
    z_score = 0.0
    if math.isfinite(effective_time) and math.isfinite(distance) and distance > 0:
        sec_per_1000 = effective_time * 1000.0 / distance
        z_score = (baseline_median - sec_per_1000) / max(baseline_sigma, 0.20)
        run_index = clamp(50.0 + 15.0 * z_score)
    else:
        run_index = 50.0

    # ---- 展: effort against race shape / position. ----
    positions = history.get("positions") or []
    front = _front_strength(positions, field)
    improvement = 0.0
    if isinstance(field, int) and field > 1 and positions and isinstance(finish, int):
        last_pos = int(positions[-1])
        improvement = max(-1.0, min(1.0, (last_pos - finish) / max(field - 1, 1)))

    race_bias = _float_or_nan(history.get("raceFrontBias"))
    if not math.isfinite(race_bias):
        race_bias = 0.0
    race_bias = max(-1.0, min(1.0, race_bias))

    if math.isfinite(front):
        style_axis = 2.0 * front - 1.0  # +1 front-runner, -1 closer
        benefit = race_bias * style_axis  # >0 benefited, <0 disadvantaged
        disadvantage = max(0.0, -benefit)
        advantage = max(0.0, benefit)
        flow_index = clamp(
            50.0
            + 30.0 * improvement
            + 30.0 * disadvantage * finish_strength
            + 15.0 * (finish_strength - 0.5)
            - 12.0 * advantage
        )
    else:
        flow_index = clamp(50.0 + 10.0 * (finish_strength - 0.5))

    # ---- 力: finish + race level, no clock/pace duplication. ----
    class_level = history.get("classLevel")
    try:
        class_level = int(class_level)
    except (TypeError, ValueError):
        class_level = 3
    class_score = CLASS_SCORES.get(max(0, min(7, class_level)), CLASS_SCORES[3])
    finish_score = 100.0 * finish_strength
    power_index = clamp(0.50 * class_score + 0.50 * finish_score)

    composite = clamp(
        PER_RUN_WEIGHTS["run"] * run_index
        + PER_RUN_WEIGHTS["flow"] * flow_index
        + PER_RUN_WEIGHTS["power"] * power_index
    )
    return {
        **history,
        "runIndex": run_index,
        "flowIndex": flow_index,
        "powerIndex": power_index,
        # Compatibility aliases: old UI/code can still render candidate-derived rows safely.
        "paceIndex": flow_index,
        "timeIndex": run_index,
        "resultIndex": power_index,
        "frontStrength": front,
        "composite": composite,
        "effectiveTimeSeconds": effective_time if math.isfinite(effective_time) else None,
        "secPer1000": sec_per_1000 if math.isfinite(sec_per_1000) else None,
        "timeBaselineMedian": baseline_median,
        "timeBaselineSigma": baseline_sigma,
        "timeBaselineSource": baseline_source,
        "timeZ": z_score,
        "timeCappedAfterThird": capped_after_third,
    }


def _projected_run_index(runs: list[dict], surface: str, distance: float | None) -> float:
    values: list[float] = []
    weights: list[float] = []
    valid_distance = math.isfinite(_float_or_nan(distance)) and float(distance) > 0
    target_surface = str(surface or "")
    for i, r in enumerate(runs[:5]):
        score = _float_or_nan(r.get("runIndex"))
        if not math.isfinite(score):
            continue
        run_surface = str(r.get("surface") or "")
        if target_surface and run_surface:
            surface_factor = 1.0 if run_surface == target_surface else 0.35
        else:
            surface_factor = 0.75
        run_distance = _float_or_nan(r.get("distance"))
        if valid_distance and math.isfinite(run_distance) and run_distance > 0:
            distance_factor = max(0.40, 1.0 - abs(run_distance - float(distance)) / 1200.0)
        else:
            distance_factor = 0.70
        values.append(score)
        weights.append(RECENCY_WEIGHTS[i] * surface_factor * distance_factor)
    if not values:
        return 50.0
    weighted = _weighted_available(values, weights)
    ceiling = max(values)
    return clamp(0.80 * weighted + 0.20 * ceiling)


def _course_performance(runs: list[dict], venue: str, surface: str, distance: float | None, fallback: float) -> float:
    valid_distance = math.isfinite(_float_or_nan(distance)) and float(distance) > 0

    def score(group: list[dict]) -> float | None:
        if not group:
            return None
        vals = [_float_or_nan(r.get("composite")) for r in group[:5]]
        vals = [v for v in vals if math.isfinite(v)]
        return recency_weighted(vals) if vals else None

    exact = [
        r for r in runs
        if r.get("venue") == venue
        and (not surface or r.get("surface") == surface)
        and (not valid_distance or (
            math.isfinite(_float_or_nan(r.get("distance")))
            and abs(float(r["distance"]) - float(distance)) <= 100
        ))
    ]
    got = score(exact)
    if got is not None:
        return got

    venue_runs = [
        r for r in runs
        if r.get("venue") == venue and (not surface or r.get("surface") == surface)
    ]
    got = score(venue_runs)
    if got is not None:
        return got

    if valid_distance:
        distance_runs = [
            r for r in runs
            if (not surface or r.get("surface") == surface)
            and math.isfinite(_float_or_nan(r.get("distance")))
            and abs(float(r["distance"]) - float(distance)) <= 200
        ]
        got = score(distance_runs)
        if got is not None:
            return got
    return fallback


def build_index_core(
    entries: list[dict],
    *,
    venue: str,
    surface: str,
    distance_m: float | None,
    time_baselines: dict | None = None,
) -> dict:
    """Build v3 走・展・力 scores from normalized pre-race facts."""
    if len(entries) < 5:
        raise ValueError(f"need at least 5 horses, got {len(entries)}")

    contexts: dict[int, dict] = {}
    front_type_count = 0

    for entry in entries:
        no = int(entry["no"])
        runs = [run_indices(h, time_baselines) for h in list(entry.get("histories", []))[:5]]
        composites = [r["composite"] for r in runs]
        recent = recency_weighted(composites) if composites else 50.0
        powers = [r["powerIndex"] for r in runs]
        base_power = recency_weighted(powers) if powers else 50.0
        fronts = [
            r["frontStrength"] for r in runs
            if math.isfinite(_float_or_nan(r.get("frontStrength")))
        ]
        front_ratio = mean(fronts) if fronts else math.nan
        if math.isfinite(front_ratio) and front_ratio >= 0.62:
            front_type_count += 1

        contexts[no] = {
            "entry": entry,
            "runs": runs,
            "frontRatio": front_ratio,
            "recent": recent,
            "basePower": base_power,
        }

    pace_regime = (
        "fast" if front_type_count >= 3 else "slow" if front_type_count <= 1 else "medium"
    )

    detail_horses: list[dict] = []
    totals: dict[int, float] = {}
    runs_by_no: dict[int, list[dict]] = {}

    for no in sorted(contexts):
        c = contexts[no]
        fr = c["frontRatio"]
        current_run = _projected_run_index(c["runs"], surface, distance_m)

        if math.isfinite(fr):
            if pace_regime == "fast":
                current_flow = clamp(35.0 + 65.0 * (1.0 - fr))
            elif pace_regime == "slow":
                current_flow = clamp(35.0 + 65.0 * fr)
            else:
                current_flow = clamp(50.0 + 20.0 * (1.0 - abs(fr - 0.5) * 2.0))
        else:
            current_flow = 50.0

        course_perf = _course_performance(
            c["runs"], venue, surface, distance_m, c["basePower"]
        )
        current_power = clamp(0.75 * c["basePower"] + 0.25 * course_perf)
        today = clamp(
            PER_RUN_WEIGHTS["run"] * current_run
            + PER_RUN_WEIGHTS["flow"] * current_flow
            + PER_RUN_WEIGHTS["power"] * current_power
        )
        total = clamp(RECENT_TOTAL_WEIGHT * c["recent"] + CURRENT_TOTAL_WEIGHT * today)
        totals[no] = total
        runs_by_no[no] = c["runs"]

        recent_strings = [
            f"{int(round(r['runIndex']))}/{int(round(r['flowIndex']))}/{int(round(r['powerIndex']))}"
            for r in c["runs"][:5]
        ]
        while len(recent_strings) < 5:
            recent_strings.append("評価外")

        detail_horses.append(
            {
                "no": no,
                "name": c["entry"].get("name", ""),
                "recent": recent_strings,
                "recentIndex": int(round(c["recent"])),
                "currentRun": int(round(current_run)),
                "currentFlow": int(round(current_flow)),
                "currentPower": int(round(current_power)),
                "todayParts": f"{int(round(current_run))}/{int(round(current_flow))}/{int(round(current_power))}",
                # Compatibility aliases until every historical page has been rebuilt.
                "pace": int(round(current_flow)),
                "course": int(round(current_power)),
                "today": int(round(today)),
                "total": int(round(total)),
            }
        )

    ordered = sorted(
        detail_horses,
        key=lambda h: (-totals[h["no"]], -h["recentIndex"], h["no"]),
    )
    rank_map = {h["no"]: i + 1 for i, h in enumerate(ordered)}
    for h in detail_horses:
        h["rank"] = rank_map[h["no"]]

    return {
        "paceRegime": pace_regime,
        "horses": detail_horses,
        "totals": totals,
        "runsByNo": runs_by_no,
    }


def rank_strengths(detail_horses: list[dict], totals: dict[int, float]) -> tuple[dict[int, float], dict[int, float]]:
    field_size = len(detail_horses)
    denom = max(field_size - 1, 1)
    total_order = sorted(
        detail_horses,
        key=lambda h: (-totals[h["no"]], -h["recentIndex"], h["no"]),
    )
    total_strength = {
        h["no"]: 1.0 - i / denom for i, h in enumerate(total_order)
    }
    recent_order = sorted(
        detail_horses,
        key=lambda h: (-h["recentIndex"], -totals[h["no"]], h["no"]),
    )
    recent_strength = {
        h["no"]: 1.0 - i / denom for i, h in enumerate(recent_order)
    }
    return total_strength, recent_strength


def _shrunk_recent_mean(values: list[float], prior: float, prior_weight: float) -> float:
    vals = [clamp01(v) for v in values if math.isfinite(_float_or_nan(v))]
    if not vals:
        return clamp01(prior)
    return clamp01((sum(vals) + clamp01(prior) * prior_weight) / (len(vals) + prior_weight))


def _parse_iso_day(value) -> int | None:
    s = str(value or "").strip()
    m = __import__("re").match(r"^(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})", s)
    if not m:
        return None
    try:
        from datetime import date
        d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        return d.toordinal()
    except ValueError:
        return None


def build_market_profile(
    runs: list[dict],
    *,
    total_rank_strength: float,
    recent_rank_strength: float,
    current_carried_weight: float | None,
    jockey_market_strength: float,
    trainer_market_strength: float,
    age: float | int | None,
    current_class_level: int,
    current_surface: str = "",
    current_distance: float | None = None,
    current_date: str = "",
    jockey_surface_market_strength: float | None = None,
    trainer_surface_market_strength: float | None = None,
) -> tuple[dict, dict]:
    """Build leakage-safe market-popularity features available before target-race start.

    v5 deliberately models *market memory* rather than today's odds.  It keeps the
    previous-popularity signal but adds stability/trend and target-condition similarity,
    which are portable to future races and do not require a race-specific override.
    """
    market_pairs = [
        (r, market_strength(r.get("popularity"), r.get("field")))
        for r in runs
        if isinstance(r.get("popularity"), int) and isinstance(r.get("field"), int)
    ]
    market_values = [v for _, v in market_pairs]
    last_market = market_values[0] if market_values else 0.5
    recent3_market = market_recency(market_values, 3)
    recent5_market = market_recency(market_values, 5)
    horse_market_mean = _shrunk_recent_mean(market_values, 0.5, 2.0)

    if len(market_values) >= 2:
        older = market_values[1:5]
        market_trend = clamp01(0.5 + (last_market - mean(older)) / 2.0)
        m = mean(market_values[:5])
        variance = mean((x - m) ** 2 for x in market_values[:5])
        market_stability = clamp01(1.0 - math.sqrt(max(0.0, variance)) / 0.45)
    else:
        market_trend = 0.5
        market_stability = 0.5

    odds_pairs = []
    for run in runs:
        raw_odds = _float_or_nan(run.get("odds"))
        if math.isfinite(raw_odds) and raw_odds > 0:
            odds_pairs.append((run, odds_strength(raw_odds)))
    odds_values = [value for _, value in odds_pairs]
    last_odds = odds_values[0] if odds_values else 0.5
    recent3_odds = market_recency(odds_values, 3) if odds_values else 0.5
    recent5_odds = market_recency(odds_values, 5) if odds_values else 0.5
    best5_odds = max(odds_values[:5]) if odds_values else 0.5
    if len(odds_values) >= 2:
        older_odds = odds_values[1:5]
        odds_trend = clamp01(0.5 + (last_odds - mean(older_odds)) / 2.0)
    else:
        odds_trend = 0.5
    history_count = clamp01(len(odds_values[:5]) / 5.0)

    current_kind = _surface_kind(current_surface) if current_surface else ""
    current_distance_num = _float_or_nan(current_distance)
    surface_values: list[float] = []
    distance_values: list[float] = []
    surface_distance_values: list[float] = []
    for run, strength in market_pairs:
        run_kind = _surface_kind(str(run.get("surface") or ""))
        run_distance = _float_or_nan(run.get("distance"))
        surface_match = bool(current_kind) and run_kind == current_kind
        distance_match = (
            math.isfinite(current_distance_num)
            and math.isfinite(run_distance)
            and abs(run_distance - current_distance_num) <= 300.0
        )
        if surface_match:
            surface_values.append(strength)
        if distance_match:
            distance_values.append(strength)
        if surface_match and distance_match:
            surface_distance_values.append(strength)

    surface_market = _shrunk_recent_mean(surface_values, recent5_market, 2.0)
    distance_market = _shrunk_recent_mean(distance_values, recent5_market, 2.0)
    surface_distance_market = _shrunk_recent_mean(
        surface_distance_values, recent5_market, 3.0
    )

    surface_odds_values: list[float] = []
    surface_distance_odds_values: list[float] = []
    for run, strength in odds_pairs:
        run_kind = _surface_kind(str(run.get("surface") or ""))
        run_distance = _float_or_nan(run.get("distance"))
        surface_match = bool(current_kind) and run_kind == current_kind
        distance_match = (
            math.isfinite(current_distance_num)
            and math.isfinite(run_distance)
            and abs(run_distance - current_distance_num) <= 300.0
        )
        if surface_match:
            surface_odds_values.append(strength)
        if surface_match and distance_match:
            surface_distance_odds_values.append(strength)
    surface_odds = _shrunk_recent_mean(surface_odds_values, recent5_odds, 2.0)
    surface_distance_odds = _shrunk_recent_mean(
        surface_distance_odds_values, recent5_odds, 3.0
    )

    last = runs[0] if runs else {}
    if (
        isinstance(last.get("finish"), int)
        and isinstance(last.get("field"), int)
        and last["field"] > 1
    ):
        last_finish = 1.0 - (last["finish"] - 1) / max(last["field"] - 1, 1)
    else:
        last_finish = 0.5
    finish_values = [
        _finish_strength(r.get("finish"), r.get("field"))
        for r in runs[:5]
        if isinstance(r.get("finish"), int) and isinstance(r.get("field"), int)
    ]
    recent_finish = market_recency(finish_values, 5) if finish_values else 0.5
    surprise_strength = clamp01(0.5 + (last_finish - last_market) / 2.0)

    last_pop = last.get("popularity")
    last_field = last.get("field")
    last_lowpop_win = float(
        isinstance(last.get("finish"), int)
        and last["finish"] == 1
        and isinstance(last_pop, int)
        and isinstance(last_field, int)
        and last_pop >= max(6, math.ceil(last_field / 2))
    )

    current_weight = _float_or_nan(current_carried_weight)
    last_weight = _float_or_nan(last.get("carriedWeight"))
    if math.isfinite(current_weight) and math.isfinite(last_weight):
        weight_delta = current_weight - last_weight
        carried_change_strength = clamp01(0.5 - weight_delta / 12.0)
        handicap_rebound_risk = clamp01(
            last_lowpop_win * max(weight_delta - 1.0, 0.0) / 5.0,
            default=0.0,
        )
    else:
        weight_delta = 0.0
        carried_change_strength = 0.5
        handicap_rebound_risk = 0.0

    age_strength = clamp01((10.0 - float(age or 5)) / 8.0)

    last_class_level = int(last.get("classLevel", current_class_level)) if last else current_class_level
    max_recent_class = max(
        [int(r.get("classLevel", 0)) for r in runs] or [current_class_level]
    )
    class_gap = max(int(current_class_level) - int(max_recent_class), 0)
    class_fit_strength = clamp01(1.0 - class_gap / 4.0)

    target_day = _parse_iso_day(current_date)
    last_day = _parse_iso_day(last.get("date")) if last else None
    if target_day is not None and last_day is not None and target_day > last_day:
        days = target_day - last_day
        # Market tends to discount both very short turnarounds and very long layoffs.
        if days < 7:
            layoff_strength = 0.38
        elif days <= 70:
            layoff_strength = 0.62
        elif days <= 140:
            layoff_strength = 0.52
        else:
            layoff_strength = 0.42
    else:
        days = 0
        layoff_strength = 0.5

    jockey_surface = clamp01(
        jockey_surface_market_strength
        if jockey_surface_market_strength is not None
        else jockey_market_strength
    )
    trainer_surface = clamp01(
        trainer_surface_market_strength
        if trainer_surface_market_strength is not None
        else trainer_market_strength
    )

    factors = {
        "total_rank_strength": clamp01(total_rank_strength),
        "recent_rank_strength": clamp01(recent_rank_strength),
        "last_market_strength": clamp01(last_market),
        "recent3_market_strength": clamp01(recent3_market),
        "recent5_market_strength": clamp01(recent5_market),
        "horse_market_mean_strength": clamp01(horse_market_mean),
        "market_trend_strength": clamp01(market_trend),
        "market_stability_strength": clamp01(market_stability),
        "surface_market_strength": clamp01(surface_market),
        "distance_market_strength": clamp01(distance_market),
        "surface_distance_market_strength": clamp01(surface_distance_market),
        "last_finish_strength": clamp01(last_finish),
        "recent_finish_strength": clamp01(recent_finish),
        "last_odds_strength": clamp01(last_odds),
        "recent3_odds_strength": clamp01(recent3_odds),
        "recent5_odds_strength": clamp01(recent5_odds),
        "best5_odds_strength": clamp01(best5_odds),
        "odds_trend_strength": clamp01(odds_trend),
        "surface_odds_strength": clamp01(surface_odds),
        "surface_distance_odds_strength": clamp01(surface_distance_odds),
        "history_count": clamp01(history_count),
        "surprise_strength": clamp01(surprise_strength),
        "jockey_market_strength": clamp01(jockey_market_strength),
        "trainer_market_strength": clamp01(trainer_market_strength),
        "jockey_surface_market_strength": jockey_surface,
        "trainer_surface_market_strength": trainer_surface,
        "age_strength": clamp01(age_strength),
        "carried_change_strength": clamp01(carried_change_strength),
        "class_fit_strength": clamp01(class_fit_strength),
        "layoff_strength": clamp01(layoff_strength),
        "last_lowpop_win": last_lowpop_win,
        "handicap_rebound_risk": clamp01(handicap_rebound_risk, default=0.0),
    }

    context = {
        "classLevel": int(current_class_level),
        "lastClassLevel": int(last_class_level),
        "maxRecentClassLevel": int(max_recent_class),
        "assignedWeightDelta": round(float(weight_delta), 1),
        "daysSinceLastRun": int(days),
        "surface": str(current_surface or ""),
        "distanceM": int(round(current_distance_num)) if math.isfinite(current_distance_num) else None,
    }
    return factors, context


def build_popularity_feature_row(horse: dict, factors: dict) -> dict:
    """Return the exact normalized feature vector used by v54 Top3 classifier."""
    return {
        "recent_index": clamp01(float(horse.get("recentIndex", 50)) / 100.0),
        "total_index": clamp01(float(horse.get("total", 50)) / 100.0),
        "current_run": clamp01(float(horse.get("currentRun", 50)) / 100.0),
        "current_flow": clamp01(float(horse.get("currentFlow", 50)) / 100.0),
        "current_power": clamp01(float(horse.get("currentPower", 50)) / 100.0),
        "today_index": clamp01(float(horse.get("today", 50)) / 100.0),
        **{key: clamp01(float(factors.get(key, 0.5))) for key in FEATURE_COLS if key not in {
            "recent_index", "total_index", "current_run", "current_flow", "current_power", "today_index"
        }},
    }


def fallback_top3_score(features: dict) -> float:
    """Cold-start / model-file fallback tuned only from pre-race information."""
    return float(
        0.35 * features.get("recent3_market_strength", 0.5)
        + 0.20 * features.get("last_market_strength", 0.5)
        + 0.15 * features.get("recent_index", 0.5)
        + 0.12 * features.get("total_index", 0.5)
        + 0.10 * features.get("recent3_odds_strength", 0.5)
        + 0.08 * features.get("jockey_market_strength", 0.5)
    )


def market_context_adjustment(factors: dict, context: dict) -> float:
    current_class_level = int(context.get("classLevel", 3))
    max_recent_class = int(context.get("maxRecentClassLevel", current_class_level))
    class_gap = max(current_class_level - max_recent_class, 0)
    class_readiness = clamp01(1.0 - class_gap / 4.0)
    return (
        0.035 * (class_readiness - 0.5)
        - 0.090 * clamp01(factors.get("handicap_rebound_risk", 0.0), default=0.0)
    )


def market_score_from_model(factors: dict, context: dict, model: dict) -> float:
    coeff = model.get("coefficients") or {}
    model_features = set(model.get("features") or [])
    if (
        str(model.get("version", "")).endswith("market-memory")
        and model_features.issubset(factors)
        and model_features
    ):
        score = float(coeff.get("intercept", 0.0))
        for key in model["features"]:
            score += float(coeff.get(key, 0.0)) * float(factors[key])
    else:
        score = (
            0.22 * factors["recent5_market_strength"]
            + 0.08 * factors["last_market_strength"]
            + 0.08 * factors["horse_market_mean_strength"]
            + 0.08 * factors["surface_distance_market_strength"]
            + 0.04 * factors["surface_market_strength"]
            + 0.03 * factors["distance_market_strength"]
            + 0.10 * factors["total_rank_strength"]
            + 0.05 * factors["recent_rank_strength"]
            + 0.08 * factors["jockey_market_strength"]
            + 0.05 * factors["trainer_market_strength"]
            + 0.04 * factors["jockey_surface_market_strength"]
            + 0.03 * factors["trainer_surface_market_strength"]
            + 0.04 * factors["recent_finish_strength"]
            + 0.03 * factors["market_trend_strength"]
            + 0.02 * factors["market_stability_strength"]
            + 0.02 * factors["class_fit_strength"]
            + 0.01 * factors["layoff_strength"]
        )
    return float(score + market_context_adjustment(factors, context))


def expected_popularity_from_scores(detail_horses: list[dict]) -> dict[int, int]:
    ordered = sorted(
        detail_horses,
        key=lambda h: (-h["_popScore"], -h["recentIndex"], -h["total"], h["no"]),
    )
    return {h["no"]: i + 1 for i, h in enumerate(ordered)}


def prediction_target_count(field_size: int) -> int:
    return min((int(field_size) + 1) // 2, 7)


def select_prediction(
    detail_horses: list[dict],
    totals: dict[int, float],
    expected_popularity: dict[int, int],
) -> tuple[dict, int, int]:
    """Select danger/main/second/opponents from already-derived horse scores."""
    if len(detail_horses) < 5:
        raise ValueError(f"need at least 5 horses, got {len(detail_horses)}")

    missing = [h["no"] for h in detail_horses if h["no"] not in expected_popularity]
    if missing:
        raise ValueError(f"expected popularity missing for horses: {missing}")

    top3 = [h for h in detail_horses if expected_popularity[h["no"]] <= 3]
    if len(top3) != 3:
        raise ValueError(f"estimated top3 count={len(top3)}")

    # Keep this tie order identical to the live rule: displayed total, recent index,
    # estimated-popularity rank, then horse number.
    danger = sorted(
        top3,
        key=lambda h: (
            h["total"],
            h["recentIndex"],
            -expected_popularity[h["no"]],
            -h["no"],
        ),
    )[0]

    target_count = prediction_target_count(len(detail_horses))
    selected = sorted(
        [h for h in detail_horses if h["no"] != danger["no"]],
        key=lambda h: (-totals[h["no"]], -h["recentIndex"], h["no"]),
    )[:target_count]
    if len(selected) != target_count:
        raise ValueError(f"selected {len(selected)} != {target_count}")

    main = selected[0]
    second = sorted(
        [h for h in selected if h["no"] != main["no"]],
        key=lambda h: (-expected_popularity[h["no"]], -totals[h["no"]], h["no"]),
    )[0]
    opponents = [
        h["no"]
        for h in sorted(
            [h for h in selected if h["no"] not in (main["no"], second["no"])],
            key=lambda h: (-totals[h["no"]], -h["recentIndex"], h["no"]),
        )
    ]

    prediction = {
        "axes": [main["no"], second["no"]],
        "opponents": opponents,
        "excluded": [danger["no"]],
    }
    return prediction, int(danger["no"]), target_count
