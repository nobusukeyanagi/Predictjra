#!/usr/bin/env python3
"""Unified Predictjra prediction-rule implementation.

The same implementation is stored in two snapshots:
- prediction_logic_candidate.py: editable experiment used by Rebuild/validate
- prediction_logic_production.py: applied snapshot used by live Update race data

A successful Rebuild/apply copies the candidate file byte-for-byte over the production
file and commits it. Keep source adapters free of duplicate prediction rules.
"""
from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Iterable

MODEL_VERSION = "predictjra-live-index-v2-market-memory"

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

SELECTION_RULE_TEXT = (
    "danger=lowest total among estimated-popularity top3; "
    "exclude danger; select top min(ceil(field/2),7) total; "
    "main=top total; second=lowest estimated popularity among selected"
)


def clamp(value: float, lo: float = 45, hi: float = 98) -> float:
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
    vals = [float(x) for x in list(values)[:5]]
    if not vals:
        return 72.0
    weights = [0.34, 0.25, 0.18, 0.13, 0.10][: len(vals)]
    total = sum(weights)
    return sum(v * w for v, w in zip(vals, weights)) / total


def market_strength(popularity: int | None, field: int | None) -> float:
    if not isinstance(popularity, int) or not isinstance(field, int) or field <= 1:
        return 0.5
    return clamp01(1 - (popularity - 1) / (field - 1))


def market_recency(values: Iterable[float], limit: int = 5) -> float:
    vals = [clamp01(v) for v in list(values)[:limit]]
    if not vals:
        return 0.5
    weights = [0.40, 0.25, 0.16, 0.11, 0.08][: len(vals)]
    sw = sum(weights)
    return clamp01(sum(v * w for v, w in zip(vals, weights)) / sw)


def run_indices(history: dict) -> dict:
    """Calculate the three per-run indices from normalized historical facts."""
    finish = history.get("finish")
    field = history.get("field")
    if not isinstance(finish, int) or not isinstance(field, int) or field <= 1:
        finish_strength = 0.5
    else:
        finish_strength = max(0.0, min(1.0, 1 - (finish - 1) / (field - 1)))

    result_index = clamp(48 + 50 * finish_strength)

    margin = _float_or_nan(history.get("margin"))
    if math.isfinite(margin):
        gap_score = clamp(96 - 6.0 * max(0.0, margin))
    else:
        gap_score = clamp(55 + 40 * finish_strength)
    last_score = clamp(55 + 40 * finish_strength)
    time_index = clamp(0.75 * gap_score + 0.25 * last_score)

    positions = history.get("positions") or []
    if isinstance(field, int) and field > 1 and positions:
        first = positions[0]
        last = positions[-1]
        front = max(0.0, min(1.0, 1 - (first - 1) / max(field - 1, 1)))
        improvement = (last - finish) / max(field, 1) if isinstance(finish, int) else 0.0
        pace_index = clamp(64 + 18 * finish_strength + 10 * max(-1, min(1, improvement)) + 4)
    else:
        front = math.nan
        pace_index = clamp(62 + 26 * finish_strength + 3.5)

    composite = 0.25 * pace_index + 0.35 * time_index + 0.40 * result_index
    return {
        **history,
        "paceIndex": pace_index,
        "timeIndex": time_index,
        "resultIndex": result_index,
        "frontStrength": front,
        "composite": composite,
    }


def _float_or_nan(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _course_score(runs: list[dict], cap: float) -> float | None:
    vals = [
        r["composite"]
        for r in runs[:5]
        if math.isfinite(_float_or_nan(r.get("composite")))
    ]
    if not vals:
        return None
    value = 0.72 * recency_weighted(vals) + 0.28 * max(vals)
    return min(cap, value)


def course_index(
    runs: list[dict],
    venue: str,
    surface: str,
    distance: float | None,
    proxy: float,
) -> float:
    """Current-course/distance suitability using the same hierarchy everywhere."""
    valid_distance = (
        distance is not None
        and math.isfinite(_float_or_nan(distance))
        and float(distance) > 0
    )
    exact = [
        r
        for r in runs
        if r.get("venue") == venue
        and (not surface or r.get("surface") == surface)
        and (
            not valid_distance
            or (
                math.isfinite(_float_or_nan(r.get("distance")))
                and abs(float(r["distance"]) - float(distance)) <= 100
            )
        )
    ]
    exact_score = _course_score(exact, 98)
    if exact_score is not None:
        return clamp(0.84 * exact_score + 0.16 * proxy)

    venue_runs = [
        r
        for r in runs
        if r.get("venue") == venue and (not surface or r.get("surface") == surface)
    ]
    venue_score = _course_score(venue_runs, 86)
    if venue_score is not None:
        return clamp(0.80 * venue_score + 0.20 * proxy)

    if valid_distance:
        dist_runs = [
            r
            for r in runs
            if (not surface or r.get("surface") == surface)
            and math.isfinite(_float_or_nan(r.get("distance")))
            and abs(float(r["distance"]) - float(distance)) <= 200
        ]
        dist_score = _course_score(dist_runs, 82)
        if dist_score is not None:
            return clamp(0.76 * dist_score + 0.24 * proxy)

    return clamp(min(78, 0.88 * proxy + 0.12 * 72))


def build_index_core(
    entries: list[dict],
    *,
    venue: str,
    surface: str,
    distance_m: float | None,
) -> dict:
    """Build all derived performance indices from normalized pre-race facts.

    Expected entry shape:
      {"no": int, "name": str, "histories": [normalized-run...], "age": optional}
    """
    if len(entries) < 5:
        raise ValueError(f"need at least 5 horses, got {len(entries)}")

    contexts: dict[int, dict] = {}
    front_type_count = 0

    for entry in entries:
        no = int(entry["no"])
        runs = [run_indices(h) for h in list(entry.get("histories", []))[:5]]
        composites = [r["composite"] for r in runs]
        fronts = [
            r["frontStrength"]
            for r in runs
            if math.isfinite(_float_or_nan(r.get("frontStrength")))
        ]
        front_ratio = mean(fronts) if fronts else math.nan
        if math.isfinite(front_ratio) and front_ratio >= 0.58:
            front_type_count += 1

        if composites:
            base = recency_weighted(composites)
            ceiling = max(composites)
            consistency = clamp(96 - 1.8 * pstdev(composites), 55, 96)
            hist_recent = 0.70 * base + 0.20 * ceiling + 0.10 * consistency
            history_weight = min(len(composites) / 5.0, 1.0) * 0.84
            recent = clamp(history_weight * hist_recent + (1 - history_weight) * 72)
        else:
            recent = 72.0

        contexts[no] = {
            "entry": entry,
            "runs": runs,
            "frontRatio": front_ratio,
            "recent": recent,
        }

    pace_regime = (
        "fast" if front_type_count >= 3 else "slow" if front_type_count <= 1 else "medium"
    )

    detail_horses: list[dict] = []
    totals: dict[int, float] = {}
    runs_by_no: dict[int, list[dict]] = {}

    for no in sorted(contexts):
        c = contexts[no]
        proxy = c["recent"]
        fr = c["frontRatio"]

        if math.isfinite(fr):
            if pace_regime == "fast":
                style = 70 + 20 * (1 - fr)
            elif pace_regime == "slow":
                style = 70 + 20 * fr
            else:
                style = 76 + 10 * (1 - abs(fr - 0.5) * 2)
            pace = clamp(0.78 * style + 0.22 * proxy)
        else:
            pace = clamp(0.78 * proxy + 0.22 * 75)

        course = course_index(c["runs"], venue, surface, distance_m, proxy)
        today = clamp(0.50 * pace + 0.50 * course)
        total = clamp(0.60 * c["recent"] + 0.40 * today)
        totals[no] = total
        runs_by_no[no] = c["runs"]

        recent_strings = [
            f"{int(round(r['paceIndex']))}/{int(round(r['timeIndex']))}/{int(round(r['resultIndex']))}"
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
                "pace": int(round(pace)),
                "course": int(round(course)),
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
) -> tuple[dict, dict]:
    """Build market-popularity features from facts available before the target race."""
    market_values = [
        market_strength(r.get("popularity"), r.get("field"))
        for r in runs
        if isinstance(r.get("popularity"), int) and isinstance(r.get("field"), int)
    ]
    last_market = market_values[0] if market_values else 0.5
    recent3_market = market_recency(market_values, 3)
    recent5_market = market_recency(market_values, 5)

    last = runs[0] if runs else {}
    if (
        isinstance(last.get("finish"), int)
        and isinstance(last.get("field"), int)
        and last["field"] > 1
    ):
        last_finish = 1.0 - (last["finish"] - 1) / max(last["field"] - 1, 1)
    else:
        last_finish = 0.5
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

    factors = {
        "total_rank_strength": clamp01(total_rank_strength),
        "recent_rank_strength": clamp01(recent_rank_strength),
        "last_market_strength": clamp01(last_market),
        "recent3_market_strength": clamp01(recent3_market),
        "recent5_market_strength": clamp01(recent5_market),
        "last_finish_strength": clamp01(last_finish),
        "surprise_strength": clamp01(surprise_strength),
        "jockey_market_strength": clamp01(jockey_market_strength),
        "trainer_market_strength": clamp01(trainer_market_strength),
        "age_strength": clamp01(age_strength),
        "carried_change_strength": clamp01(carried_change_strength),
        "last_lowpop_win": last_lowpop_win,
        "handicap_rebound_risk": clamp01(handicap_rebound_risk, default=0.0),
    }

    last_class_level = int(last.get("classLevel", current_class_level)) if last else current_class_level
    max_recent_class = max(
        [int(r.get("classLevel", 0)) for r in runs] or [current_class_level]
    )
    context = {
        "classLevel": int(current_class_level),
        "lastClassLevel": int(last_class_level),
        "maxRecentClassLevel": int(max_recent_class),
        "assignedWeightDelta": round(float(weight_delta), 1),
    }
    return factors, context


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
            0.36 * factors["recent5_market_strength"]
            + 0.14 * factors["last_market_strength"]
            + 0.16 * factors["total_rank_strength"]
            + 0.08 * factors["recent_rank_strength"]
            + 0.08 * factors["jockey_market_strength"]
            + 0.06 * factors["trainer_market_strength"]
            + 0.06 * factors["last_finish_strength"]
            + 0.06 * factors["age_strength"]
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
