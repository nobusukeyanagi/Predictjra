#!/usr/bin/env python3
"""Validate v3 candidate 走・展・力 rules and candidate/production isolation."""
from __future__ import annotations

from pathlib import Path

from prediction_logic_candidate import (
    FEATURE_COLS,
    MODEL_VERSION,
    PER_RUN_WEIGHTS,
    RECENCY_WEIGHTS,
    SELECTION_RULE_TEXT,
    build_index_core,
    build_market_profile,
    market_score_from_model,
    prediction_target_count,
    run_indices,
    select_prediction,
)


def sample_run(
    *,
    finish: int,
    field: int = 12,
    popularity: int = 4,
    class_level: int = 3,
    time_seconds: float = 120.0,
    third_time: float = 120.2,
    positions: list[int] | None = None,
    race_bias: float = 0.0,
    venue: str = "札幌",
) -> dict:
    return {
        "date": "2026-07-01",
        "venue": venue,
        "finish": finish,
        "field": field,
        "popularity": popularity,
        "classLevel": class_level,
        "carriedWeight": 57.0,
        "surface": "芝",
        "distance": 2000.0,
        "trackCondition": "良",
        "time": f"2:{time_seconds-120:04.1f}" if time_seconds >= 120 else str(time_seconds),
        "timeSeconds": time_seconds,
        "thirdTimeSeconds": third_time,
        "positions": positions or [8, 8, 6, 4],
        "raceFrontBias": race_bias,
        "margin": max(0.0, time_seconds - 119.5),
    }


def main() -> None:
    assert "v3-run-flow-power" in MODEL_VERSION
    assert PER_RUN_WEIGHTS == {"run": 0.40, "flow": 0.25, "power": 0.35}
    assert RECENCY_WEIGHTS == [0.35, 0.25, 0.18, 0.13, 0.09]
    assert prediction_target_count(5) == 3
    assert prediction_target_count(6) == 3
    assert prediction_target_count(13) == 7
    assert prediction_target_count(18) == 7

    # v5 popularity features remain leakage-safe: only previous runs and target
    # program conditions are needed. Same-condition memory must be explicit.
    profile, context = build_market_profile(
        [
            sample_run(finish=2, popularity=2, class_level=3),
            {**sample_run(finish=5, popularity=7, class_level=2), "surface": "ダート", "distance": 1600.0},
        ],
        total_rank_strength=0.8,
        recent_rank_strength=0.7,
        current_carried_weight=57.0,
        jockey_market_strength=0.65,
        trainer_market_strength=0.60,
        jockey_surface_market_strength=0.70,
        trainer_surface_market_strength=0.64,
        age=4,
        current_class_level=3,
        current_surface="芝",
        current_distance=2000.0,
        current_date="2026-07-18",
    )
    for key in (
        "horse_market_mean_strength", "market_trend_strength",
        "market_stability_strength", "surface_market_strength",
        "distance_market_strength", "surface_distance_market_strength",
        "jockey_surface_market_strength", "trainer_surface_market_strength",
        "class_fit_strength", "layoff_strength",
    ):
        assert key in FEATURE_COLS and 0 <= profile[key] <= 1, key
    assert context["distanceM"] == 2000
    assert isinstance(market_score_from_model(profile, context, {}), float)

    # Every per-run component is an explicit 0-100 score.
    baseline = {
        "groups": {
            "札幌|芝|2000|良": {
                "n": 10, "medianSecPer1000": 60.0, "sigmaSecPer1000": 0.5
            }
        }
    }
    one = run_indices(
        sample_run(finish=3, time_seconds=119.8, third_time=120.0), baseline
    )
    assert one["timeBaselineSource"] == "札幌|芝|2000|良"
    # 119.8 sec / 2000m = 59.9 sec/1000; Z=(60.0-59.9)/0.5=0.2 => 53.0.
    assert abs(one["runIndex"] - 53.0) < 1e-9, one["runIndex"]
    for key in ("runIndex", "flowIndex", "powerIndex", "composite"):
        assert 0 <= one[key] <= 100, (key, one[key])

    # 4th or worse: once >1.0 sec behind the 3rd horse, extra loss is ignored.
    capped_a = run_indices(sample_run(finish=8, time_seconds=122.0, third_time=120.0), baseline)
    capped_b = run_indices(sample_run(finish=8, time_seconds=130.0, third_time=120.0), baseline)
    assert capped_a["timeCappedAfterThird"] and capped_b["timeCappedAfterThird"]
    assert capped_a["effectiveTimeSeconds"] == capped_b["effectiveTimeSeconds"] == 121.0
    assert abs(capped_a["runIndex"] - capped_b["runIndex"]) < 1e-9

    # Same finish: higher race level must produce a higher 力 score.
    maiden = run_indices(sample_run(finish=3, class_level=0))
    g1 = run_indices(sample_run(finish=3, class_level=7))
    assert g1["powerIndex"] > maiden["powerIndex"]

    entries = []
    for no in range(1, 9):
        histories = [
            sample_run(
                finish=min(no + shift, 12),
                popularity=min(no, 8),
                time_seconds=118.5 + no * 0.35 + shift * 0.15,
                third_time=120.0 + shift * 0.10,
                positions=[max(1, no), max(1, no), max(1, no - 1), max(1, no - 2)],
                race_bias=(-0.25 if shift % 2 == 0 else 0.20),
            )
            for shift in range(5)
        ]
        entries.append({"no": no, "name": f"馬{no}", "histories": histories, "age": 4})

    core = build_index_core(entries, venue="札幌", surface="芝", distance_m=2000)
    assert len(core["horses"]) == 8
    assert set(core["totals"]) == set(range(1, 9))
    assert all(len(h["recent"]) == 5 for h in core["horses"])
    assert all(h["todayParts"].count("/") == 2 for h in core["horses"])
    assert all(0 <= h["recentIndex"] <= 100 for h in core["horses"])
    assert all(0 <= h["today"] <= 100 and 0 <= h["total"] <= 100 for h in core["horses"])

    horses = [
        {"no": 1, "total": 90, "recentIndex": 90},
        {"no": 2, "total": 82, "recentIndex": 82},
        {"no": 3, "total": 70, "recentIndex": 70},
        {"no": 4, "total": 88, "recentIndex": 88},
        {"no": 5, "total": 86, "recentIndex": 86},
        {"no": 6, "total": 84, "recentIndex": 84},
        {"no": 7, "total": 78, "recentIndex": 78},
        {"no": 8, "total": 76, "recentIndex": 76},
    ]
    totals = {h["no"]: float(h["total"]) for h in horses}
    expected = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8}
    prediction, danger, target = select_prediction(horses, totals, expected)
    assert danger == 3
    assert target == 4
    assert prediction == {"axes": [1, 6], "opponents": [4, 5], "excluded": [3]}

    scripts = Path(__file__).resolve().parent
    live_text = (scripts / "predict_engine.py").read_text(encoding="utf-8")
    rebuild_text = (scripts / "rebuild_history.py").read_text(encoding="utf-8")
    compat_text = (scripts / "prediction_logic.py").read_text(encoding="utf-8")

    # Critical safety boundary: live still uses production until apply.
    assert "from prediction_logic_production import" in live_text
    assert "from prediction_logic_candidate import" in rebuild_text
    assert "from prediction_logic_production import *" in compat_text
    assert "prediction_logic_candidate" not in live_text

    for forbidden in (
        "def prediction_target_count(",
        "def select_prediction(",
        "def run_indices(",
        "def course_index(",
    ):
        assert forbidden not in live_text, forbidden
        assert forbidden not in rebuild_text, forbidden

    assert SELECTION_RULE_TEXT in live_text or "SELECTION_RULE_TEXT" in live_text
    assert SELECTION_RULE_TEXT in rebuild_text or "SELECTION_RULE_TEXT" in rebuild_text

    print("Predictjra v3 candidate 走・展・力 + isolation tests: OK")


if __name__ == "__main__":
    main()
