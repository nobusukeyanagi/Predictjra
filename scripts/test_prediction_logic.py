#!/usr/bin/env python3
"""Validate the candidate logic and candidate/production isolation contract."""
from __future__ import annotations

from pathlib import Path

from prediction_logic_candidate import (
    MODEL_VERSION,
    SELECTION_RULE_TEXT,
    build_index_core,
    prediction_target_count,
    select_prediction,
)


def sample_run(*, finish: int, field: int, popularity: int, venue: str = "札幌") -> dict:
    return {
        "date": "2026-07-01",
        "venue": venue,
        "finish": finish,
        "field": field,
        "popularity": popularity,
        "classLevel": 3,
        "carriedWeight": 57.0,
        "surface": "芝",
        "distance": 2000.0,
        "positions": [4, 4, 3, 2],
        "margin": max(0.0, (finish - 1) * 0.2),
    }


def main() -> None:
    assert MODEL_VERSION
    assert prediction_target_count(5) == 3
    assert prediction_target_count(6) == 3
    assert prediction_target_count(13) == 7
    assert prediction_target_count(18) == 7

    entries = []
    for no in range(1, 9):
        entries.append({
            "no": no,
            "name": f"馬{no}",
            "histories": [sample_run(finish=min(no, 8), field=8, popularity=min(no, 8))],
            "age": 4,
        })
    core = build_index_core(entries, venue="札幌", surface="芝", distance_m=2000)
    assert len(core["horses"]) == 8
    assert set(core["totals"]) == set(range(1, 9))
    assert all(len(h["recent"]) == 5 for h in core["horses"])

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

    # Critical safety boundary: live uses production, Rebuild uses candidate.
    assert "from prediction_logic_production import" in live_text
    assert "from prediction_logic_candidate import" in rebuild_text
    assert "from prediction_logic_production import *" in compat_text
    assert "prediction_logic_candidate" not in live_text

    # No duplicate selection/index implementation is allowed in adapters.
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

    print("Predictjra candidate prediction-logic + isolation tests: OK")


if __name__ == "__main__":
    main()
