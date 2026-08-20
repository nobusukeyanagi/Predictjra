#!/usr/bin/env python3
"""Smoke-test the applied production logic API without pinning tuneable behavior."""
from __future__ import annotations

from prediction_logic_production import (
    MODEL_VERSION,
    build_index_core,
    prediction_target_count,
    select_prediction,
)


def sample_run(no: int) -> dict:
    return {
        "date": "2026-07-01", "venue": "札幌", "finish": min(no, 8),
        "field": 8, "popularity": min(no, 8), "classLevel": 3,
        "carriedWeight": 57.0, "surface": "芝", "distance": 2000.0,
        "positions": [4, 4, 3, 2], "margin": max(0.0, (min(no, 8) - 1) * 0.2),
    }


def main() -> None:
    assert isinstance(MODEL_VERSION, str) and MODEL_VERSION
    assert 1 <= prediction_target_count(5) <= 5
    assert 1 <= prediction_target_count(18) <= 18

    entries = [
        {"no": no, "name": f"馬{no}", "histories": [sample_run(no)], "age": 4}
        for no in range(1, 9)
    ]
    core = build_index_core(entries, venue="札幌", surface="芝", distance_m=2000)
    assert len(core["horses"]) == 8
    assert set(core["totals"]) == set(range(1, 9))

    expected = {i: i for i in range(1, 9)}
    prediction, danger, target = select_prediction(core["horses"], core["totals"], expected)
    assert isinstance(target, int) and 1 <= target <= 8
    assert danger is None or danger in expected
    assert set(prediction) >= {"axes", "opponents", "excluded"}
    assert len(prediction["axes"]) == 2
    assert len(set(prediction["axes"])) == 2

    print("Predictjra production prediction-logic smoke test: OK")


if __name__ == "__main__":
    main()
