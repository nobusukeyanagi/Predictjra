#!/usr/bin/env python3
"""v91 bridge tests: D3 winMain is independent and result labels are delayed correctly."""
from __future__ import annotations

from single_win_runtime import RollingRebuildSingleWin


def sample_race(with_result: bool = False) -> dict:
    horses = []
    for no, total in [(1, 82), (2, 80), (3, 75), (4, 70), (5, 65)]:
        horses.append({
            "no": no,
            "recentIndex": total - 2,
            "currentRun": 70 + no,
            "currentFlow": 68 + no,
            "currentPower": 69 + no,
            "today": total - 1,
            "total": total,
            "rank": no,
            "expectedPopularity": no,
            "singleEV": 50 + no,
        })
    result = {}
    if with_result:
        result = {
            "places": [[2], [1], [3]],
            "winPayouts": [{"horses": [2], "payout": 420}],
            "trifectas": [{"horses": [2, 1, 3], "payout": 5000}],
        }
    return {
        "raceId": "202601010101",
        "prediction": {"axes": [1, 3], "opponents": [2, 4, 5]},
        "danger": [],
        "predictionDisabled": False,
        "result": result,
        "modelMeta": {
            "indexDetail": {
                "horses": horses,
                "raceConditions": {"surface": "芝", "distanceM": 1600},
            },
            "nonStarters": [],
        },
    }


def test_finish_day_adds_result_labels() -> None:
    selector = RollingRebuildSingleWin()
    selector.begin_day("2026-01-04")
    pre = sample_race(False)
    main, meta, rows = selector.decide("2026-01-04", pre)
    assert sum(int(r["is_winner"]) for r in rows) == 0
    final = sample_race(True)
    final["winMain"] = main
    final["modelMeta"]["singleWin"] = meta
    selector.finish_day("2026-01-04", [(final, meta, rows)])
    assert selector.training_rows
    assert sum(int(r["is_winner"]) for r in selector.training_rows) == 1
    assert sum(int(r["is_top3"]) for r in selector.training_rows) == 3
    assert "actionReturns" in meta


def test_trifecta_axes_are_not_mutated() -> None:
    selector = RollingRebuildSingleWin()
    selector.begin_day("2026-01-04")
    race = sample_race(False)
    before = list(race["prediction"]["axes"])
    main, _meta, _rows = selector.decide("2026-01-04", race)
    assert race["prediction"]["axes"] == before
    assert main in [1, 2, 3, 4, 5]


if __name__ == "__main__":
    tests = [test_finish_day_adds_result_labels, test_trifecta_axes_are_not_mutated]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"OK: {len(tests)} v91 single-win runtime tests passed")
