#!/usr/bin/env python3
"""v106 bridge tests: cross-year winMain is leakage-safe and race-number agnostic."""
from __future__ import annotations

from single_win_runtime import (
    RollingRebuildSingleWin,
    apply_d3_field_size_guard,
    apply_d3_final_guard,
    apply_d3_v98_action_guard,
    apply_d3_v99_full_period_guard,
    apply_d3_v100_policy_r12_guard,
    apply_d3_v101_policy_r7_second_axis_guard,
    apply_d3_v102_policy_r2_second_axis_guard,
    apply_d3_v103_d3ev_r7_second_axis_guard,
    apply_d3_v104_d3ev_r3_second_axis_guard,
)


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
    assert "mainBeforeV97Guard" in meta
    assert "policyActionMain" in meta
    assert "mainBeforeV98Guard" in meta
    assert "v98Guard" in meta
    assert "mainBeforeV99Guard" in meta
    assert "v99Guard" in meta
    assert "mainBeforeV100Guard" in meta
    assert "v100Guard" in meta
    assert "mainBeforeV101Guard" in meta
    assert "v101Guard" in meta
    assert "mainBeforeV102Guard" in meta
    assert "v102Guard" in meta
    assert "mainBeforeV103Guard" in meta
    assert "v103Guard" in meta
    assert "v104Guard" in meta
    assert meta["selectionMode"] == "crossyear_robust_regime"
    assert meta["finalGuard"] is None


def test_trifecta_axes_are_not_mutated() -> None:
    selector = RollingRebuildSingleWin()
    selector.begin_day("2026-01-04")
    race = sample_race(False)
    before = list(race["prediction"]["axes"])
    main, _meta, _rows = selector.decide("2026-01-04", race)
    assert race["prediction"]["axes"] == before
    assert main in [1, 2, 3, 4, 5]

def test_v106_effective_path_is_race_number_agnostic() -> None:
    selector = RollingRebuildSingleWin()
    selector.begin_day("2026-01-04")
    low = sample_race(False)
    high = sample_race(False)
    low["raceNo"] = 2
    high["raceNo"] = 12
    low["horseCount"] = high["horseCount"] = 5
    low_main, low_meta, _ = selector.decide("2026-01-04", low)
    high_main, high_meta, _ = selector.decide("2026-01-04", high)
    assert low_main == high_main
    assert low_meta["selectionMode"] == "crossyear_robust_regime"
    assert high_meta["selectionMode"] == "crossyear_robust_regime"
    assert low_meta["finalGuard"] is None and high_meta["finalGuard"] is None
    for key in ("v98Guard", "v99Guard", "v100Guard", "v101Guard", "v102Guard", "v103Guard", "v104Guard"):
        assert low_meta[key] is None and high_meta[key] is None
    assert low_meta["regimeHorizonDays"] == [30, 120, 365]


def test_d3_field_size_guard_boundaries() -> None:
    base = {"prediction": {"axes": [7, 3]}}

    for horse_count, race_no in [(5, 1), (10, 12), (16, 1), (11, 5), (12, 6), (13, 8)]:
        race = dict(base, horseCount=horse_count, raceNo=race_no)
        main, guard = apply_d3_field_size_guard("d3_ev", 2, race)
        assert main == 7
        assert guard == "d3_ev_field_size_guard"

    for horse_count, race_no in [(11, 4), (12, 9), (13, 12), (14, 6), (15, 6), (17, 6), (18, 6)]:
        race = dict(base, horseCount=horse_count, raceNo=race_no)
        main, guard = apply_d3_field_size_guard("d3_ev", 2, race)
        assert main == 2
        assert guard is None

    race = dict(base, horseCount=16, raceNo=11)
    main, guard = apply_d3_field_size_guard("policy", 2, race)
    assert main == 2 and guard is None
    main, guard = apply_d3_field_size_guard("payout_ev", 2, race)
    assert main == 2 and guard is None


def test_d3_midcard_policy_guard_boundaries() -> None:
    base = {"prediction": {"axes": [7, 3]}, "horseCount": 16}

    # v96 overrides the older field-size guard in the robust 5R-8R band.
    for race_no in (5, 6, 7, 8):
        race = dict(base, raceNo=race_no)
        main, guard = apply_d3_final_guard(
            "d3_ev", 2, race, policy_main=5
        )
        assert main == 5
        assert guard == "d3_ev_midcard_policy_guard"

    # Adjacent races retain v94 behavior; 16 runners therefore fall back to axis 7.
    for race_no in (4, 9):
        race = dict(base, raceNo=race_no)
        main, guard = apply_d3_final_guard(
            "d3_ev", 2, race, policy_main=5
        )
        assert main == 7
        assert guard == "d3_ev_field_size_guard"

    # Non-d3_ev regimes are unaffected by v96.
    race = dict(base, raceNo=6)
    main, guard = apply_d3_final_guard("policy", 2, race, policy_main=5)
    assert main == 2 and guard is None

    # Missing fallback remains safe and preserves the cumulative v94 guard.
    main, guard = apply_d3_final_guard("d3_ev", 2, race)
    assert main == 7 and guard == "d3_ev_field_size_guard"


def test_v97_field_policy_guard_boundaries() -> None:
    # v97 is a final-only reliability layer for medium fields.
    for action in ("policy", "d3_ev", "payout_ev"):
        for horse_count in (10, 11, 12, 13, 14):
            race = {
                "prediction": {"axes": [7, 3]},
                "horseCount": horse_count,
                "raceNo": 10,
            }
            main, guard = apply_d3_final_guard(
                action, 2, race, policy_main=5, policy_action_main=6
            )
            assert main == 6
            assert guard == "field_10_14_policy_action_guard"

    # Adjacent field sizes preserve the cumulative v96/v94 result.
    race = {"prediction": {"axes": [7, 3]}, "horseCount": 15, "raceNo": 10}
    main, guard = apply_d3_final_guard(
        "policy", 2, race, policy_action_main=6
    )
    assert main == 2 and guard is None

    # If the stable policy already matches, do not manufacture a guard marker.
    race = {"prediction": {"axes": [7, 3]}, "horseCount": 12, "raceNo": 10}
    main, guard = apply_d3_final_guard(
        "policy", 2, race, policy_action_main=2
    )
    assert main == 2 and guard is None



def _v98_scored() -> list[dict]:
    return [
        {"horse_number": 1, "_recent": 70, "_total": 75, "_expected_popularity": 7, "current_flow": 0.70},
        {"horse_number": 2, "_recent": 72, "_total": 74, "_expected_popularity": 4, "current_flow": 0.50},
        {"horse_number": 3, "_recent": 68, "_total": 76, "_expected_popularity": 3, "current_flow": 0.66},
    ]


def test_v98_action_specific_guards() -> None:
    race = {"prediction": {"axes": [2, 3]}, "horseCount": 16, "raceNo": 10}
    scored = _v98_scored()

    # payout_ev: axis Recent support fires first.
    main, guard = apply_d3_v98_action_guard(
        "payout_ev", 1, race, scored, {"policy": 1}
    )
    assert main == 2 and guard == "payout_ev_axis_recent_guard"

    # payout_ev: a distinct policy horse with Total support has final precedence.
    main, guard = apply_d3_v98_action_guard(
        "payout_ev", 1, race, scored, {"policy": 3}
    )
    assert main == 3 and guard == "payout_ev_policy_total_guard"

    # d3_ev: axis is eligible only inside expected-popularity ranks 1-5.
    main, guard = apply_d3_v98_action_guard(
        "d3_ev", 1, race, scored, {"policy": 3}
    )
    assert main == 2 and guard == "d3_ev_axis_top5_expected_pop_guard"

    scored_tail = [dict(x) for x in scored]
    scored_tail[1]["_expected_popularity"] = 6
    main, guard = apply_d3_v98_action_guard(
        "d3_ev", 1, race, scored_tail, {"policy": 3}
    )
    assert main == 1 and guard is None

    # policy: 0.20 lower currentFlow is inside the contrarian pocket.
    main, guard = apply_d3_v98_action_guard(
        "policy", 1, race, scored, {"policy": 3}
    )
    assert main == 2 and guard == "policy_axis_contrarian_flow_guard"

    scored_near = [dict(x) for x in scored]
    scored_near[1]["current_flow"] = 0.56
    main, guard = apply_d3_v98_action_guard(
        "policy", 1, race, scored_near, {"policy": 3}
    )
    assert main == 1 and guard is None

def test_v100_policy_r12_axis_guard() -> None:
    race = {"prediction": {"axes": [7, 3]}, "raceNo": 12, "horseCount": 16}

    main, guard = apply_d3_v100_policy_r12_guard("policy", 2, race)
    assert main == 7
    assert guard == "policy_r12_axis_guard"

    # Adjacent race numbers and other regime actions are untouched.
    for race_no in (11, 13):
        main, guard = apply_d3_v100_policy_r12_guard(
            "policy", 2, dict(race, raceNo=race_no)
        )
        assert main == 2 and guard is None
    for action in ("d3_ev", "payout_ev"):
        main, guard = apply_d3_v100_policy_r12_guard(action, 2, race)
        assert main == 2 and guard is None

    # If the established axis already matches the v99 final pick, preserve it without
    # creating a misleading guard marker.
    main, guard = apply_d3_v100_policy_r12_guard("policy", 7, race)
    assert main == 7 and guard is None

    # Missing/invalid axis must fail closed.
    main, guard = apply_d3_v100_policy_r12_guard(
        "policy", 2, {"prediction": {"axes": []}, "raceNo": 12}
    )
    assert main == 2 and guard is None



def test_v101_policy_r7_second_axis_guard() -> None:
    race = {"prediction": {"axes": [7, 3]}, "raceNo": 7, "horseCount": 16}

    main, guard = apply_d3_v101_policy_r7_second_axis_guard("policy", 2, race)
    assert main == 3
    assert guard == "policy_r7_second_axis_guard"

    # Adjacent race numbers and other actions are untouched.
    for race_no in (6, 8):
        main, guard = apply_d3_v101_policy_r7_second_axis_guard(
            "policy", 2, dict(race, raceNo=race_no)
        )
        assert main == 2 and guard is None
    for action in ("d3_ev", "payout_ev"):
        main, guard = apply_d3_v101_policy_r7_second_axis_guard(action, 2, race)
        assert main == 2 and guard is None

    # Matching/missing second axes fail closed without a misleading marker.
    main, guard = apply_d3_v101_policy_r7_second_axis_guard("policy", 3, race)
    assert main == 3 and guard is None
    main, guard = apply_d3_v101_policy_r7_second_axis_guard(
        "policy", 2, {"prediction": {"axes": [7]}, "raceNo": 7}
    )
    assert main == 2 and guard is None


def test_v102_policy_r2_second_axis_guard() -> None:
    race = {"prediction": {"axes": [7, 3]}, "raceNo": 2, "horseCount": 16}

    main, guard = apply_d3_v102_policy_r2_second_axis_guard("policy", 2, race)
    assert main == 3
    assert guard == "policy_r2_second_axis_guard"

    # Adjacent race numbers and other actions are untouched.
    for race_no in (1, 3):
        main, guard = apply_d3_v102_policy_r2_second_axis_guard(
            "policy", 2, dict(race, raceNo=race_no)
        )
        assert main == 2 and guard is None
    for action in ("d3_ev", "payout_ev"):
        main, guard = apply_d3_v102_policy_r2_second_axis_guard(action, 2, race)
        assert main == 2 and guard is None

    # Matching/missing second axes fail closed without a misleading marker.
    main, guard = apply_d3_v102_policy_r2_second_axis_guard("policy", 3, race)
    assert main == 3 and guard is None
    main, guard = apply_d3_v102_policy_r2_second_axis_guard(
        "policy", 2, {"prediction": {"axes": [7]}, "raceNo": 2}
    )
    assert main == 2 and guard is None



def test_v103_d3ev_r7_second_axis_guard() -> None:
    race = {"prediction": {"axes": [7, 3]}, "raceNo": 7, "horseCount": 16}

    main, guard = apply_d3_v103_d3ev_r7_second_axis_guard("d3_ev", 2, race)
    assert main == 3
    assert guard == "d3_ev_r7_second_axis_guard"

    # Adjacent race numbers and other actions are untouched.
    for race_no in (6, 8):
        main, guard = apply_d3_v103_d3ev_r7_second_axis_guard(
            "d3_ev", 2, dict(race, raceNo=race_no)
        )
        assert main == 2 and guard is None
    for action in ("policy", "payout_ev"):
        main, guard = apply_d3_v103_d3ev_r7_second_axis_guard(action, 2, race)
        assert main == 2 and guard is None

    # Matching/missing second axes fail closed without a misleading marker.
    main, guard = apply_d3_v103_d3ev_r7_second_axis_guard("d3_ev", 3, race)
    assert main == 3 and guard is None
    main, guard = apply_d3_v103_d3ev_r7_second_axis_guard(
        "d3_ev", 2, {"prediction": {"axes": [7]}, "raceNo": 7}
    )
    assert main == 2 and guard is None

def test_v104_d3ev_r3_second_axis_guard() -> None:
    race = {"prediction": {"axes": [7, 3]}, "raceNo": 3, "horseCount": 16}

    main, guard = apply_d3_v104_d3ev_r3_second_axis_guard("d3_ev", 2, race)
    assert main == 3
    assert guard == "d3_ev_r3_second_axis_guard"

    for race_no in (2, 4):
        main, guard = apply_d3_v104_d3ev_r3_second_axis_guard(
            "d3_ev", 2, dict(race, raceNo=race_no)
        )
        assert main == 2 and guard is None
    for action in ("policy", "payout_ev"):
        main, guard = apply_d3_v104_d3ev_r3_second_axis_guard(action, 2, race)
        assert main == 2 and guard is None

    main, guard = apply_d3_v104_d3ev_r3_second_axis_guard("d3_ev", 3, race)
    assert main == 3 and guard is None
    main, guard = apply_d3_v104_d3ev_r3_second_axis_guard(
        "d3_ev", 2, {"prediction": {"axes": [7]}, "raceNo": 3}
    )
    assert main == 2 and guard is None


def test_missing_field_size_is_not_treated_as_small_field() -> None:
    race = {"prediction": {"axes": [7, 3]}, "raceNo": 10}
    main, guard = apply_d3_field_size_guard("d3_ev", 2, race)
    assert main == 2 and guard is None


def test_v99_full_period_payout_total_guard() -> None:
    scored = [
        {"horse_number": 1, "_total": 72},
        {"horse_number": 2, "_total": 65},
        {"horse_number": 3, "_total": 64},
    ]

    # Boundary 65 is included.
    main, guard = apply_d3_v99_full_period_guard(
        1, scored, {"payout_ev": 2}
    )
    assert main == 2
    assert guard == "full_period_payout_total65_guard"

    # 64 remains on the cumulative v98 pick.
    main, guard = apply_d3_v99_full_period_guard(
        1, scored, {"payout_ev": 3}
    )
    assert main == 1 and guard is None

    # Matching or missing candidates never manufacture a guard marker.
    main, guard = apply_d3_v99_full_period_guard(
        1, scored, {"payout_ev": 1}
    )
    assert main == 1 and guard is None
    main, guard = apply_d3_v99_full_period_guard(1, scored, {})
    assert main == 1 and guard is None


if __name__ == "__main__":
    tests = [
        test_finish_day_adds_result_labels,
        test_trifecta_axes_are_not_mutated,
        test_v106_effective_path_is_race_number_agnostic,
        test_d3_field_size_guard_boundaries,
        test_d3_midcard_policy_guard_boundaries,
        test_v97_field_policy_guard_boundaries,
        test_v98_action_specific_guards,
        test_v99_full_period_payout_total_guard,
        test_v100_policy_r12_axis_guard,
        test_v101_policy_r7_second_axis_guard,
        test_v102_policy_r2_second_axis_guard,
        test_v103_d3ev_r7_second_axis_guard,
        test_v104_d3ev_r3_second_axis_guard,
        test_missing_field_size_is_not_treated_as_small_field,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"OK: {len(tests)} v106 single-win runtime tests passed")
