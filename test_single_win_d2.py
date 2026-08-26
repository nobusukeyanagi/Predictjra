#!/usr/bin/env python3
from single_win_d2 import Policy, choose_main, choose_second, race_feature_rows, selected_set, trifecta_return


def sample_race():
    horses = []
    for no, total, recent, ep, ev in [
        (1, 88, 86, 1, 45),
        (2, 85, 84, 4, 72),
        (3, 82, 80, 2, 55),
        (4, 78, 79, 5, 68),
        (5, 75, 74, 3, 40),
        (6, 70, 70, 6, 75),
    ]:
        horses.append({
            "no": no,
            "recentIndex": recent,
            "currentRun": total,
            "currentFlow": total - 2,
            "currentPower": total - 1,
            "today": total,
            "total": total,
            "rank": no,
            "expectedPopularity": ep,
            "singleEV": ev,
        })
    return {
        "raceId": "TEST00000001",
        "prediction": {"axes": [1, 4], "opponents": [2]},
        "danger": [3],
        "result": {
            "places": [[2], [1], [4]],
            "winPayouts": [{"horses": [2], "payout": 520}],
            "trifectas": [{"horses": [2, 1, 4], "payout": 8200}],
        },
        "modelMeta": {
            "nonStarters": [],
            "indexDetail": {
                "paceRegime": "fast",
                "raceConditions": {"surface": "芝", "distanceM": 1600},
                "horses": horses,
            },
        },
    }


def test_feature_contract():
    race = sample_race()
    rows = race_feature_rows("2026-08-01", race)
    assert len(rows) == 6
    row2 = next(r for r in rows if r["horse_number"] == 2)
    assert row2["is_winner"] == 1
    assert row2["is_top3"] == 1
    assert row2["win_payout"] == 520
    assert row2["selected"] == 1
    assert next(r for r in rows if r["horse_number"] == 3)["danger"] == 1


def test_policy_and_trifecta():
    race = sample_race()
    rows = race_feature_rows("2026-08-01", race)
    for r in rows:
        # Synthetic D2 prediction: horse 2 has the best value while remaining ability-safe.
        r["d2_win_prob"] = {1: .28, 2: .24, 3: .18, 4: .15, 5: .09, 6: .06}[r["horse_number"]]
        r["d2_top3_prob"] = {1: .70, 2: .66, 3: .60, 4: .50, 5: .40, 6: .30}[r["horse_number"]]
        r["d2_ev"] = {1: .78, 2: 1.16, 3: .84, 4: .91, 5: .72, 6: .70}[r["horse_number"]]
    selected = selected_set(race)
    main = choose_main(rows, selected, Policy(top_k=3, max_total_gap=8, min_win_ratio=.5, min_top3_ratio=.5, top3_power=0, ability_power=0, legacy_power=0))
    assert main == 2
    second = choose_second(selected, main, rows)
    assert second == 4
    assert trifecta_return(race, selected, main, second) == 8200


if __name__ == "__main__":
    test_feature_contract()
    test_policy_and_trifecta()
    print("OK: single-win D2 synthetic tests passed")
