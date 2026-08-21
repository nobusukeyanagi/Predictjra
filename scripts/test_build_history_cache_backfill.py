#!/usr/bin/env python3
"""Focused leakage/completeness tests for 2026 result-derived history backfill."""
from __future__ import annotations

import tarfile
import tempfile
from pathlib import Path

import pandas as pd

from build_history_cache import (
    build_cache,
    PROHIBITED_CURRENT_COLUMNS,
    RESULT_CARD_COLUMNS,
    synthesize_card_from_result,
    parse_netkeiba_full_result,
    validate_archive_structure,
    validate_result_payout,
)


def sample_result(race_id: str = "202606010301", race_date: str = "2026-01-10") -> pd.DataFrame:
    rows = []
    for horse in range(1, 13):
        finish = horse
        rows.append({
            "race_id": race_id,
            "race_name": "3歳未勝利",
            "date": race_date,
            "surface": "ダート",
            "distance_m": 1200,
            "waku": (horse + 1) // 2,
            "horse_number": horse,
            "horse_name": f"テストホース{horse}",
            "horse_id": f"2023{horse:06d}",
            "sex_age": "牡3",
            "weight_carried": 57.0,
            "jockey": f"騎手{horse}",
            "jockey_id": f"{horse:05d}",
            "trainer": f"調教師{horse}",
            "trainer_id": f"{1000 + horse:05d}",
            "finish_position": finish,
            "popularity": finish,
            "win_odds": 1.5 + horse,
            "time": f"1:1{horse % 10}.0",
            "margin": "",
            "horse_weight": f"{440 + horse}(0)",
        })
    return pd.DataFrame(rows)


def sample_payout(race_id: str = "202606010301") -> pd.DataFrame:
    return pd.DataFrame([
        {"race_id": race_id, "bet_type": "三連単", "combination": "1-2-3", "amount": 2380, "popularity": 3},
    ])


def test_result_projection_is_leakage_safe() -> None:
    result = sample_result()
    card = synthesize_card_from_result(result, "202606010301")
    assert list(card.columns) == RESULT_CARD_COLUMNS
    assert not (PROHIBITED_CURRENT_COLUMNS & set(card.columns))
    for post_race in ("finish_position", "time", "margin"):
        assert post_race not in card.columns
    assert len(card) == 12
    assert set(card["horse_number"].astype(int)) == set(range(1, 13))


def test_result_and_payout_must_agree() -> None:
    result = sample_result()
    payout = sample_payout()
    validate_result_payout(result, payout, "202606010301")

    bad = payout.copy()
    bad.loc[0, "combination"] = "1-3-2"
    try:
        validate_result_payout(result, bad, "202606010301")
    except ValueError as exc:
        assert "conflicts with result top finishers" in str(exc)
    else:
        raise AssertionError("mismatched trifecta payout was not rejected")


def test_dead_heat_trifecta_is_accepted() -> None:
    result = sample_result()
    # Official dead-heat style: horses 1 and 2 share first; next horse is recorded third.
    result.loc[result["horse_number"] == 2, "finish_position"] = 1
    result.loc[result["horse_number"] == 3, "finish_position"] = 3
    payout = pd.DataFrame([
        {"race_id": "202606010301", "bet_type": "三連単", "combination": "1-2-3", "amount": 1200, "popularity": 1},
        {"race_id": "202606010301", "bet_type": "三連単", "combination": "2-1-3", "amount": 1300, "popularity": 2},
    ])
    validate_result_payout(result, payout, "202606010301")


def test_archive_structure_requires_all_12_races_and_no_day_gap() -> None:
    complete = {
        "2026-01-04": {
            "raceFiles": [f"2026060101{r:02d}.csv" for r in range(1, 13)],
            "errors": [],
        },
        "2026-01-05": {
            "raceFiles": [f"2026060102{r:02d}.csv" for r in range(1, 13)],
            "errors": [],
        },
    }
    validate_archive_structure(complete)

    missing_race = {k: {**v, "raceFiles": list(v["raceFiles"])} for k, v in complete.items()}
    missing_race["2026-01-04"]["raceFiles"].pop()
    try:
        validate_archive_structure(missing_race)
    except RuntimeError as exc:
        assert "structural holes" in str(exc)
    else:
        raise AssertionError("missing race was not rejected")

    missing_day = {
        "2026-01-04": complete["2026-01-04"],
        "2026-01-11": {
            "raceFiles": [f"2026060103{r:02d}.csv" for r in range(1, 13)],
            "errors": [],
        },
    }
    try:
        validate_archive_structure(missing_day)
    except RuntimeError as exc:
        assert "missing meeting-day" in str(exc)
    else:
        raise AssertionError("missing meeting-day was not rejected")




def test_full_cache_build_from_result_only_day() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "source"
        cache = Path(tmp) / "cache"
        (root / "data" / "race_cards").mkdir(parents=True)
        result_dir = root / "data" / "race_results" / "2026"
        payout_dir = root / "data" / "race_payouts"
        result_dir.mkdir(parents=True)
        payout_dir.mkdir(parents=True)

        for race_no in range(1, 13):
            rid = f"2026060101{race_no:02d}"
            result = sample_result(rid, "2026-01-04")
            payout = sample_payout(rid)
            result.to_csv(result_dir / f"{rid}.csv", index=False, encoding="utf-8-sig")
            payout.to_csv(payout_dir / f"{rid}.csv", index=False, encoding="utf-8-sig")

        manifest = build_cache(root, cache, web_discovery=False)
        assert manifest["safeDates"] == ["2026-01-04"]
        assert manifest["resultDerivedDates"] == ["2026-01-04"]
        assert manifest["dateRaceCounts"]["2026-01-04"] == 12
        assert manifest["runnerSnapshotSummary"][0]["resultDerivedCards"] == 12

        archive = cache / "history-source.tar.gz"
        assert archive.is_file()
        with tarfile.open(archive, "r:gz") as tf:
            member = tf.extractfile("data/race_cards/20260104/202606010101.csv")
            assert member is not None
            text = member.read().decode("utf-8-sig")
        header = text.splitlines()[0].split(",")
        assert not (PROHIBITED_CURRENT_COLUMNS & set(header))
        assert "finish_position" not in header
        assert "time" not in header


def test_exact_discovered_race_set_is_enforced() -> None:
    by_date = {
        "2026-01-04": {
            "raceFiles": [f"2026060101{r:02d}.csv" for r in range(1, 13)],
            "errors": [],
        },
    }
    expected = {"2026-01-04": [f"2026060101{r:02d}" for r in range(1, 13)]}
    validate_archive_structure(by_date, expected_by_date=expected)
    bad = {"2026-01-04": expected["2026-01-04"][:-1]}
    try:
        validate_archive_structure(by_date, expected_by_date=bad)
    except RuntimeError as exc:
        assert "does not match independently discovered" in str(exc)
    else:
        raise AssertionError("unexpected archive race was not rejected")


def test_netkeiba_result_repair_parser() -> None:
    html = """
    <html><head><title>レース一覧 | 2026年1月4日</title></head><body>
    <h1 class='RaceName'>3歳未勝利</h1>
    <div class='RaceData01'>10:05発走 / ダ1200m (右) / 天候:晴 / 馬場:良</div>
    <table class='RaceTable01'>
      <tr><th>着順</th><th>枠</th><th>馬番</th><th>馬名</th><th>性齢</th><th>斤量</th><th>騎手</th><th>タイム</th><th>着差</th><th>人気</th><th>単勝オッズ</th><th>後3F</th><th>コーナー通過順</th><th>厩舎</th><th>馬体重(増減)</th></tr>
    """ + "".join(
        f"<tr><td>{i}</td><td>{(i+1)//2}</td><td>{i}</td>"
        f"<td><a href='https://db.netkeiba.com/horse/2023{i:06d}/'>馬{i}</a></td>"
        f"<td>牡3</td><td>57</td>"
        f"<td><a href='https://db.netkeiba.com/jockey/{1000+i:05d}/'>騎手{i}</a></td>"
        f"<td>1:1{i%10}.0</td><td></td><td>{i}</td><td>{1.0+i:.1f}</td><td>38.0</td><td>1-1</td>"
        f"<td><a href='https://db.netkeiba.com/trainer/{2000+i:05d}/'>美浦調教師{i}</a></td><td>450(0)</td></tr>"
        for i in range(1, 13)
    ) + """
    </table>
    <table><tr><th>3連単</th><td>1 → 2 → 3</td><td>2,380円</td><td>3人気</td></tr></table>
    </body></html>
    """
    result, payout = parse_netkeiba_full_result(html, "202606010101", pd.Timestamp("2026-01-04").date())
    assert len(result) == 12
    assert result.iloc[0]["horse_id"] == "2023000001"
    assert int(result.iloc[0]["popularity"]) == 1
    assert payout.iloc[0]["combination"] == "1-2-3"
    assert int(payout.iloc[0]["amount"]) == 2380


def main() -> int:
    test_result_projection_is_leakage_safe()
    test_result_and_payout_must_agree()
    test_dead_heat_trifecta_is_accepted()
    test_archive_structure_requires_all_12_races_and_no_day_gap()
    test_full_cache_build_from_result_only_day()
    test_exact_discovered_race_set_is_enforced()
    test_netkeiba_result_repair_parser()
    print("OK: historical result backfill leakage/completeness tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
