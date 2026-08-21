#!/usr/bin/env python3
"""Focused leakage/completeness/multisource tests for historical backfill v7."""
from __future__ import annotations

import tarfile
import tempfile
from pathlib import Path

import pandas as pd

import build_history_cache as bhc


def sample_result(race_id: str = "202606010301", race_date: str = "2026-01-10", runners: int = 12) -> pd.DataFrame:
    rows = []
    for horse in range(1, runners + 1):
        rows.append({
            "race_id": race_id,
            "race_name": "3歳未勝利",
            "date": race_date,
            "surface": "ダート",
            "distance_m": 1200,
            "waku": min(8, (horse + 1) // 2),
            "horse_number": horse,
            "horse_name": f"テストホース{horse}",
            "horse_id": f"2023{horse:06d}",
            "sex_age": "牡3",
            "weight_carried": 57.0,
            "jockey": f"騎手{horse}",
            "jockey_id": f"{horse:05d}",
            "trainer": f"調教師{horse}",
            "trainer_id": f"{1000 + horse:05d}",
            "finish_position": horse,
            "popularity": horse,
            "win_odds": 1.5 + horse,
            "time": f"1:1{horse % 10}.0",
            "margin": "",
            "last_3f": 36.0,
            "passing_order": "1-1",
            "horse_weight": f"{440 + horse}(0)",
        })
    return pd.DataFrame(rows)


def sample_payout(race_id: str = "202606010301") -> pd.DataFrame:
    return pd.DataFrame([{
        "race_id": race_id,
        "bet_type": "三連単",
        "combination": "1-2-3",
        "amount": 2380,
        "popularity": 3,
    }])


def test_result_projection_is_leakage_safe() -> None:
    card = bhc.synthesize_card_from_result(sample_result(), "202606010301")
    assert list(card.columns) == bhc.RESULT_CARD_COLUMNS
    assert not (bhc.PROHIBITED_CURRENT_COLUMNS & set(card.columns))
    for post_race in ("finish_position", "time", "margin"):
        assert post_race not in card.columns


def test_result_and_payout_must_agree_and_dead_heat_is_allowed() -> None:
    result = sample_result()
    bhc.validate_result_payout(result, sample_payout(), "202606010301")
    bad = sample_payout()
    bad.loc[0, "combination"] = "1-3-2"
    try:
        bhc.validate_result_payout(result, bad, "202606010301")
    except ValueError as exc:
        assert "conflicts" in str(exc)
    else:
        raise AssertionError("mismatched trifecta was not rejected")

    dead = sample_result()
    dead.loc[dead["horse_number"] == 2, "finish_position"] = 1
    dead.loc[dead["horse_number"] == 3, "finish_position"] = 3
    payout = pd.DataFrame([
        {"race_id": "202606010301", "bet_type": "三連単", "combination": "1-2-3", "amount": 1200, "popularity": 1},
        {"race_id": "202606010301", "bet_type": "三連単", "combination": "2-1-3", "amount": 1300, "popularity": 2},
    ])
    bhc.validate_result_payout(dead, payout, "202606010301")


def test_authoritative_short_meeting_is_complete_without_12r_assumption() -> None:
    ids = [f"2026050101{r:02d}" for r in range(1, 8)]
    by_date = {"2026-02-07": {"raceFiles": [f"{rid}.csv" for rid in ids], "errors": []}}
    bhc.validate_archive_structure(by_date, expected_by_date={"2026-02-07": ids})

    missing = ids[:-1]
    try:
        bhc.validate_archive_structure(by_date, expected_by_date={"2026-02-07": missing})
    except RuntimeError as exc:
        assert "does not match independently discovered" in str(exc)
    else:
        raise AssertionError("unexpected race was not rejected")


def test_source_fallback_never_invents_missing_race_numbers() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        result_dir = root / "data" / "race_results" / "2026"
        result_dir.mkdir(parents=True)
        observed = [1, 2, 3, 4, 6, 7]
        for race_no in observed:
            rid = f"2026050101{race_no:02d}"
            sample_result(rid, "2026-02-07").to_csv(result_dir / f"{rid}.csv", index=False, encoding="utf-8-sig")
        expected, _ = bhc.discover_expected_from_source(
            root,
            start=pd.Timestamp("2026-02-07").date(),
            end=pd.Timestamp("2026-02-07").date(),
        )
        assert expected["2026-02-07"] == [f"2026050101{r:02d}" for r in observed]
        assert "202605010105" not in expected["2026-02-07"]


def test_sportsnavi_meeting_list_extracts_only_real_races() -> None:
    html = "".join(
        f"<a href='/keiba/race/result/26050101{r:02d}'>R{r}</a>" for r in range(1, 8)
    )
    ids = bhc.extract_sportsnavi_meeting_race_ids(html, "2026050101")
    assert ids == [f"2026050101{r:02d}" for r in range(1, 8)]


def sports_html(race_id: str = "202606010101") -> str:
    rows = []
    for i in range(1, 13):
        rows.append(
            f"<tr><td>{i}</td><td>{min(8,(i+1)//2)}</td><td>{i}</td>"
            f"<td><a href='/keiba/directory/horse/2023{i:06d}/'>馬{i}</a> 牡3/{450+i}(0)</td>"
            f"<td>1:1{i%10}.0 {'-' if i==1 else '1/2馬身'}</td>"
            f"<td>01-01 {35+i/10:.1f}</td>"
            f"<td><a href='/keiba/directory/jockey/{1000+i:05d}/'>騎手{i}</a> 57.0</td>"
            f"<td>{i}({1+i:.1f})</td>"
            f"<td><a href='/keiba/directory/trainer/{2000+i:05d}/'>調教師{i}</a></td></tr>"
        )
    return f"""
    <html><head><title>競馬 - 2026年サラ系3歳未勝利 結果</title></head><body>
      <h2>サラ系3歳未勝利</h2>
      <div>ダート・右 1200m 天気：晴 馬場：良</div>
      <table><tr><th>馬券</th><th>馬番</th><th>払戻金</th><th>人気</th></tr>
      <tr><td>3連単</td><td>1-2-3</td><td>2,380円</td><td>3</td></tr></table>
      <table><tr><th>着順</th><th>枠番</th><th>馬番</th><th>馬名性齢/馬体重</th><th>タイム着差</th><th>通過順位上がり3F</th><th>騎手名斤量</th><th>人気（オッズ）</th><th>調教師</th></tr>
      {''.join(rows)}
      </table>
    </body></html>
    """


def test_sportsnavi_result_and_payout_parser() -> None:
    rid = "202606010101"
    result, payout = bhc.parse_sportsnavi_full_result(
        sports_html(rid), rid, pd.Timestamp("2026-01-04").date()
    )
    assert len(result) == 12
    assert result.iloc[0]["horse_id"] == "2023000001"
    assert result.iloc[0]["jockey_id"] == "01001"
    assert result.iloc[0]["trainer_id"] == "02001"
    assert int(result.iloc[0]["popularity"]) == 1
    assert payout.iloc[0]["combination"] == "1-2-3"
    assert int(payout.iloc[0]["amount"]) == 2380


def test_payout_only_repair_preserves_valid_result() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        result_dir = root / "data" / "race_results" / "2026"
        payout_dir = root / "data" / "race_payouts"
        result_dir.mkdir(parents=True)
        payout_dir.mkdir(parents=True)
        rid = "202606010101"
        result_path = result_dir / f"{rid}.csv"
        sample_result(rid, "2026-01-04").to_csv(result_path, index=False, encoding="utf-8-sig")
        original = result_path.read_bytes()

        old = bhc.fetch_multisource_payout
        try:
            bhc.fetch_multisource_payout = lambda race_id: (sample_payout(race_id), "test://payout")
            info = bhc.repair_result_archive_from_web(
                root,
                start=pd.Timestamp("2026-01-04").date(),
                end=pd.Timestamp("2026-01-04").date(),
                verify_static_lists=False,
            )
        finally:
            bhc.fetch_multisource_payout = old
        assert not info["unresolved"]
        assert info["repaired"][0]["repairKind"] == "payout-only"
        assert result_path.read_bytes() == original
        assert (payout_dir / f"{rid}.csv").is_file()


def test_one_fetch_failure_does_not_disable_later_repairs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        result_dir = root / "data" / "race_results" / "2026"
        payout_dir = root / "data" / "race_payouts"
        result_dir.mkdir(parents=True)
        payout_dir.mkdir(parents=True)
        ids = []
        for n in range(1, 8):
            rid = f"2026060101{n:02d}"
            ids.append(rid)
            sample_result(rid, "2026-01-04").to_csv(result_dir / f"{rid}.csv", index=False, encoding="utf-8-sig")

        old = bhc.fetch_multisource_payout
        try:
            def fake(race_id: str):
                if race_id == ids[0]:
                    raise RuntimeError("temporary source failure")
                return sample_payout(race_id), "test://payout"
            bhc.fetch_multisource_payout = fake
            info = bhc.repair_result_archive_from_web(
                root,
                start=pd.Timestamp("2026-01-04").date(),
                end=pd.Timestamp("2026-01-04").date(),
                verify_static_lists=False,
            )
        finally:
            bhc.fetch_multisource_payout = old
        assert len(info["unresolved"]) == 1
        assert len(info["repaired"]) == 6
        for rid in ids[1:]:
            assert (payout_dir / f"{rid}.csv").is_file()


def test_full_cache_build_accepts_verified_shortened_day_and_has_zero_skips() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "source"
        cache = Path(tmp) / "cache"
        (root / "data" / "race_cards").mkdir(parents=True)
        result_dir = root / "data" / "race_results" / "2026"
        payout_dir = root / "data" / "race_payouts"
        result_dir.mkdir(parents=True)
        payout_dir.mkdir(parents=True)
        for race_no in range(1, 8):
            rid = f"2026060101{race_no:02d}"
            sample_result(rid, "2026-01-04").to_csv(result_dir / f"{rid}.csv", index=False, encoding="utf-8-sig")
            sample_payout(rid).to_csv(payout_dir / f"{rid}.csv", index=False, encoding="utf-8-sig")

        manifest = bhc.build_cache(root, cache, web_discovery=False)
        assert manifest["safeDates"] == ["2026-01-04"]
        assert manifest["skippedDates"] == []
        assert manifest["dateRaceCounts"]["2026-01-04"] == 7
        with tarfile.open(cache / "history-source.tar.gz", "r:gz") as tf:
            member = tf.extractfile("data/race_cards/20260104/202606010101.csv")
            assert member is not None
            header = member.read().decode("utf-8-sig").splitlines()[0].split(",")
        assert not (bhc.PROHIBITED_CURRENT_COLUMNS & set(header))
        assert "finish_position" not in header


def test_netkeiba_parser_still_works() -> None:
    html = """
    <html><body><h1 class='RaceName'>3歳未勝利</h1>
    <div class='RaceData01'>10:05発走 / ダ1200m (右) / 天候:晴 / 馬場:良</div>
    <table class='RaceTable01'><tr><th>着順</th><th>枠</th><th>馬番</th><th>馬名</th><th>性齢</th><th>斤量</th><th>騎手</th><th>タイム</th><th>着差</th><th>人気</th><th>単勝オッズ</th><th>後3F</th><th>コーナー通過順</th><th>厩舎</th><th>馬体重(増減)</th></tr>
    """ + "".join(
        f"<tr><td>{i}</td><td>{min(8,(i+1)//2)}</td><td>{i}</td>"
        f"<td><a href='/horse/2023{i:06d}/'>馬{i}</a></td><td>牡3</td><td>57</td>"
        f"<td><a href='/jockey/{1000+i:05d}/'>騎手{i}</a></td><td>1:1{i%10}.0</td><td></td>"
        f"<td>{i}</td><td>{1+i:.1f}</td><td>38.0</td><td>1-1</td>"
        f"<td><a href='/trainer/{2000+i:05d}/'>調教師{i}</a></td><td>450(0)</td></tr>"
        for i in range(1, 13)
    ) + """</table><table><tr><th>三連単</th><td>1 → 2 → 3</td><td>2,380</td><td>3</td></tr></table></body></html>"""
    result, payout = bhc.parse_netkeiba_full_result(html, "202606010101", pd.Timestamp("2026-01-04").date())
    assert len(result) == 12
    assert int(payout.iloc[0]["amount"]) == 2380


def main() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for test in sorted(tests, key=lambda f: f.__name__):
        test()
    print(f"OK: {len(tests)} historical backfill v7 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
