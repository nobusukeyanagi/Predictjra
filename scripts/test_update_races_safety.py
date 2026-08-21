#!/usr/bin/env python3
"""Regression tests for fail-closed live result updates."""
from __future__ import annotations

from datetime import date

import update_races as legacy
import update_races_v2 as updater

TARGET = date(2026, 8, 16)
RID1 = "202601010801"
RID2 = "202601010802"


def sample_result(combo=(1, 2, 3), payout=2380):
    return {
        "places": [[1], [2], [3]],
        "trifectas": [{"horses": list(combo), "payout": payout}],
    }


def sample_race(rid: str):
    return {
        "raceId": rid,
        "venue": "札幌",
        "raceNo": int(rid[-2:]),
        "horseCount": 12,
        "horseFrames": legacy.jra_frame_map(12),
        "prediction": {"axes": [1, 2], "opponents": [3, 4, 5, 6, 7]},
        "status": "pending",
        "payout": 0,
        "trifectaPayouts": [],
    }


def sample_data(*rids: str):
    return {
        "days": [{
            "date": TARGET.isoformat(),
            "races": [sample_race(rid) for rid in rids],
        }]
    }


def test_result_requires_prepared_day() -> None:
    diagnostics = {"races": []}
    try:
        updater.result_day({"days": []}, TARGET, diagnostics)
    except RuntimeError as exc:
        assert "never creates predictions" in str(exc)
    else:
        raise AssertionError("result mode must fail when prepare was not completed")
    assert diagnostics["publishBlocked"] is True


def test_partial_result_does_not_mutate_day() -> None:
    data = sample_data(RID1, RID2)
    diagnostics = {"races": []}
    original_fetch = legacy.fetch_result
    original_cancel = legacy.confirm_cancelled_race
    try:
        def fake_fetch(rid, _target):
            if rid == RID1:
                return sample_result(), "test"
            return None, "none"
        legacy.fetch_result = fake_fetch
        legacy.confirm_cancelled_race = lambda *_args: (False, "")
        try:
            updater.result_day(data, TARGET, diagnostics)
        except RuntimeError as exc:
            assert RID2 in str(exc)
        else:
            raise AssertionError("partial result update must fail closed")
    finally:
        legacy.fetch_result = original_fetch
        legacy.confirm_cancelled_race = original_cancel

    races = data["days"][0]["races"]
    assert [r["status"] for r in races] == ["pending", "pending"]
    assert diagnostics["publishBlocked"] is True
    assert diagnostics["unresolvedRaceIds"] == [RID2]


def test_confirmed_cancellation_is_removed_only_after_full_resolution() -> None:
    data = sample_data(RID1, RID2)
    diagnostics = {"races": []}
    original_fetch = legacy.fetch_result
    original_cancel = legacy.confirm_cancelled_race
    try:
        legacy.fetch_result = lambda rid, _target: (
            (sample_result(), "test") if rid == RID1 else (None, "none")
        )
        legacy.confirm_cancelled_race = lambda rid, _target: (
            (True, "test://jra") if rid == RID2 else (False, "")
        )
        changed = updater.result_day(data, TARGET, diagnostics)
    finally:
        legacy.fetch_result = original_fetch
        legacy.confirm_cancelled_race = original_cancel

    assert changed == 2
    races = data["days"][0]["races"]
    assert [r["raceId"] for r in races] == [RID1]
    assert races[0]["status"] == "hit"
    assert diagnostics["publishBlocked"] is False
    assert diagnostics["cancelledRaces"] == [{"raceId": RID2, "source": "test://jra"}]


def test_result_payout_mismatch_is_rejected() -> None:
    legacy.validate_result_payload(sample_result(), RID1)
    bad = sample_result(combo=(1, 3, 2))
    try:
        legacy.validate_result_payload(bad, RID1)
    except ValueError as exc:
        assert "conflicts" in str(exc)
    else:
        raise AssertionError("contradictory trifecta payout must be rejected")


def test_jra_program_parser_does_not_create_cancelled_notice_race() -> None:
    html = """
    <html><body>
      <div>2026年8月16日</div>
      <h2>1回札幌8日</h2>
      <div>1レース</div><div>2レース</div><div>3レース</div>
      <div>札幌競馬は第4レース以降を取りやめ</div>
    </body></html>
    """
    ids = legacy.extract_jra_calendar_race_ids(html, TARGET)
    assert ids == ["202601010801", "202601010802", "202601010803"]
    assert "202601010804" not in ids


if __name__ == "__main__":
    tests = [
        test_result_requires_prepared_day,
        test_partial_result_does_not_mutate_day,
        test_confirmed_cancellation_is_removed_only_after_full_resolution,
        test_result_payout_mismatch_is_rejected,
        test_jra_program_parser_does_not_create_cancelled_notice_race,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"OK: {len(tests)} live-update safety regression tests passed")
