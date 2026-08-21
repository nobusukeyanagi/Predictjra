#!/usr/bin/env python3
"""Regression tests for deterministic JRA frame-color data."""
from __future__ import annotations

from datetime import date

import update_races as legacy
import update_races_v2 as updater


def frame_sequence(field_size: int) -> list[int]:
    return [legacy.jra_frame_number(h, field_size) for h in range(1, field_size + 1)]


def test_known_allocations() -> None:
    expected = {
        5: [1, 2, 3, 4, 5],
        8: [1, 2, 3, 4, 5, 6, 7, 8],
        9: [1, 2, 3, 4, 5, 6, 7, 8, 8],
        10: [1, 2, 3, 4, 5, 6, 7, 7, 8, 8],
        12: [1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 8, 8],
        15: [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8],
        16: [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8],
        17: [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 8],
        18: [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7, 8, 8, 8],
    }
    for field_size, frames in expected.items():
        actual = frame_sequence(field_size)
        assert actual == frames, (field_size, actual, frames)


def test_bad_20260822_style_frames_are_repaired() -> None:
    # Regression for the 2026-08-22 issue: a scraper returned valid-looking but
    # contradictory 1..8 values (e.g. 10/11/12 -> frame 1).
    bad = [1, 2, 3, 4, 5, 6, 7, 8, 7, 1, 1, 1]
    entries = [
        {"horse": h, "frame": bad[h - 1], "name": f"horse-{h}"}
        for h in range(1, 13)
    ]
    repaired, mismatches = legacy.normalize_entry_frames(entries)
    assert [e["frame"] for e in repaired] == [1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 8, 8]
    assert mismatches
    assert {m["horse"] for m in mismatches} >= {6, 7, 8, 10, 11, 12}


def test_missing_frames_are_repaired() -> None:
    entries = [
        {"horse": h, "frame": None, "name": f"horse-{h}"}
        for h in range(1, 11)
    ]
    repaired, mismatches = legacy.normalize_entry_frames(entries)
    assert len(mismatches) == 10
    assert [e["frame"] for e in repaired] == [1, 2, 3, 4, 5, 6, 7, 7, 8, 8]


def test_partial_runner_parse_is_rejected() -> None:
    entries = [
        {"horse": h, "frame": None, "name": f"horse-{h}"}
        for h in [1, 2, 3, 5, 6]
    ]
    try:
        legacy.normalize_entry_frames(entries)
    except ValueError as exc:
        assert "runner sequence incomplete" in str(exc)
    else:
        raise AssertionError("partial runner set must not be accepted")



def test_prepare_repairs_existing_bad_frames_without_network() -> None:
    target = date(2026, 8, 22)
    race_id = "202601020101"
    bad = {str(h): f for h, f in enumerate(
        [1, 2, 3, 4, 5, 6, 7, 8, 7, 1, 1, 1], start=1
    )}
    data = {
        "days": [{
            "date": target.isoformat(),
            "races": [{
                "raceId": race_id,
                "venue": "札幌",
                "raceNo": 1,
                "horseCount": 12,
                "horseFrames": bad,
                "prediction": {"axes": [7, 9], "opponents": [4, 1, 8, 2]},
                "modelMeta": {"version": updater.MODEL_VERSION},
                "status": "pending",
            }],
        }]
    }
    diagnostics = {"races": []}
    original_discover = legacy.discover_race_ids
    original_fetch = legacy.fetch_entries
    try:
        legacy.discover_race_ids = lambda _target: ([race_id], "test")
        legacy.fetch_entries = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("network entry fetch must not be needed for frame-only repair")
        )
        changed = updater.prepare_day(data, target, diagnostics)
    finally:
        legacy.discover_race_ids = original_discover
        legacy.fetch_entries = original_fetch

    assert changed == 1
    repaired = data["days"][0]["races"][0]["horseFrames"]
    assert repaired == legacy.jra_frame_map(12)
    assert diagnostics["races"][0]["status"] == "frame-repaired"


def test_frame_map_validation() -> None:
    correct = legacy.jra_frame_map(12)
    assert legacy.horse_frame_map_complete(correct, 12)

    wrong = dict(correct)
    wrong["12"] = 1
    assert not legacy.horse_frame_map_complete(wrong, 12)

    missing = dict(correct)
    missing.pop("8")
    assert not legacy.horse_frame_map_complete(missing, 12)


if __name__ == "__main__":
    tests = [
        test_known_allocations,
        test_bad_20260822_style_frames_are_repaired,
        test_missing_frames_are_repaired,
        test_partial_runner_parse_is_rejected,
        test_prepare_repairs_existing_bad_frames_without_network,
        test_frame_map_validation,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"OK: {len(tests)} frame-color regression tests passed")
