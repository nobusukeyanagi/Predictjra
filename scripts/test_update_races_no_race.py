#!/usr/bin/env python3
"""Regression tests for scheduled no-race-day handling."""
from __future__ import annotations

from datetime import date

import run_update_races_safe as safe
import update_races as legacy

TARGET = date(2026, 9, 2)
RACE_ID = "202605040101"


def test_empty_day_and_empty_discovery_is_normal_no_race() -> None:
    original_load = legacy.load_data
    original_find = legacy.find_day
    original_discover = legacy.discover_race_ids
    try:
        legacy.load_data = lambda: {"days": []}
        legacy.find_day = lambda *_args, **_kwargs: None
        legacy.discover_race_ids = lambda _target: ([], "none")
        skip, source = safe.should_skip_no_race_result(TARGET)
    finally:
        legacy.load_data = original_load
        legacy.find_day = original_find
        legacy.discover_race_ids = original_discover

    assert skip is True
    assert source == "none"
    diagnostics = safe.build_no_race_diagnostics(TARGET, source)
    assert diagnostics["success"] is True
    assert diagnostics["publishBlocked"] is False
    assert diagnostics["noRaceDay"] is True
    assert diagnostics["changedRaces"] == 0
    assert diagnostics["unresolvedRaceIds"] == []


def test_existing_prepared_day_never_skips() -> None:
    original_load = legacy.load_data
    original_find = legacy.find_day
    original_discover = legacy.discover_race_ids
    try:
        legacy.load_data = lambda: {"days": []}
        legacy.find_day = lambda *_args, **_kwargs: {"races": [{"raceId": RACE_ID}]}
        legacy.discover_race_ids = lambda _target: (_ for _ in ()).throw(
            AssertionError("discovery should not run when prepared data exists")
        )
        skip, source = safe.should_skip_no_race_result(TARGET)
    finally:
        legacy.load_data = original_load
        legacy.find_day = original_find
        legacy.discover_race_ids = original_discover

    assert skip is False
    assert source == "prepared-data"


def test_race_day_without_prepared_data_still_fails_closed() -> None:
    original_load = legacy.load_data
    original_find = legacy.find_day
    original_discover = legacy.discover_race_ids
    try:
        legacy.load_data = lambda: {"days": []}
        legacy.find_day = lambda *_args, **_kwargs: None
        legacy.discover_race_ids = lambda _target: ([RACE_ID], "netkeiba")
        skip, source = safe.should_skip_no_race_result(TARGET)
    finally:
        legacy.load_data = original_load
        legacy.find_day = original_find
        legacy.discover_race_ids = original_discover

    assert skip is False
    assert source == "netkeiba"


if __name__ == "__main__":
    tests = [
        test_empty_day_and_empty_discovery_is_normal_no_race,
        test_existing_prepared_day_never_skips,
        test_race_day_without_prepared_data_still_fails_closed,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"OK: {len(tests)} no-race-day regression tests passed")
