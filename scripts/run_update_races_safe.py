#!/usr/bin/env python3
"""Safe scheduled wrapper around update_races_v2.

The production updater intentionally fails closed when result mode has no prepared
race card.  That is correct on a JRA race day, but a scheduled result run also
occurs on genuine no-race days.  This wrapper distinguishes those two cases:

* no prepared card + no discoverable JRA races -> successful no-op
* no prepared card + discoverable JRA races -> delegate and fail closed as before

Prepare mode is delegated unchanged; update_races_v2 already treats a zero-race
future date as a successful no-op.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime

import update_races as legacy
import update_races_v2 as updater


def build_no_race_diagnostics(target, discovery_source: str) -> dict:
    now = datetime.now(updater.JST).isoformat(timespec="seconds")
    return {
        "version": updater.MODEL_VERSION,
        "mode": "result",
        "target": target.isoformat(),
        "startedAt": now,
        "finishedAt": now,
        "races": [],
        "discoveredRaceIds": [],
        "discoverySource": discovery_source,
        "changedRaces": 0,
        "resolvedRaceCount": 0,
        "unresolvedRaceIds": [],
        "publishBlocked": False,
        "noRaceDay": True,
        "success": True,
    }


def should_skip_no_race_result(target) -> tuple[bool, str]:
    """Return True only when there is no prepared card and discovery finds no races."""
    data = legacy.load_data()
    day = legacy.find_day(data, target, create=False)
    if day and day.get("races"):
        return False, "prepared-data"

    race_ids, source = legacy.discover_race_ids(target)
    if race_ids:
        return False, source
    return True, source


def delegate_to_production(args: argparse.Namespace) -> int:
    argv = ["update_races_v2.py", "--mode", args.mode]
    if args.date:
        argv.extend(["--date", args.date])
    if args.diagnostics:
        argv.extend(["--diagnostics", args.diagnostics])

    old_argv = sys.argv
    try:
        sys.argv = argv
        return updater.main()
    finally:
        sys.argv = old_argv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["prepare", "result"], required=True)
    parser.add_argument("--date", help="YYYY-MM-DD. Omit for scheduled default.")
    parser.add_argument("--diagnostics", help="Write diagnostic JSON here.")
    args = parser.parse_args()

    target = updater.resolve_target(args.mode, args.date)

    if args.mode == "result":
        skip, source = should_skip_no_race_result(target)
        if skip:
            diagnostics = build_no_race_diagnostics(target, source)
            updater.write_diagnostics(args.diagnostics, diagnostics)
            print(
                f"NO-RACE {target}: no JRA races discovered; "
                "scheduled result update completed as a normal no-op."
            )
            return 0

    return delegate_to_production(args)


if __name__ == "__main__":
    raise SystemExit(main())
