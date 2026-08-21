#!/usr/bin/env python3
"""Production updater for Predictjra.

Race discovery, base entries, results, payouts, and persistence helpers come from
update_races.py.  Pre-race prediction is built by predict_engine.py, which normalizes
the live card and calls prediction_logic_production.py, the explicitly applied production
snapshot. Historical Rebuild/validate uses prediction_logic_candidate.py instead.
"""
from __future__ import annotations

import argparse
import copy
import json
import traceback
from datetime import date, datetime, timedelta
from pathlib import Path

import update_races as legacy
from predict_engine import MODEL_VERSION, build_prediction, fetch_rich_card

JST = legacy.JST


def write_diagnostics(path: str | None, payload: dict) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare_day(data: dict, target: date, diagnostics: dict) -> int:
    ids, discovery_source = legacy.discover_race_ids(target)
    diagnostics["discoveredRaceIds"] = ids
    diagnostics["discoverySource"] = discovery_source
    if not ids:
        print(f"No JRA race IDs found for {target}; no changes.")
        return 0

    day = legacy.find_day(data, target, create=True)
    existing = {r["raceId"]: r for r in day["races"]}
    staged: list[dict] = []
    errors: list[dict] = []

    for race_id in ids:
        existing_race = existing.get(race_id)
        current_prediction = bool(
            existing_race
            and existing_race.get("prediction")
            and existing_race.get("modelMeta", {}).get("version") == MODEL_VERSION
        )
        if current_prediction:
            field_size = int(existing_race.get("horseCount") or 0)
            if legacy.horse_frame_map_complete(
                existing_race.get("horseFrames"), field_size
            ):
                print(f"SKIP {race_id}: already prepared with {MODEL_VERSION}")
                continue
            if 1 <= field_size <= 18:
                repaired = dict(existing_race)
                before = dict(existing_race.get("horseFrames") or {})
                repaired["horseFrames"] = legacy.jra_frame_map(field_size)
                repaired["dataSources"] = {
                    **repaired.get("dataSources", {}),
                    "frames": "jra-deterministic-repair",
                }
                staged.append(repaired)
                diagnostics["races"].append({
                    "raceId": race_id,
                    "status": "frame-repaired",
                    "horseCount": field_size,
                    "oldHorseFrames": before,
                    "horseFrames": repaired["horseFrames"],
                })
                print(
                    f"REPAIR {race_id}: corrected horseFrames from deterministic "
                    f"JRA allocation ({field_size} runners)"
                )
                continue

        venue, race_no = legacy.race_meta(race_id)
        try:
            base_entries, entry_source = legacy.fetch_entries(race_id, target)
            horses = [int(e["horse"]) for e in base_entries]
            if len(horses) < 5:
                raise ValueError(f"only {len(horses)} horse numbers parsed")
            field_size = max(horses)
            if horses != list(range(1, field_size + 1)):
                raise ValueError(f"incomplete horse-number sequence: {horses}")

            rich_card, rich_source = fetch_rich_card(
                race_id,
                base_entries,
                legacy.request_html,
                legacy.selenium_html,
            )
            rich_card["targetDate"] = target.isoformat()
            built = build_prediction(rich_card)
            prediction = built["prediction"]

            # 枠番 is a deterministic function of horse number and field size.
            # Do not let a CSS/HTML parser from any source overwrite it.
            frames = legacy.jra_frame_map(field_size)
            names = {
                str(e["horse"]): e["name"]
                for e in base_entries if e.get("name")
            }
            for e in rich_card["entries"]:
                if e.get("name"):
                    names[str(e["no"])] = e["name"]

            race = dict(existing_race or {})
            race.update({
                "raceId": race_id,
                "venue": venue,
                "raceNo": race_no,
                "horseCount": field_size,
                "horseFrames": frames,
                "horseNames": names,
                "prediction": prediction,
                "danger": built["danger"],
                "modelMeta": built["modelMeta"],
                "result": race.get("result"),
                "status": race.get("status", "pending"),
                "payout": int(race.get("payout", 0)),
                "trifectaPayouts": race.get("trifectaPayouts", []),
                "stake": legacy.stake_for_prediction(prediction),
                "dataSources": {
                    **race.get("dataSources", {}),
                    "discovery": discovery_source,
                    "entries": entry_source,
                    "frames": "jra-deterministic",
                    "indexDetail": rich_source,
                },
            })
            staged.append(race)
            diagnostics["races"].append({
                "raceId": race_id,
                "status": "prepared",
                "horses": len(horses),
                "danger": built["danger"],
                "prediction": prediction,
                "qualityWarnings": built["indexDetail"].get("qualityWarnings", []),
                "sources": race["dataSources"],
            })
            print(
                f"PREPARED {target} {venue}{race_no}R {race_id} "
                f"index={rich_source} danger={built['danger'][0]}"
            )
        except Exception as exc:  # noqa: BLE001
            error = {
                "raceId": race_id,
                "status": "error",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            errors.append(error)
            diagnostics["races"].append(error)
            print(f"ERROR prepare {race_id}: {exc}")

    diagnostics["prepareErrors"] = errors

    # Accuracy-first: do not publish a newly-generated partial card set.
    # Races already prepared with the current MODEL_VERSION are accepted; every
    # remaining discovered race must stage successfully before the day is saved.
    already_ok = {
        rid for rid, r in existing.items()
        if (
            r.get("modelMeta", {}).get("version") == MODEL_VERSION
            and r.get("prediction")
            and legacy.horse_frame_map_complete(
                r.get("horseFrames"), r.get("horseCount")
            )
        )
    }
    staged_ids = {r["raceId"] for r in staged}
    missing = set(ids) - already_ok - staged_ids
    if missing:
        diagnostics["publishBlocked"] = True
        diagnostics["missingRaceIds"] = sorted(missing)
        raise RuntimeError(
            "prepare incomplete; publication blocked for race IDs: "
            + ", ".join(sorted(missing))
        )

    # Only mutate the day after all target races are valid.
    staged_map = {r["raceId"]: r for r in staged}
    new_races = []
    for rid in ids:
        if rid in staged_map:
            new_races.append(staged_map[rid])
        elif rid in existing:
            new_races.append(existing[rid])

    # Final publication invariant: no pending/prepared race may reach races.json
    # with a missing or contradictory frame map.
    bad_frames = [
        r.get("raceId", "?") for r in new_races
        if not legacy.horse_frame_map_complete(
            r.get("horseFrames"), r.get("horseCount")
        )
    ]
    if bad_frames:
        diagnostics["publishBlocked"] = True
        diagnostics["frameValidationErrors"] = bad_frames
        raise RuntimeError(
            "horseFrames validation failed; publication blocked for race IDs: "
            + ", ".join(bad_frames)
        )

    day["races"] = new_races
    diagnostics["publishBlocked"] = False
    return len(staged)


def result_day(data: dict, target: date, diagnostics: dict) -> int:
    day = legacy.find_day(data, target, create=False)
    if not day or not day.get("races"):
        diagnostics["publishBlocked"] = True
        raise RuntimeError(
            f"no prepared races for {target}; result mode never creates predictions"
        )

    changed = 0
    staged_races: list[dict] = []
    unresolved: list[str] = []
    errors: list[dict] = []
    cancelled: list[dict] = []

    for original in day.get("races", []):
        race = copy.deepcopy(original)
        race_id = race["raceId"]
        try:
            prediction = race.get("prediction")
            if not prediction or len(prediction.get("axes", [])) != 2:
                raise ValueError("prepared prediction missing or invalid")

            # A rerun after an earlier successful/partial attempt should not depend on
            # fetching already-finalized races again. Validate the stored payload before
            # accepting it as resolved.
            if race.get("status") in {"hit", "miss"} and race.get("result"):
                legacy.validate_result_payload(race["result"], race_id)
                staged_races.append(race)
                diagnostics["races"].append({
                    "raceId": race_id,
                    "status": "already-finalized",
                })
                continue

            result, result_source = legacy.fetch_result(race_id, target)
            if not result:
                is_cancelled, cancel_source = legacy.confirm_cancelled_race(race_id, target)
                if is_cancelled:
                    cancelled.append({
                        "raceId": race_id,
                        "source": cancel_source,
                    })
                    diagnostics["races"].append({
                        "raceId": race_id,
                        "status": "cancelled",
                        "cancellationSource": cancel_source,
                    })
                    changed += 1  # removed from the conducted-race set
                    print(f"CANCELLED {race_id}: confirmed by {cancel_source}")
                    continue

                unresolved.append(race_id)
                diagnostics["races"].append({
                    "raceId": race_id,
                    "status": "result-unresolved",
                })
                print(f"UNRESOLVED {race_id}: no result and no cancellation evidence")
                continue

            legacy.validate_result_payload(result, race_id)
            winning = [
                t for t in result["trifectas"]
                if legacy.combo_is_covered(prediction, t["horses"])
            ]
            payout = sum(int(t["payout"]) for t in winning)

            race["result"] = result
            race["status"] = "hit" if winning else "miss"
            race["payout"] = payout
            race["trifectaPayouts"] = [
                int(t["payout"]) for t in result["trifectas"]
            ]
            race["stake"] = legacy.stake_for_prediction(prediction)
            race["dataSources"] = {
                **race.get("dataSources", {}),
                "result": result_source,
            }
            staged_races.append(race)
            changed += 1

            main = int((prediction.get("axes") or [0])[0] or 0)
            first = [int(x) for x in (result.get("places") or [[]])[0]]
            diagnostics["races"].append({
                "raceId": race_id,
                "status": race["status"],
                "trifectaReturn": payout,
                "mainHorseWon": main in first,
                "resultSource": result_source,
            })
            print(
                f"RESULT {race_id}: {race['status']} payout={payout} "
                f"mainWin={main in first} source={result_source}"
            )
        except Exception as exc:  # noqa: BLE001
            unresolved.append(race_id)
            error = {
                "raceId": race_id,
                "status": "result-error",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            errors.append(error)
            diagnostics["races"].append(error)
            print(f"ERROR result {race_id}: {exc}")

    diagnostics["cancelledRaces"] = cancelled
    diagnostics["resultErrors"] = errors
    diagnostics["unresolvedRaceIds"] = sorted(set(unresolved))

    # Fail closed: do not mutate/save even one result unless every prepared race is
    # either finalized or explicitly confirmed as non-conducted.
    if unresolved:
        diagnostics["publishBlocked"] = True
        raise RuntimeError(
            "result update incomplete; publication blocked for race IDs: "
            + ", ".join(sorted(set(unresolved)))
        )

    day["races"] = staged_races
    diagnostics["publishBlocked"] = False
    diagnostics["resolvedRaceCount"] = len(staged_races)
    return changed


def resolve_target(mode: str, explicit: str | None) -> date:
    if explicit:
        return date.fromisoformat(explicit)
    today = datetime.now(JST).date()
    return today + timedelta(days=1) if mode == "prepare" else today


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["prepare", "result"], required=True)
    parser.add_argument("--date", help="YYYY-MM-DD. Omit for scheduled default.")
    parser.add_argument("--diagnostics", help="Write diagnostic JSON here.")
    args = parser.parse_args()

    target = resolve_target(args.mode, args.date)
    diagnostics = {
        "version": MODEL_VERSION,
        "mode": args.mode,
        "target": target.isoformat(),
        "startedAt": datetime.now(JST).isoformat(timespec="seconds"),
        "races": [],
    }

    data = legacy.load_data()
    print(f"mode={args.mode} target={target} model={MODEL_VERSION}")

    try:
        changed = (
            prepare_day(data, target, diagnostics)
            if args.mode == "prepare"
            else result_day(data, target, diagnostics)
        )
        if changed:
            legacy.save_data(data)
            print(f"Saved {legacy.DATA_PATH} ({changed} race updates)")
        else:
            print("No data changes.")
        diagnostics["changedRaces"] = changed
        diagnostics["success"] = True
        return 0
    except Exception as exc:  # noqa: BLE001
        diagnostics["success"] = False
        diagnostics["fatalError"] = str(exc)
        diagnostics["fatalTraceback"] = traceback.format_exc()
        print(f"FATAL: {exc}")
        return 1
    finally:
        diagnostics["finishedAt"] = datetime.now(JST).isoformat(timespec="seconds")
        write_diagnostics(args.diagnostics, diagnostics)


if __name__ == "__main__":
    raise SystemExit(main())
