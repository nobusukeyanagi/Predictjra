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
from market_history import (
    apply_updates as apply_market_updates,
    enrich_card_with_market_history,
    load_market_history,
    result_updates as market_result_updates,
    save_market_history,
)

JST = legacy.JST


def write_diagnostics(path: str | None, payload: dict) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def is_debut_race_record(race: dict | None) -> bool:
    if not race:
        return False
    if race.get("predictionDisabledReason") == "新馬戦":
        return True
    title = str(((race.get("modelMeta") or {}).get("indexDetail") or {}).get("title") or "")
    race_name = str(race.get("raceName") or "")
    return "新馬" in title or "新馬" in race_name


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
    debut_ids: set[str] = set()
    market_payload = load_market_history()

    for race_id in ids:
        existing_race = existing.get(race_id)
        current_prediction = bool(
            existing_race
            and existing_race.get("prediction")
            and existing_race.get("modelMeta", {}).get("version") == MODEL_VERSION
        )
        current_debut_placeholder = bool(
            existing_race
            and is_debut_race_record(existing_race)
            and existing_race.get("predictionDisabled") is True
        )
        if current_prediction or current_debut_placeholder:
            field_size = int(existing_race.get("horseCount") or 0)
            if legacy.horse_frame_map_complete(
                existing_race.get("horseFrames"), field_size
            ):
                if current_debut_placeholder:
                    debut_ids.add(race_id)
                print(
                    f"SKIP {race_id}: already prepared "
                    f"({'新馬結果表示のみ' if current_debut_placeholder else MODEL_VERSION})"
                )
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
                if current_debut_placeholder:
                    debut_ids.add(race_id)
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
            race_name = str(rich_card.get("raceName") or "")

            # 枠番 is a deterministic function of horse number and field size.
            frames = legacy.jra_frame_map(field_size)
            names = {
                str(e["horse"]): e["name"]
                for e in base_entries if e.get("name")
            }
            for e in rich_card["entries"]:
                if e.get("name"):
                    names[str(e["no"])] = e["name"]

            market_horse_ids = {
                str(int(e["no"])): str(e.get("horseId") or "")
                for e in rich_card.get("entries", []) if e.get("horseId")
            }

            if "新馬" in race_name:
                # New-race races remain visible as result-only rows.  No prediction,
                # selection, investment or recovery-rate denominator is created.
                debut_ids.add(race_id)
                race = dict(existing_race or {})
                race.update({
                    "raceId": race_id,
                    "venue": venue,
                    "raceNo": race_no,
                    "raceName": race_name,
                    "horseCount": field_size,
                    "horseFrames": frames,
                    "horseNames": names,
                    "prediction": None,
                    "danger": [],
                    "predictionDisabled": True,
                    "predictionDisabledReason": "新馬戦",
                    "modelMeta": {
                        "version": MODEL_VERSION,
                        "marketHorseIds": market_horse_ids,
                        "indexDetail": {
                            "title": f"{venue}{race_no}R {race_name}".strip(),
                            "horseCount": field_size,
                            "horses": [],
                        },
                    },
                    "result": race.get("result"),
                    "status": race.get("status", "pending"),
                    "payout": 0,
                    "trifectaPayouts": race.get("trifectaPayouts", []),
                    "stake": 0,
                    "winReturn": 0,
                    "winStake": 0,
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
                    "status": "debut-result-only",
                    "raceName": race_name,
                    "horses": len(horses),
                    "sources": race["dataSources"],
                })
                print(f"RESULT-ONLY {race_id}: {race_name} (新馬戦・予想なし)")
                continue

            odds_attached = enrich_card_with_market_history(
                rich_card, market_payload, target.isoformat()
            )
            built = build_prediction(rich_card)
            built["modelMeta"]["marketHorseIds"] = market_horse_ids
            built["modelMeta"]["historicalOddsAttached"] = int(odds_attached)
            prediction = built["prediction"]

            race = dict(existing_race or {})
            race.update({
                "raceId": race_id,
                "venue": venue,
                "raceNo": race_no,
                "raceName": race_name,
                "horseCount": field_size,
                "horseFrames": frames,
                "horseNames": names,
                "prediction": prediction,
                "danger": built["danger"],
                "predictionDisabled": False,
                "predictionDisabledReason": "",
                "modelMeta": built["modelMeta"],
                "result": race.get("result"),
                "status": race.get("status", "pending"),
                "payout": int(race.get("payout", 0)),
                "trifectaPayouts": race.get("trifectaPayouts", []),
                "stake": legacy.stake_for_prediction(prediction),
                "winReturn": int(race.get("winReturn", 0)),
                "winStake": 100,
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

    # Accuracy-first: all discovered races must be represented. New-race races are
    # valid only as explicit result-only placeholders; all other races need v54 prediction.
    already_ok = {
        rid for rid, r in existing.items()
        if (
            (
                r.get("modelMeta", {}).get("version") == MODEL_VERSION
                and r.get("prediction")
            )
            or (
                is_debut_race_record(r)
                and r.get("predictionDisabled") is True
            )
        )
        and legacy.horse_frame_map_complete(r.get("horseFrames"), r.get("horseCount"))
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

    staged_map = {r["raceId"]: r for r in staged}
    new_races = []
    for rid in ids:
        if rid in staged_map:
            new_races.append(staged_map[rid])
        elif rid in existing:
            new_races.append(existing[rid])

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
    diagnostics["debutResultOnlyRaceIds"] = sorted(debut_ids)
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
    market_payload = load_market_history()
    pending_market_updates: list[dict] = []

    for original in day.get("races", []):
        race = copy.deepcopy(original)
        race_id = race["raceId"]
        try:
            debut = is_debut_race_record(race) or race.get("predictionDisabled") is True
            prediction = race.get("prediction")
            if not debut and (not prediction or len(prediction.get("axes", [])) != 2):
                raise ValueError("prepared prediction missing or invalid")

            finalized_status = {"result-only"} if debut else {"hit", "miss"}
            stored_result = race.get("result") or {}
            if (
                race.get("status") in finalized_status
                and stored_result
                and stored_result.get("winPayouts")
                and (debut or "winReturn" in race)
            ):
                legacy.validate_result_payload(stored_result, race_id)
                staged_races.append(race)
                pending_market_updates.extend(
                    market_result_updates(race, stored_result, target.isoformat())
                )
                diagnostics["races"].append({
                    "raceId": race_id,
                    "status": "already-finalized",
                    "predictionDisabled": debut,
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
                    changed += 1
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
            race["result"] = result
            race["trifectaPayouts"] = [
                int(t["payout"]) for t in result["trifectas"]
            ]
            race["dataSources"] = {
                **race.get("dataSources", {}),
                "result": result_source,
            }

            if debut:
                race["prediction"] = None
                race["danger"] = []
                race["predictionDisabled"] = True
                race["predictionDisabledReason"] = "新馬戦"
                race["status"] = "result-only"
                race["payout"] = 0
                race["stake"] = 0
                race["winReturn"] = 0
                race["winStake"] = 0
                diagnostics["races"].append({
                    "raceId": race_id,
                    "status": "result-only",
                    "resultSource": result_source,
                })
                print(f"RESULT {race_id}: 新馬戦 result-only source={result_source}")
            else:
                winning = [
                    t for t in result["trifectas"]
                    if legacy.combo_is_covered(prediction, t["horses"])
                ]
                payout = sum(int(t["payout"]) for t in winning)
                main = int((prediction.get("axes") or [0])[0] or 0)
                win_return = sum(
                    int(item["payout"])
                    for item in result.get("winPayouts", [])
                    if main in [int(x) for x in item.get("horses", [])]
                )

                # status remains the legacy trifecta status for backwards compatibility.
                # UI/summary count an "的中" when either winReturn or trifecta payout hits.
                race["status"] = "hit" if winning else "miss"
                race["payout"] = payout
                race["stake"] = legacy.stake_for_prediction(prediction)
                race["winReturn"] = int(win_return)
                race["winStake"] = 100
                diagnostics["races"].append({
                    "raceId": race_id,
                    "status": race["status"],
                    "winReturn": int(win_return),
                    "trifectaReturn": payout,
                    "anyHit": bool(win_return or payout),
                    "resultSource": result_source,
                })
                print(
                    f"RESULT {race_id}: tri={race['status']} winReturn={win_return} "
                    f"trifectaReturn={payout} source={result_source}"
                )

            pending_market_updates.extend(
                market_result_updates(race, result, target.isoformat())
            )
            staged_races.append(race)
            changed += 1
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

    # Fail closed across predicted and result-only new-race rows alike.
    if unresolved:
        diagnostics["publishBlocked"] = True
        raise RuntimeError(
            "result update incomplete; publication blocked for race IDs: "
            + ", ".join(sorted(set(unresolved)))
        )

    market_changed = apply_market_updates(
        market_payload, pending_market_updates, target.isoformat()
    )
    if market_changed:
        save_market_history(market_payload)
    diagnostics["marketHistoryUpdates"] = market_changed

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
