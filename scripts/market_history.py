#!/usr/bin/env python3
"""Durable previous-race market facts used by Predictjra v54.

Only facts from races strictly before the prediction target date are read.  The
file is intentionally separate from races.json so public prediction/result data
can be rebuilt without losing historical win-odds memory.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "market_history.json"
MAX_RUNS_PER_HORSE = 12


def _clean_id(value) -> str:
    s = "".join(ch for ch in str(value or "") if ch.isdigit())
    return s


def load_market_history(path: Path = DEFAULT_PATH) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": "predictjra-market-history-v54", "throughDate": "", "horses": {}}
    if not isinstance(payload, dict) or not isinstance(payload.get("horses"), dict):
        return {"version": "predictjra-market-history-v54", "throughDate": "", "horses": {}}
    return payload


def save_market_history(payload: dict, path: Path = DEFAULT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


def enrich_card_with_market_history(card: dict, payload: dict, target_date: str) -> int:
    """Attach previous win odds to the already parsed five-run histories.

    Matching is horse-id + exact historical date.  Same-day/future records are
    rejected even if the cache was accidentally populated too early.
    """
    horses = payload.get("horses") or {}
    attached = 0
    for entry in card.get("entries", []):
        horse_id = _clean_id(entry.get("horseId"))
        if not horse_id:
            continue
        records = horses.get(horse_id) or []
        by_date = {
            str(r.get("date")): r
            for r in records
            if str(r.get("date") or "") and str(r.get("date")) < target_date
        }
        for run in entry.get("histories", []):
            run_date = str(run.get("date") or "")
            if not run_date or run_date >= target_date:
                continue
            record = by_date.get(run_date)
            if not record:
                continue
            try:
                odds = float(record.get("odds"))
            except (TypeError, ValueError):
                continue
            if odds <= 0:
                continue
            run["odds"] = odds
            attached += 1
    return attached


def result_updates(race: dict, result: dict, target_date: str) -> list[dict]:
    """Convert full-runner market rows from a final result into horse-history updates."""
    horse_ids = ((race.get("modelMeta") or {}).get("marketHorseIds") or {})
    updates = []
    for row in result.get("runnerMarket") or []:
        no = str(int(row.get("horse", 0) or 0))
        horse_id = _clean_id(horse_ids.get(no))
        if not horse_id:
            continue
        try:
            odds = float(row.get("odds"))
        except (TypeError, ValueError):
            continue
        if odds <= 0:
            continue
        updates.append({
            "horseId": horse_id,
            "date": target_date,
            "odds": round(odds, 1),
            "popularity": int(row.get("popularity")) if row.get("popularity") is not None else None,
            "surface": row.get("surface") or "",
            "distance": row.get("distance"),
        })
    return updates


def apply_updates(payload: dict, updates: list[dict], through_date: str) -> int:
    horses = payload.setdefault("horses", {})
    changed = 0
    for item in updates:
        horse_id = _clean_id(item.get("horseId"))
        day = str(item.get("date") or "")
        if not horse_id or not day:
            continue
        records = list(horses.get(horse_id) or [])
        before = json.dumps(records, sort_keys=True, ensure_ascii=False)
        records = [r for r in records if str(r.get("date")) != day]
        records.append({
            "date": day,
            "odds": item.get("odds"),
            "popularity": item.get("popularity"),
            "surface": item.get("surface") or "",
            "distance": item.get("distance"),
        })
        records.sort(key=lambda r: str(r.get("date") or ""), reverse=True)
        records = records[:MAX_RUNS_PER_HORSE]
        horses[horse_id] = records
        after = json.dumps(records, sort_keys=True, ensure_ascii=False)
        if after != before:
            changed += 1
    if through_date and str(through_date) > str(payload.get("throughDate") or ""):
        payload["throughDate"] = str(through_date)
    payload["version"] = "predictjra-market-history-v54"
    return changed
