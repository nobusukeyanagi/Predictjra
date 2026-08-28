#!/usr/bin/env python3
"""Tune and fit the YS4 D2 single-win main-pick model from data/races.json.

Usage:
    python scripts/optimize_single_win_d2.py

Outputs (default):
    data/single_win_d2_model.joblib
    data/single_win_d2_metrics.json

Validation design
-----------------
1. Build race/horse feature rows only from persisted pre-race indexDetail values.
2. For every target date, train only on dates strictly earlier than that date.
3. Produce out-of-fold (OOF) P(win), P(top3), and expected payout for that date.
4. Split OOF dates chronologically: first 70% tune policy, final 30% untouched holdout.
5. Grid-search only on the tuning portion. Holdout is reported once with the fixed policy.
6. Fit the final live model on all completed historical races after validation.

The optimizer never edits production prediction logic or races.json.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from joblib import dump as joblib_dump

from single_win_d2 import (
    D2Model,
    FEATURE_COLS,
    MODEL_VERSION,
    Policy,
    choose_main,
    choose_second,
    legacy_fallback_scores,
    race_feature_rows,
    rows_from_history,
    selected_set,
    trifecta_return,
)

JST = ZoneInfo("Asia/Tokyo")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/races.json"))
    p.add_argument("--model-out", type=Path, default=Path("data/single_win_d2_model.joblib"))
    p.add_argument("--metrics-out", type=Path, default=Path("data/single_win_d2_metrics.json"))
    p.add_argument("--holdout-ratio", type=float, default=0.30)
    p.add_argument("--min-train-races", type=int, default=180)
    p.add_argument("--goal-roi", type=float, default=100.0)
    p.add_argument("--no-model-write", action="store_true")
    return p.parse_args()


def race_map(data: dict) -> dict[str, dict]:
    out = {}
    for day in data.get("days", []) or []:
        date_s = str(day.get("date") or "")
        for race in day.get("races", []) or []:
            if race.get("predictionDisabled") is True:
                continue
            rid = str(race.get("raceId") or "")
            if rid:
                out[rid] = {"date": date_s, "race": race}
    return out


def group_rows(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        out[str(row["race_id"])].append(row)
    return out


def _win_return(race: dict, main: int) -> int:
    total = 0
    for item in ((race.get("result") or {}).get("winPayouts", []) or []):
        if int(main) in [int(x) for x in (item.get("horses") or [])]:
            total += int(item.get("payout") or 0)
    return total


def _top3_hit(race: dict, main: int) -> bool:
    top3 = set()
    for group in (((race.get("result") or {}).get("places") or [])[:3]):
        top3.update(int(x) for x in group)
    return int(main) in top3


def _tri_stake(race: dict) -> int:
    p = race.get("prediction") or {}
    return len(p.get("opponents", []) or []) * 600


def _baseline_metrics(records: list[dict], rmap: dict[str, dict]) -> dict:
    rows = []
    for rec in records:
        rid = rec["race_id"]
        race = rmap[rid]["race"]
        p = race.get("prediction") or {}
        if not (p.get("axes") or []):
            continue
        main = int(p["axes"][0])
        win_return = _win_return(race, main)
        tri_return = int(race.get("payout") or 0)
        rows.append({
            "date": rec["date"],
            "race_id": rid,
            "main": main,
            "win_return": win_return,
            "top3": int(_top3_hit(race, main)),
            "tri_return": tri_return,
            "tri_stake": _tri_stake(race),
        })
    return aggregate(rows)


def evaluate_policy(policy: Policy, records: list[dict], rmap: dict[str, dict]) -> tuple[dict, list[dict]]:
    rows = []
    for rec in records:
        rid = rec["race_id"]
        race = rmap[rid]["race"]
        selected = selected_set(race)
        if len(selected) < 2:
            continue
        scored = rec["scored"]
        main = choose_main(scored, selected, policy)
        second = choose_second(selected, main, scored)
        win_return = _win_return(race, main)
        tri_return = trifecta_return(race, selected, main, second)
        rows.append({
            "date": rec["date"],
            "race_id": rid,
            "main": main,
            "second": second,
            "win_return": win_return,
            "top3": int(_top3_hit(race, main)),
            "tri_return": tri_return,
            "tri_stake": _tri_stake(race),
        })
    return aggregate(rows), rows


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {
            "races": 0,
            "winReturn": 0,
            "winStake": 0,
            "winRecoveryRate": 0.0,
            "wins": 0,
            "winHitRate": 0.0,
            "top3Hits": 0,
            "top3Rate": 0.0,
            "trifectaReturn": 0,
            "trifectaStake": 0,
            "trifectaRecoveryRate": 0.0,
            "winsorizedWinRecoveryRate50x": 0.0,
            "top3WinReturnShare": 0.0,
        }
    win_returns = [int(r["win_return"]) for r in rows]
    win_return = sum(win_returns)
    stake = n * 100
    tri_return = sum(int(r["tri_return"]) for r in rows)
    tri_stake = sum(int(r["tri_stake"]) for r in rows)
    wins = sum(1 for x in win_returns if x > 0)
    top3 = sum(int(r["top3"]) for r in rows)
    winsor = sum(min(x, 5000) for x in win_returns)
    biggest = sum(sorted([x for x in win_returns if x > 0], reverse=True)[:3])
    return {
        "races": n,
        "winReturn": win_return,
        "winStake": stake,
        "winRecoveryRate": round(win_return / stake * 100.0, 2),
        "wins": wins,
        "winHitRate": round(wins / n * 100.0, 2),
        "top3Hits": top3,
        "top3Rate": round(top3 / n * 100.0, 2),
        "trifectaReturn": tri_return,
        "trifectaStake": tri_stake,
        "trifectaRecoveryRate": round(tri_return / tri_stake * 100.0, 2) if tri_stake else 0.0,
        "winsorizedWinRecoveryRate50x": round(winsor / stake * 100.0, 2),
        "top3WinReturnShare": round(biggest / win_return * 100.0, 2) if win_return else 0.0,
    }


def monthly(rows: list[dict]) -> dict[str, dict]:
    by_month: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_month[str(r["date"])[:7]].append(r)
    return {m: aggregate(rs) for m, rs in sorted(by_month.items())}


def oof_predictions(rows: list[dict], min_train_races: int) -> tuple[list[dict], dict]:
    by_race = group_rows(rows)
    dates = sorted({str(r["date"]) for r in rows})
    race_dates = {rid: race_rows[0]["date"] for rid, race_rows in by_race.items()}
    records: list[dict] = []
    fit_dates = 0
    fallback_dates = 0

    for date_s in dates:
        train_rows = [r for r in rows if str(r["date"]) < date_s]
        train_races = len({r["race_id"] for r in train_rows})
        model = None
        if train_races >= min_train_races:
            try:
                model = D2Model().fit(train_rows)
                fit_dates += 1
            except ValueError:
                model = None
        if model is None:
            fallback_dates += 1

        target_rids = sorted(rid for rid, d in race_dates.items() if d == date_s)
        for rid in target_rids:
            race_rows = by_race[rid]
            scored = model.score_race(race_rows) if model is not None else legacy_fallback_scores(race_rows)
            records.append({
                "date": date_s,
                "race_id": rid,
                "scored": scored,
                "usedModel": model is not None,
                "trainRaces": train_races,
            })
    return records, {
        "dates": len(dates),
        "modelFitDates": fit_dates,
        "fallbackDates": fallback_dates,
        "firstDate": dates[0] if dates else None,
        "lastDate": dates[-1] if dates else None,
    }


def policy_grid():
    for values in itertools.product(
        (2, 3, 4, 5, 7),            # top_k
        (4.0, 6.0, 8.0, 10.0, 12.0),# max_total_gap
        (0.35, 0.50, 0.65),         # min_win_ratio
        (0.35, 0.50, 0.65),         # min_top3_ratio
        (0.0, 0.20, 0.40),          # top3_power
        (0.0, 0.15, 0.30),          # ability_power
        (0.0, 0.15),                # legacy_power
    ):
        yield Policy(*values)


def tune_policy(records: list[dict], rmap: dict[str, dict], baseline: dict) -> tuple[Policy, dict, list[dict], int]:
    best = None
    best_tuple = None
    best_rows = None
    tested = 0
    for policy in policy_grid():
        metrics, rows = evaluate_policy(policy, records, rmap)
        tested += 1
        # Stability constraints: do not buy a headline ROI by destroying reliability/trifecta.
        if metrics["top3Rate"] + 3.0 < baseline["top3Rate"]:
            continue
        if metrics["trifectaRecoveryRate"] + 5.0 < baseline["trifectaRecoveryRate"]:
            continue
        if metrics["top3WinReturnShare"] > 45.0:
            continue
        # Robust ROI is primary; raw ROI is secondary. This resists one giant payout.
        key = (
            metrics["winsorizedWinRecoveryRate50x"],
            metrics["winRecoveryRate"],
            metrics["top3Rate"],
            metrics["trifectaRecoveryRate"],
            -policy.max_total_gap,
        )
        if best_tuple is None or key > best_tuple:
            best_tuple = key
            best = policy
            best_rows = rows
    if best is None:
        # Conservative fallback should constraints be impossible on a short history.
        best = Policy()
        metrics, best_rows = evaluate_policy(best, records, rmap)
        return best, metrics, best_rows, tested
    metrics, _ = evaluate_policy(best, records, rmap)
    return best, metrics, best_rows or [], tested


def main() -> None:
    args = parse_args()
    data = json.loads(args.data.read_text(encoding="utf-8"))
    rows = rows_from_history(data)
    rmap = race_map(data)
    if not rows or not rmap:
        raise SystemExit("No usable historical races found")

    oof, oof_meta = oof_predictions(rows, args.min_train_races)
    modeled = [r for r in oof if r["usedModel"]]
    dates = sorted({r["date"] for r in modeled})
    if len(dates) < 6:
        raise SystemExit(f"Too few model-scored dates for holdout: {len(dates)}")

    split_at = max(1, min(len(dates) - 1, int(math.floor(len(dates) * (1.0 - args.holdout_ratio)))))
    tune_dates = set(dates[:split_at])
    holdout_dates = set(dates[split_at:])
    tune_records = [r for r in modeled if r["date"] in tune_dates]
    holdout_records = [r for r in modeled if r["date"] in holdout_dates]

    baseline_tune = _baseline_metrics(tune_records, rmap)
    baseline_holdout = _baseline_metrics(holdout_records, rmap)
    best_policy, tuned_metrics, tuned_rows, tested = tune_policy(tune_records, rmap, baseline_tune)
    holdout_metrics, holdout_rows = evaluate_policy(best_policy, holdout_records, rmap)
    all_metrics, all_rows = evaluate_policy(best_policy, modeled, rmap)
    baseline_all = _baseline_metrics(modeled, rmap)

    final_model = D2Model().fit(rows)
    trained_through = max(str(r["date"]) for r in rows)
    model_payload = {
        "version": MODEL_VERSION,
        "featureCols": FEATURE_COLS,
        "trainedThrough": trained_through,
        "policy": best_policy.to_dict(),
        "model": final_model,
    }
    if not args.no_model_write:
        args.model_out.parent.mkdir(parents=True, exist_ok=True)
        joblib_dump(model_payload, args.model_out)

    report = {
        "version": MODEL_VERSION,
        "generatedAt": datetime.now(JST).isoformat(timespec="seconds"),
        "goalWinRecoveryRate": float(args.goal_roi),
        "leakagePolicy": {
            "currentRaceOdds": False,
            "actualPopularity": False,
            "bodyweightOrChange": False,
            "sameDayTrainingLabels": False,
            "trainingRule": "date < target_date only",
        },
        "featureCols": FEATURE_COLS,
        "history": {
            "horseRows": len(rows),
            "races": len(rmap),
            "trainedThrough": trained_through,
            **oof_meta,
        },
        "validation": {
            "tuneDateRange": [min(tune_dates), max(tune_dates)] if tune_dates else [],
            "holdoutDateRange": [min(holdout_dates), max(holdout_dates)] if holdout_dates else [],
            "policyCandidatesTested": tested,
            "baselineTune": baseline_tune,
            "d2Tune": tuned_metrics,
            "baselineHoldout": baseline_holdout,
            "d2Holdout": holdout_metrics,
            "baselineAllModelScored": baseline_all,
            "d2AllModelScored": all_metrics,
            "d2TuneMonthly": monthly(tuned_rows),
            "d2HoldoutMonthly": monthly(holdout_rows),
            "d2AllMonthly": monthly(all_rows),
        },
        "selectedPolicy": best_policy.to_dict(),
        "goalReachedOnUntouchedHoldout": holdout_metrics["winRecoveryRate"] >= float(args.goal_roi),
        "adoptionRule": (
            "Adopt only if untouched holdout improves win ROI without >3pp top3-rate loss, "
            ">5pp trifecta-ROI loss, or >45% concentration in the three biggest win returns."
        ),
    }
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "version": MODEL_VERSION,
        "policy": best_policy.to_dict(),
        "baselineHoldout": baseline_holdout,
        "d2Holdout": holdout_metrics,
        "goalReached": report["goalReachedOnUntouchedHoldout"],
        "modelOut": None if args.no_model_write else str(args.model_out),
        "metricsOut": str(args.metrics_out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
