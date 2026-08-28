#!/usr/bin/env python3
"""Chronological OOF optimizer for Predictjra D3 single-win reranking.

This script is experimental: it never edits races.json or production logic.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from joblib import dump as joblib_dump

from single_win_d2 import (
    choose_second,
    race_feature_rows,
    rows_from_history,
    selected_set,
    trifecta_return,
)
from single_win_d3 import (
    ABILITY_FEATURE_COLS,
    D3Model,
    D3Policy,
    MODEL_VERSION,
    choose_main,
    legacy_fallback_scores,
)

JST = ZoneInfo("Asia/Tokyo")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/races.json"))
    p.add_argument("--model-out", type=Path, default=Path("data/single_win_d3_model.joblib"))
    p.add_argument("--metrics-out", type=Path, default=Path("data/single_win_d3_metrics.json"))
    p.add_argument("--holdout-ratio", type=float, default=0.30)
    p.add_argument("--min-train-races", type=int, default=180)
    p.add_argument("--goal-roi", type=float, default=80.0)
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


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {
            "races": 0, "winReturn": 0, "winStake": 0, "winRecoveryRate": 0.0,
            "wins": 0, "winHitRate": 0.0, "top3Hits": 0, "top3Rate": 0.0,
            "trifectaReturn": 0, "trifectaStake": 0, "trifectaRecoveryRate": 0.0,
            "winsorizedWinRecoveryRate50x": 0.0, "top3WinReturnShare": 0.0,
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


def robustness(rows: list[dict]) -> dict:
    mm = monthly(rows)
    rois = [v["winsorizedWinRecoveryRate50x"] for v in mm.values() if v["races"] >= 50]
    if not rois:
        return {"monthlyMedianROI": 0.0, "monthlyQ25ROI": 0.0, "monthlyMinROI": 0.0, "robustScore": 0.0}
    overall = aggregate(rows)["winsorizedWinRecoveryRate50x"]
    median = float(np.median(rois))
    q25 = float(np.quantile(rois, 0.25))
    minimum = float(min(rois))
    # Reward total return while explicitly punishing one bad month/regime collapse.
    robust = 0.50 * overall + 0.30 * q25 + 0.20 * median
    return {
        "monthlyMedianROI": round(median, 2),
        "monthlyQ25ROI": round(q25, 2),
        "monthlyMinROI": round(minimum, 2),
        "robustScore": round(robust, 4),
    }


def block_robustness(rows: list[dict], blocks: int = 5) -> dict:
    """Contiguous chronological robustness independent of calendar-month boundaries."""
    if not rows:
        return {"blockQ25ROI": 0.0, "blockMinROI": 0.0, "blocks": []}
    ordered = sorted(rows, key=lambda r: (str(r.get("date", "")), str(r.get("race_id", ""))))
    n = len(ordered)
    if n < 150:
        m = aggregate(ordered)
        return {
            "blockQ25ROI": m["winsorizedWinRecoveryRate50x"],
            "blockMinROI": m["winsorizedWinRecoveryRate50x"],
            "blocks": [m],
        }
    count = max(3, min(int(blocks), 6))
    chunks = []
    for i in range(count):
        a = round(n * i / count)
        b = round(n * (i + 1) / count)
        if b > a:
            chunks.append(aggregate(ordered[a:b]))
    rois = [c["winsorizedWinRecoveryRate50x"] for c in chunks]
    return {
        "blockQ25ROI": round(float(np.quantile(rois, 0.25)), 2) if rois else 0.0,
        "blockMinROI": round(float(min(rois)), 2) if rois else 0.0,
        "blocks": chunks,
    }


def floor80_score(metrics: dict, block: dict, target: float = 80.0) -> float:
    """Higher is better; explicitly penalizes falling short of the 80% floor."""
    overall = float(metrics.get("winsorizedWinRecoveryRate50x", 0.0))
    rb = metrics.get("robustness") or {}
    monthly_q25 = float(rb.get("monthlyQ25ROI", 0.0))
    block_q25 = float(block.get("blockQ25ROI", 0.0))
    deficit = (
        max(0.0, target - overall)
        + 0.45 * max(0.0, target - monthly_q25)
        + 0.35 * max(0.0, target - block_q25)
    )
    return round(-deficit, 4)


def _baseline_metrics(records: list[dict], rmap: dict[str, dict]) -> dict:
    rows = []
    for rec in records:
        race = rmap[rec["race_id"]]["race"]
        p = race.get("prediction") or {}
        if not (p.get("axes") or []):
            continue
        main = int(p["axes"][0])
        rows.append({
            "date": rec["date"],
            "race_id": rec["race_id"],
            "main": main,
            "win_return": _win_return(race, main),
            "top3": int(_top3_hit(race, main)),
            "tri_return": int(race.get("payout") or 0),
            "tri_stake": _tri_stake(race),
        })
    m = aggregate(rows)
    m["robustness"] = robustness(rows)
    m["blockRobustness"] = block_robustness(rows)
    m["floor80Score"] = floor80_score(m, m["blockRobustness"], 80.0)
    return m


def evaluate_policy(policy: D3Policy, records: list[dict], rmap: dict[str, dict]) -> tuple[dict, list[dict]]:
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
        rows.append({
            "date": rec["date"],
            "race_id": rid,
            "main": main,
            "second": second,
            "win_return": _win_return(race, main),
            "top3": int(_top3_hit(race, main)),
            "tri_return": trifecta_return(race, selected, main, second),
            "tri_stake": _tri_stake(race),
        })
    m = aggregate(rows)
    m["robustness"] = robustness(rows)
    m["blockRobustness"] = block_robustness(rows)
    m["floor80Score"] = floor80_score(m, m["blockRobustness"], 80.0)
    return m, rows


def oof_predictions(rows: list[dict], min_train_races: int) -> tuple[list[dict], dict]:
    by_race = group_rows(rows)
    dates = sorted({str(r["date"]) for r in rows})
    race_dates = {rid: race_rows[0]["date"] for rid, race_rows in by_race.items()}
    records: list[dict] = []
    fit_dates = 0
    fallback_dates = 0

    # Incremental date list avoids repeatedly searching target races.
    rows_by_date: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        rows_by_date[str(r["date"])].append(r)

    history_rows: list[dict] = []
    for date_s in dates:
        train_races = len({r["race_id"] for r in history_rows})
        model = None
        if train_races >= min_train_races:
            try:
                model = D3Model().fit(history_rows)
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
        # Critical leakage barrier: today's labels join training only after today's scoring.
        history_rows.extend(rows_by_date[date_s])

    return records, {
        "dates": len(dates),
        "modelFitDates": fit_dates,
        "fallbackDates": fallback_dates,
        "firstDate": dates[0] if dates else None,
        "lastDate": dates[-1] if dates else None,
    }


def policy_grid():
    # D3.1 keeps the search roughly the same size as D2/D3 while adding the two
    # value-gate controls. This avoids a huge hyperparameter explosion.
    for values in itertools.product(
        (2, 3, 4),                 # top_k
        (6.0, 9.0),                # max_total_gap
        (0.50, 0.65),              # min_win_ratio
        (0.45, 0.55),              # min_top3_ratio
        (1.00, 1.25),              # edge_power
        (0.25, 0.50),              # ev_power
        (0.0, 0.20),               # top3_power
        (0.0, 0.15),               # ability_power
        (0.0, 0.08),               # favorite_penalty
        (1.00, 1.08, 1.16),        # min_edge_to_switch
        (1.00, 1.08),              # switch_margin
    ):
        yield D3Policy(
            top_k=values[0],
            max_total_gap=values[1],
            min_win_ratio=values[2],
            min_top3_ratio=values[3],
            edge_power=values[4],
            ev_power=values[5],
            top3_power=values[6],
            ability_power=values[7],
            favorite_penalty=values[8],
            min_edge_to_switch=values[9],
            min_ev_to_switch=0.72,
            switch_margin=values[10],
        )


def tune_policy(records: list[dict], rmap: dict[str, dict], baseline: dict) -> tuple[D3Policy, dict, list[dict], int, int]:
    best = None
    best_key = None
    best_rows = None
    tested = 0
    valid = 0

    for policy in policy_grid():
        metrics, rows = evaluate_policy(policy, records, rmap)
        tested += 1

        # More demanding than D2: main reliability must be meaningfully above the old
        # baseline, trifecta must remain positive, and returns cannot hinge on 3 jackpots.
        min_top3 = max(40.0, baseline["top3Rate"] + 10.0)
        if metrics["top3Rate"] < min_top3:
            continue
        if metrics["trifectaRecoveryRate"] < max(100.0, baseline["trifectaRecoveryRate"] - 5.0):
            continue
        if metrics["top3WinReturnShare"] > 35.0:
            continue
        valid += 1

        rb = metrics["robustness"]
        br = metrics["blockRobustness"]
        key = (
            metrics["floor80Score"],
            rb["robustScore"],
            br["blockQ25ROI"],
            rb["monthlyQ25ROI"],
            metrics["winsorizedWinRecoveryRate50x"],
            metrics["winRecoveryRate"],
            metrics["top3Rate"],
            metrics["trifectaRecoveryRate"],
            -policy.max_total_gap,
        )
        if best_key is None or key > best_key:
            best_key = key
            best = policy
            best_rows = rows

    if best is None:
        best = D3Policy()
        metrics, best_rows = evaluate_policy(best, records, rmap)
        return best, metrics, best_rows, tested, valid
    metrics, _ = evaluate_policy(best, records, rmap)
    return best, metrics, best_rows or [], tested, valid


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
    best_policy, tune_metrics, tune_rows, tested, valid = tune_policy(tune_records, rmap, baseline_tune)
    holdout_metrics, holdout_rows = evaluate_policy(best_policy, holdout_records, rmap)
    all_metrics, all_rows = evaluate_policy(best_policy, modeled, rmap)
    baseline_all = _baseline_metrics(modeled, rmap)

    final_model = D3Model().fit(rows)
    trained_through = max(str(r["date"]) for r in rows)
    payload = {
        "version": MODEL_VERSION,
        "abilityFeatureCols": ABILITY_FEATURE_COLS,
        "trainedThrough": trained_through,
        "policy": best_policy.to_dict(),
        "model": final_model,
    }
    if not args.no_model_write:
        args.model_out.parent.mkdir(parents=True, exist_ok=True)
        joblib_dump(payload, args.model_out)

    goal_reached = holdout_metrics["winRecoveryRate"] >= float(args.goal_roi)
    promotion_ready = (
        holdout_metrics["winRecoveryRate"] >= float(args.goal_roi)
        and holdout_metrics["winRecoveryRate"] >= baseline_holdout["winRecoveryRate"] + 5.0
        and holdout_metrics["top3Rate"] >= 40.0
        and holdout_metrics["trifectaRecoveryRate"] >= 100.0
        and holdout_metrics["top3WinReturnShare"] <= 35.0
    )

    report = {
        "version": MODEL_VERSION,
        "generatedAt": datetime.now(JST).isoformat(timespec="seconds"),
        "goalWinRecoveryRate": float(args.goal_roi),
        "design": {
            "abilityMarketDecoupled": True,
            "payoutRegressionRemoved": True,
            "marketProxy": "historical expected-popularity rank + field bucket",
            "returnProxy": "geometric blend of 0.80*ability/market edge and robust payout prior EV",
            "recencyHalfLifeDays": 120,
            "recentModelWindowDays": 120,
            "recentModelBlend": 0.30,
            "policyObjective": "80% floor deficit + monthly/chronological lower-quartile robustness",
            "valueGate": "stable ability anchor; switch only on sufficient edge and score margin",
        },
        "leakagePolicy": {
            "currentRaceOdds": False,
            "actualPopularity": False,
            "bodyweightOrChange": False,
            "sameDayTrainingLabels": False,
            "trainingRule": "date < target_date only",
        },
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
            "validPolicyCandidates": valid,
            "baselineTune": baseline_tune,
            "d3Tune": tune_metrics,
            "baselineHoldout": baseline_holdout,
            "d3Holdout": holdout_metrics,
            "baselineAllModelScored": baseline_all,
            "d3AllModelScored": all_metrics,
            "d3TuneMonthly": monthly(tune_rows),
            "d3HoldoutMonthly": monthly(holdout_rows),
            "d3AllMonthly": monthly(all_rows),
        },
        "selectedPolicy": best_policy.to_dict(),
        "goalReachedOnUntouchedHoldout": goal_reached,
        "promotionReady": promotion_ready,
        "adoptionRule": (
            "Do not promote merely because tune ROI is high. Untouched holdout must improve "
            "the requested ROI floor (default 80%), improve baseline by >=5pp, keep top3>=40%, "
            "trifecta ROI>=100%, and top-3 win-return concentration<=35%. 100% remains an "
            "upside target, not a forced result."
        ),
    }
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "version": MODEL_VERSION,
        "policy": best_policy.to_dict(),
        "baselineHoldout": baseline_holdout,
        "d3Holdout": holdout_metrics,
        "goalReached": goal_reached,
        "promotionReady": promotion_ready,
        "modelOut": None if args.no_model_write else str(args.model_out),
        "metricsOut": str(args.metrics_out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
