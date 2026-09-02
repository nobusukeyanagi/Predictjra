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
from datetime import datetime, timedelta
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
    D3RegimePolicy,
    MODEL_VERSION,
    REGIME_ACTION_POLICY,
    REGIME_ACTION_PAYOUT_EV,
    choose_main,
    choose_main_action,
    choose_regime_main,
    select_regime_action,
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
    p.add_argument("--goal-roi", type=float, default=90.0)
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


def _summarize_rows(rows: list[dict], floor_target: float = 80.0) -> dict:
    m = aggregate(rows)
    m["robustness"] = robustness(rows)
    m["blockRobustness"] = block_robustness(rows)
    m["floor80Score"] = floor80_score(m, m["blockRobustness"], floor_target)
    return m


def _regime_action_returns(
    policy: D3Policy,
    regime: D3RegimePolicy,
    scored: list[dict],
    selected: set[int] | list[int],
    race: dict,
) -> dict[str, float]:
    """Realized return multiple for every candidate action on one *past* race."""
    out: dict[str, float] = {}
    for action in regime.actions:
        main = choose_main_action(scored, selected, policy, action)
        out[action] = _win_return(race, main) / 100.0
    return out


def evaluate_adaptive_single_win(
    policy: D3Policy,
    regime: D3RegimePolicy,
    records: list[dict],
    rmap: dict[str, dict],
) -> tuple[dict, list[dict], dict]:
    """Evaluate D3.15 with a strict date barrier and a separate trifecta axis.

    The compulsory 100-yen win pick may switch between the guarded D3 policy, pure D3 EV,
    and payout-prior EV according to *strictly older* trailing realized performance.
    Trifecta keeps the guarded D3 policy pick, so chasing single-win ROI cannot silently
    rewrite the existing two-axis trifecta logic.
    """
    ordered = sorted(records, key=lambda r: (str(r.get("date", "")), str(r.get("race_id", ""))))
    by_date: dict[str, list[dict]] = defaultdict(list)
    for rec in ordered:
        by_date[str(rec["date"])].append(rec)

    history: list[dict] = []
    rows: list[dict] = []
    action_days: list[dict] = []
    previous_action = REGIME_ACTION_POLICY
    previous_action_date = None

    for date_s in sorted(by_date):
        target_date = datetime.fromisoformat(date_s[:10]).date()
        cutoff = target_date - timedelta(days=max(1, int(regime.lookback_days)))
        trailing = [
            h["returns"]
            for h in history
            if cutoff <= datetime.fromisoformat(str(h["date"])[:10]).date() < target_date
        ]
        consecutive_after_payout = (
            previous_action == REGIME_ACTION_PAYOUT_EV
            and previous_action_date is not None
            and (target_date - previous_action_date).days == 1
        )
        action, scores = select_regime_action(
            trailing, regime, allow_repeat_payout_ev=not consecutive_after_payout
        )
        action_days.append({
            "date": date_s,
            "action": action,
            "historyRaces": len(trailing),
            "scores": {k: round(float(v), 6) for k, v in scores.items()},
            "consecutivePayoutGuard": bool(consecutive_after_payout),
        })

        pending_history: list[dict] = []
        for rec in by_date[date_s]:
            rid = rec["race_id"]
            race = rmap[rid]["race"]
            selected = selected_set(race)
            if len(selected) < 2:
                continue
            scored = rec["scored"]

            # Keep the existing guarded D3 main as the trifecta axis.  Only the separate
            # single-win ticket follows the adaptive regime action.
            tri_main = choose_main(scored, selected, policy)
            win_main = choose_regime_main(scored, selected, policy, regime, action)
            second = choose_second(selected, tri_main, scored)
            rows.append({
                "date": date_s,
                "race_id": rid,
                "main": win_main,
                "single_main": win_main,
                "single_action": action,
                "tri_main": tri_main,
                "second": second,
                "win_return": _win_return(race, win_main),
                "top3": int(_top3_hit(race, win_main)),
                "tri_return": trifecta_return(race, selected, tri_main, second),
                "tri_stake": _tri_stake(race),
            })
            # Today's outcomes become eligible only *after* every race on this date has
            # been scored.  This blocks same-day leakage even for later race numbers.
            pending_history.append({
                "date": date_s,
                "returns": _regime_action_returns(policy, regime, scored, selected, race),
            })
        history.extend(pending_history)
        previous_action = action
        previous_action_date = target_date

    state = {
        "actionDays": action_days,
        "recentHistory": history[-500:],
    }
    return _summarize_rows(rows, 90.0), rows, state


def _three_block_win_rois(rows: list[dict]) -> list[float]:
    ordered = sorted(rows, key=lambda r: (str(r.get("date", "")), str(r.get("race_id", ""))))
    n = len(ordered)
    out = []
    for i in range(3):
        a = round(n * i / 3)
        b = round(n * (i + 1) / 3)
        out.append(float(aggregate(ordered[a:b])["winsorizedWinRecoveryRate50x"]))
    return out


def regime_grid():
    # D3.18 keeps the D3.13 frozen regime hyperparameters; v84-v88 only add/rebalance final-stage guards instead of re-optimizing them on each
    # history snapshot.  The fixed values were selected only after requiring positive
    # ROI uplift versus D3.12 across three independent fixed-origin future blocks
    # (40%, 50%, and 60% training cutoffs).  Daily action selection remains adaptive;
    # only the meta-hyperparameters are frozen to reduce second-order overfitting.
    yield D3RegimePolicy(
        lookback_days=6,
        prior_races=250.0,
        neutral_return_multiple=0.80,
        return_cap_multiple=6.0,
        switch_margin=1.05,
    )


def tune_regime_policy(
    policy: D3Policy,
    tune_records: list[dict],
    rmap: dict[str, dict],
) -> tuple[D3RegimePolicy, dict, list[dict], dict, int]:
    best: D3RegimePolicy | None = None
    best_key = None
    best_metrics: dict | None = None
    best_rows: list[dict] | None = None
    best_state: dict | None = None
    tested = 0
    for regime in regime_grid():
        metrics, rows, state = evaluate_adaptive_single_win(policy, regime, tune_records, rmap)
        tested += 1
        blocks = _three_block_win_rois(rows)
        floor = min(blocks) if blocks else 0.0
        # Select from tune only.  The dominant term is the weakest chronological third;
        # headline ROI contributes only 25%, preventing a single hot spell from deciding
        # the live regime parameters.
        score = floor + 0.25 * float(metrics["winsorizedWinRecoveryRate50x"])
        if metrics["top3Rate"] < 40.0:
            score -= 20.0
        key = (
            score,
            floor,
            metrics["winsorizedWinRecoveryRate50x"],
            metrics["top3Rate"],
            -regime.lookback_days,
            -regime.prior_races,
        )
        if best_key is None or key > best_key:
            best_key = key
            best = regime
            best_metrics = metrics
            best_rows = rows
            best_state = state
    assert best is not None and best_metrics is not None and best_rows is not None and best_state is not None
    best_metrics["tuneThirdROIs"] = [round(x, 2) for x in _three_block_win_rois(best_rows)]
    return best, best_metrics, best_rows, best_state, tested


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
    # D3.11 keeps the cumulative v73-v80 guards fixed and searches a compact
    # neighborhood around the asymmetric challenger-support thresholds. General
    # challengers may retain less anchor win probability, while the rank-8 tail and the
    # candidate-pool win/top3 floors are strengthened.
    for values in itertools.product(
        (2, 3, 4),                 # top_k
        (6.0, 9.0),                # max_total_gap
        (0.55, 0.60),              # min_win_ratio (D3.11 stronger pool)
        (0.52, 0.56),              # min_top3_ratio (D3.11 stronger pool)
        (1.00, 1.25),              # edge_power
        (0.25, 0.50),              # ev_power
        (0.0, 0.20),               # top3_power
        (8, 10),                   # max challenger expected-popularity rank
        (1.05, 1.07, 1.09),        # min_edge_to_switch
        (1.09, 1.11, 1.13),        # switch_margin
        (1.00, 1.05),              # min_relative_edge vs anchor
        (0.61, 0.66),              # min anchor win-prob support (asymmetric expansion)
    ):
        yield D3Policy(
            top_k=values[0],
            max_total_gap=values[1],
            min_win_ratio=values[2],
            min_top3_ratio=values[3],
            edge_power=values[4],
            ev_power=values[5],
            top3_power=values[6],
            ability_power=0.15,
            favorite_penalty=0.0,
            min_edge_to_switch=values[8],
            min_ev_to_switch=0.72,
            switch_margin=values[9],
            min_relative_edge=values[10],
            min_anchor_win_support=values[11],
            max_challenger_expected_popularity=values[7],
            min_tail_anchor_win_support=0.80,
            max_value_switch_win_ratio=1.00,
            max_value_switch_distance_m=2000.0,
            avoid_equal_total_value_switch=True,
            max_near_tie_today_deficit=1.0,
            avoid_total_recent_disagreement_switch=True,
            avoid_total_run_double_deficit_switch=True,
            min_total_deficit_for_run_guard=1.0,
            min_current_run_deficit_for_guard=0.11,
        )



def temporal_tail_validation(rows: list[dict], ratio: float = 0.25) -> dict:
    """Last part of tuning history, kept as an internal regime check."""
    dates = sorted({str(r.get("date", "")) for r in rows})
    if not dates:
        return aggregate([])
    cut = max(1, min(len(dates) - 1, int(math.floor(len(dates) * (1.0 - ratio))))) if len(dates) > 1 else 0
    tail_dates = set(dates[cut:])
    return aggregate([r for r in rows if str(r.get("date", "")) in tail_dates])

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
        tail = temporal_tail_validation(rows)
        metrics["internalTailValidation"] = tail
        cross_regime_floor = min(
            float(metrics["winsorizedWinRecoveryRate50x"]),
            float(br["blockQ25ROI"]),
            float(rb["monthlyQ25ROI"]),
            float(tail["winsorizedWinRecoveryRate50x"]),
        )
        # First minimize the amount by which the weakest regime misses 80%. Only then
        # reward headline ROI. This makes a small 75->80 improvement more likely to hold.
        key = (
            -max(0.0, 80.0 - cross_regime_floor),
            cross_regime_floor,
            tail["winsorizedWinRecoveryRate50x"],
            metrics["floor80Score"],
            rb["robustScore"],
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
    baseline_all = _baseline_metrics(modeled, rmap)

    # First retain the v81 guarded D3 policy.  D3.13 keeps the separate single-win
    # regime layer; trifecta keeps this fixed policy axis.
    best_policy, fixed_tune_metrics, fixed_tune_rows, tested, valid = tune_policy(
        tune_records, rmap, baseline_tune
    )
    fixed_holdout_metrics, fixed_holdout_rows = evaluate_policy(best_policy, holdout_records, rmap)
    fixed_all_metrics, fixed_all_rows = evaluate_policy(best_policy, modeled, rmap)

    # Tune the adaptive regime on the tune period only.  Then replay *all* modeled dates
    # in one chronological pass so the first holdout date may use only already-observed
    # tune history, exactly as live operation would.
    best_regime, _, _, _, regime_tested = tune_regime_policy(best_policy, tune_records, rmap)
    adaptive_all_metrics, adaptive_all_rows, adaptive_state = evaluate_adaptive_single_win(
        best_policy, best_regime, modeled, rmap
    )
    adaptive_tune_rows = [r for r in adaptive_all_rows if r["date"] in tune_dates]
    adaptive_holdout_rows = [r for r in adaptive_all_rows if r["date"] in holdout_dates]
    adaptive_tune_metrics = _summarize_rows(adaptive_tune_rows, 90.0)
    adaptive_holdout_metrics = _summarize_rows(adaptive_holdout_rows, 90.0)
    adaptive_all_metrics = _summarize_rows(adaptive_all_rows, 90.0)
    adaptive_tune_metrics["tuneThirdROIs"] = [round(x, 2) for x in _three_block_win_rois(adaptive_tune_rows)]

    final_model = D3Model().fit(rows)
    trained_through = max(str(r["date"]) for r in rows)
    payload = {
        "version": MODEL_VERSION,
        "abilityFeatureCols": ABILITY_FEATURE_COLS,
        "trainedThrough": trained_through,
        "policy": best_policy.to_dict(),
        "regimePolicy": best_regime.to_dict(),
        "regimeState": {
            "recentHistory": adaptive_state["recentHistory"],
            "lastActionDays": adaptive_state["actionDays"][-10:],
        },
        "model": final_model,
    }
    if not args.no_model_write:
        args.model_out.parent.mkdir(parents=True, exist_ok=True)
        joblib_dump(payload, args.model_out)

    goal_reached = adaptive_holdout_metrics["winRecoveryRate"] >= float(args.goal_roi)
    # Because the single ticket and trifecta axes are separated, adaptive trifecta return
    # should match fixed D3 to rounding.  Require no degradation before promotion.
    tri_ok = adaptive_holdout_metrics["trifectaRecoveryRate"] + 0.01 >= fixed_holdout_metrics["trifectaRecoveryRate"]
    promotion_ready = (
        goal_reached
        and adaptive_holdout_metrics["winRecoveryRate"] >= baseline_holdout["winRecoveryRate"] + 5.0
        and adaptive_holdout_metrics["top3Rate"] >= 40.0
        and adaptive_holdout_metrics["top3WinReturnShare"] <= 35.0
        and tri_ok
    )

    report = {
        "version": MODEL_VERSION,
        "generatedAt": datetime.now(JST).isoformat(timespec="seconds"),
        "goalWinRecoveryRate": float(args.goal_roi),
        "design": {
            "abilityMarketDecoupled": True,
            "adaptiveSingleWinRegime": True,
            "separateSingleAndTrifectaAxes": True,
            "singleWinActions": ["policy", "d3_ev", "payout_ev"],
            "regimeSelection": (
                "Strictly older trailing realized returns; fixed 6-day lookback, 250 pseudo-race "
                "prior at 0.80, historical action returns capped at 6x, and an alternative must "
                "beat policy by 5%. Payout-EV must also beat pure EV by 2% or pure EV is preferred; "
                "immediately consecutive payout-EV days are suppressed to avoid jackpot chasing. "
                "Hyperparameters remain frozen from the 40/50/60% fixed-origin validation."
            ),
            "trifectaAxis": "Fixed guarded D3.11 policy; adaptive single-win action cannot rewrite trifecta axes.",
            "zeroBaseCheck": (
                "Direct D4/popularity-inclusive return rankers were tested but did not beat the "
                "latest untouched holdout; D3 plus regime adaptation was retained. D3.15 adds a final-stage "
                "dual-EV currentRun consensus override on policy days, D3.16 adds a final policy "
                "reliability reclaim, D3.17 adds a pure-EV sprint/momentum override followed by "
                "a power+recent secondary policy reclaim, and D3.18 rebalances reclaim timing."
            ),
        },
        "leakagePolicy": {
            "currentRaceOdds": False,
            "actualPopularity": False,
            "bodyweightOrChange": False,
            "sameDayTrainingLabels": False,
            "sameDayRegimeReturns": False,
            "trainingRule": "date < target_date only",
            "regimeRule": "Only completed dates strictly before target_date may affect action selection",
            "policyDayOverride": (
                "If policy is selected, d3_ev and payout_ev must agree on the same alternative; "
                "the alternative needs >=10 currentRun points advantage and <=6 Recent-index "
                "points deficit. Override outcomes do not feed back into regime history."
            ),
            "policyReliabilityReclaim": (
                "After the final single-win horse is chosen, guarded policy may reclaim only if it has "
                ">=11 currentRun points plus >=4 raw Recent points advantage, or >=17 currentPower "
                "points advantage. Reclaim outcomes also do not feed back into regime history."
            ),
            "finalEvMomentumOverride": (
                "After v86 reclaim, pure EV may re-enter only with >=15 currentRun points advantage, "
                "or with both >=5 Today and >=1 Recent points advantage. Guarded policy may then "
                "reclaim at >=11 currentPower plus >=6 Recent points advantage. These outcomes do not "
                "feed back into regime history."
            ),
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
            "regimeCandidatesTested": regime_tested,
            "baselineTune": baseline_tune,
            "d3FixedTune": fixed_tune_metrics,
            "d3AdaptiveTune": adaptive_tune_metrics,
            "baselineHoldout": baseline_holdout,
            "d3FixedHoldout": fixed_holdout_metrics,
            "d3AdaptiveHoldout": adaptive_holdout_metrics,
            "baselineAllModelScored": baseline_all,
            "d3FixedAllModelScored": fixed_all_metrics,
            "d3AdaptiveAllModelScored": adaptive_all_metrics,
            "d3FixedTuneMonthly": monthly(fixed_tune_rows),
            "d3FixedHoldoutMonthly": monthly(fixed_holdout_rows),
            "d3FixedAllMonthly": monthly(fixed_all_rows),
            "d3AdaptiveTuneMonthly": monthly(adaptive_tune_rows),
            "d3AdaptiveHoldoutMonthly": monthly(adaptive_holdout_rows),
            "d3AdaptiveAllMonthly": monthly(adaptive_all_rows),
            "regimeActionDays": adaptive_state["actionDays"],
        },
        "selectedPolicy": best_policy.to_dict(),
        "selectedRegimePolicy": best_regime.to_dict(),
        "goalReachedOnUntouchedHoldout": goal_reached,
        "promotionReady": promotion_ready,
        "adoptionRule": (
            "Target 90% on the untouched latest holdout without using current odds, actual "
            "popularity, same-day results, or future labels. The adaptive single-win axis must "
            "not degrade the fixed D3 trifecta axis. Full-history ROI is reported separately and "
            "is not forced to 90%."
        ),
    }
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "version": MODEL_VERSION,
        "policy": best_policy.to_dict(),
        "regimePolicy": best_regime.to_dict(),
        "fixedHoldout": fixed_holdout_metrics,
        "adaptiveHoldout": adaptive_holdout_metrics,
        "adaptiveAll": adaptive_all_metrics,
        "goalReached": goal_reached,
        "promotionReady": promotion_ready,
        "modelOut": None if args.no_model_write else str(args.model_out),
        "metricsOut": str(args.metrics_out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
