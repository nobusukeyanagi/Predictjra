#!/usr/bin/env python3
"""Rebuild Predictjra predictions/results for 2026-08-15 and 2026-08-16.

Goals
-----
* Reconstruct predictions from archived PRE-RACE snapshots.
* Never use current-race odds, actual popularity, horse bodyweight/change, or race result
  as inputs to the performance/selection score.
* Use actual popularity only as a teacher label for the estimated-popularity model.
  Each race is estimated with a model trained while excluding that race itself.
* Re-fetch archived final results and trifecta payouts and recalculate hit/miss, return,
  stake, and daily totals.
* Refuse to write partial data: 72 races and all required validation checks must pass.

Historical snapshot source:
  https://github.com/sugaimo15/keibayosoku
  ref: claude/horse-racing-predictor-ak6crm

The archived prediction CSVs were created before odds/bodyweight were populated. This
script verifies that condition again before using their non-market pre-race score fields.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

JST = ZoneInfo("Asia/Tokyo")
TARGET_DATES = ("2026-08-15", "2026-08-16")
EXPECTED_RACES_PER_DAY = 36
EXPECTED_TOTAL_RACES = 72

TRACKS = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
}
TRACK_ORDER = {name: i for i, name in enumerate(TRACKS.values(), start=1)}

# This race already has the fully hand-reviewed Predictjra detailed index.
# Keep its "総合" ordering from the modal; only 想人 is replaced by the
# cross-race calibrated popularity estimator.
SAPPORO11_ID = "202601010811"
SAPPORO11_TOTAL = {
    1: 65, 2: 65, 3: 64, 4: 80, 5: 75, 6: 74, 7: 79, 8: 87,
    9: 75, 10: 83, 11: 75, 12: 75, 13: 81, 14: 79, 15: 78, 16: 74,
}
# Secondary ordering used by the current detailed index when displayed totals tie.
SAPPORO11_RANK = {
    8: 1, 10: 2, 13: 3, 4: 4, 7: 5, 14: 6, 15: 7, 12: 8,
    9: 9, 11: 10, 5: 11, 16: 12, 6: 13, 1: 14, 2: 15, 3: 16,
}

REBUILD_VERSION = "predictjra-history-20260815-16-v1"
SOURCE_REPO = "sugaimo15/keibayosoku"
SOURCE_REF = "claude/horse-racing-predictor-ak6crm"


def clean_str(v) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip()


def numeric_or_nan(v) -> float:
    return pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]


def minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    if s.notna().sum() == 0:
        return pd.Series(0.5, index=series.index, dtype=float)
    s = s.fillna(s.mean())
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-12:
        return pd.Series(0.5, index=series.index, dtype=float)
    return (s - lo) / (hi - lo)


def parse_age(sex_age) -> float:
    m = re.search(r"(\d+)", clean_str(sex_age))
    return float(m.group(1)) if m else 5.0


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def source_commit(source_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return ""


@dataclass
class RaceModel:
    race_id: str
    date: str
    card: pd.DataFrame
    pred: pd.DataFrame
    result: pd.DataFrame
    payout: pd.DataFrame
    total_internal: dict[int, float]
    total_display: dict[int, int]
    tie_order: dict[int, float]
    popularity_features: pd.DataFrame
    actual_popularity: dict[int, int]


def validate_no_market_input(card: pd.DataFrame, pred: pd.DataFrame, race_id: str) -> None:
    """Fail if the archived prediction snapshot contains current odds/bodyweight.

    We deliberately do not merely drop these values because the precomputed score could
    already have incorporated them. A contaminated snapshot must not be used.
    """
    for label, df in (("card", card), ("prediction", pred)):
        if "win_odds" in df.columns:
            odds = pd.to_numeric(df["win_odds"], errors="coerce")
            if (odds > 0).any():
                bad = df.loc[odds > 0, ["horse_number", "win_odds"]].to_dict("records")
                raise ValueError(f"{race_id} {label}: numeric current odds found: {bad[:4]}")
        if "horse_weight" in df.columns:
            nonblank = df["horse_weight"].apply(clean_str)
            if nonblank.ne("").any():
                bad = df.loc[nonblank.ne(""), ["horse_number", "horse_weight"]].to_dict("records")
                raise ValueError(f"{race_id} {label}: current horse weight found: {bad[:4]}")


def build_race_model(source_root: Path, date_s: str, pred_path: Path) -> RaceModel:
    race_id = pred_path.stem
    card_path = source_root / "data" / "race_cards" / date_s.replace("-", "") / f"{race_id}.csv"
    result_path = source_root / "data" / "race_results" / "2026" / f"{race_id}.csv"
    payout_path = source_root / "data" / "race_payouts" / f"{race_id}.csv"

    card = read_csv(card_path)
    pred = read_csv(pred_path)
    result = read_csv(result_path)
    payout = read_csv(payout_path)

    validate_no_market_input(card, pred, race_id)

    for df_name, df in (("card", card), ("pred", pred), ("result", result)):
        if df.empty:
            raise ValueError(f"{race_id}: empty {df_name}")
        if "race_id" not in df.columns or set(df["race_id"].astype(str)) != {race_id}:
            raise ValueError(f"{race_id}: invalid race_id in {df_name}")

    card_nums = set(pd.to_numeric(card["horse_number"], errors="raise").astype(int))
    pred_nums = set(pd.to_numeric(pred["horse_number"], errors="raise").astype(int))
    result_nums = set(pd.to_numeric(result["horse_number"], errors="raise").astype(int))
    if card_nums != pred_nums:
        raise ValueError(f"{race_id}: card/pred horse mismatch {sorted(card_nums ^ pred_nums)}")
    if not result_nums.issubset(card_nums):
        raise ValueError(f"{race_id}: result contains horse not in card {sorted(result_nums-card_nums)}")
    if len(card_nums) < 5:
        raise ValueError(f"{race_id}: only {len(card_nums)} horses")

    required_pred = {"horse_number", "score", "ml_win_prob", "predicted_rank", "ml_rank"}
    missing = required_pred - set(pred.columns)
    if missing:
        raise ValueError(f"{race_id}: prediction missing columns {sorted(missing)}")

    p = pred.copy()
    p["horse_number"] = pd.to_numeric(p["horse_number"], errors="raise").astype(int)
    p["rule_norm"] = minmax(p["score"])
    p["ml_norm"] = minmax(p["ml_win_prob"])
    # General total-performance proxy for the historical all-race reconstruction.
    # Both terms are pre-race and market-neutral in the validated snapshot.
    p["total_raw"] = 0.65 * p["rule_norm"] + 0.35 * p["ml_norm"]

    # Display-scale integer is audit metadata; selection uses unrounded total_raw.
    p["total_display"] = (55 + 40 * p["total_raw"]).round().astype(int)

    total_internal = dict(zip(p["horse_number"], p["total_raw"].astype(float)))
    total_display = dict(zip(p["horse_number"], p["total_display"].astype(int)))
    tie_order = dict(zip(
        p["horse_number"],
        pd.to_numeric(p["predicted_rank"], errors="coerce").fillna(999).astype(float)
    ))

    if race_id == SAPPORO11_ID:
        total_internal = {n: float(v) for n, v in SAPPORO11_TOTAL.items()}
        total_display = dict(SAPPORO11_TOTAL)
        tie_order = {n: float(v) for n, v in SAPPORO11_RANK.items()}

    # Popularity-model features. No actual-popularity value is placed in X.
    n = len(p)
    denom = max(n - 1, 1)
    p["rule_rank_strength"] = 1 - (
        pd.to_numeric(p["predicted_rank"], errors="coerce").fillna(n) - 1
    ) / denom
    p["ml_rank_strength"] = 1 - (
        pd.to_numeric(p["ml_rank"], errors="coerce").fillna(n) - 1
    ) / denom
    p["age_strength"] = p.get("sex_age", pd.Series([""] * n)).apply(
        lambda x: float(np.clip((10.0 - parse_age(x)) / 8.0, 0.0, 1.0))
    )
    carried = pd.to_numeric(p.get("weight_carried"), errors="coerce")
    if carried.notna().sum():
        centered = carried.fillna(carried.mean())
        span = max(float(centered.max() - centered.min()), 1.0)
        p["carried_strength"] = 0.5 - (centered - centered.mean()) / (2 * span)
    else:
        p["carried_strength"] = 0.5

    features = p[[
        "horse_number", "rule_norm", "ml_norm", "rule_rank_strength",
        "ml_rank_strength", "age_strength", "carried_strength"
    ]].copy()
    features["race_id"] = race_id

    actual_popularity = {}
    if "popularity" not in result.columns:
        raise ValueError(f"{race_id}: result popularity missing")
    for _, row in result.iterrows():
        no = int(row["horse_number"])
        pop = pd.to_numeric(pd.Series([row["popularity"]]), errors="coerce").iloc[0]
        if pd.notna(pop):
            actual_popularity[no] = int(pop)
    if set(actual_popularity) != result_nums:
        raise ValueError(f"{race_id}: incomplete actual popularity")

    return RaceModel(
        race_id=race_id,
        date=date_s,
        card=card,
        pred=p,
        result=result,
        payout=payout,
        total_internal=total_internal,
        total_display=total_display,
        tie_order=tie_order,
        popularity_features=features,
        actual_popularity=actual_popularity,
    )


FEATURE_COLS = [
    "rule_norm", "ml_norm", "rule_rank_strength",
    "ml_rank_strength", "age_strength", "carried_strength"
]


def target_market_strength(pop: int, field_size: int) -> float:
    return 1.0 - (float(pop) - 1.0) / max(field_size - 1, 1)


def fit_ridge(rows: pd.DataFrame, ridge: float = 2.0) -> np.ndarray:
    X = rows[FEATURE_COLS].astype(float).to_numpy()
    X = np.column_stack([np.ones(len(X)), X])
    y = rows["market_strength"].astype(float).to_numpy()
    reg = np.eye(X.shape[1]) * ridge
    reg[0, 0] = 0.0  # do not regularize intercept
    return np.linalg.solve(X.T @ X + reg, X.T @ y)


def predict_strength(features: pd.DataFrame, beta: np.ndarray) -> np.ndarray:
    X = features[FEATURE_COLS].astype(float).to_numpy()
    X = np.column_stack([np.ones(len(X)), X])
    return X @ beta


def estimate_popularities(races: dict[str, RaceModel]) -> tuple[dict[str, dict[int, int]], dict]:
    # Build teacher table from final popularity. It is never an input feature.
    teacher_parts = []
    for rid, race in races.items():
        f = race.popularity_features.copy()
        f["field_size"] = len(race.card)
        f["actual_popularity"] = f["horse_number"].map(race.actual_popularity)
        # Scratched/non-starter horses may exist in the archived card but have no final
        # popularity. They remain prediction candidates for the historical snapshot,
        # but are not teacher rows for popularity calibration.
        f = f[f["actual_popularity"].notna()].copy()
        f["market_strength"] = [
            target_market_strength(int(p), len(race.card))
            for p in f["actual_popularity"]
        ]
        teacher_parts.append(f)
    teacher = pd.concat(teacher_parts, ignore_index=True)

    estimates: dict[str, dict[int, int]] = {}
    abs_errors = []
    top3_overlaps = []

    # Leave-one-race-out: a race's own final popularity never trains its estimator.
    for rid, race in races.items():
        train = teacher[teacher["race_id"] != rid]
        beta = fit_ridge(train)
        feats = race.popularity_features.copy()
        feats["_pred_market"] = predict_strength(feats, beta)
        # Stable deterministic tie break: performance proxy, then horse number.
        feats["_total"] = feats["horse_number"].map(race.total_internal)
        feats = feats.sort_values(
            ["_pred_market", "_total", "horse_number"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        feats["estimated_popularity"] = np.arange(1, len(feats) + 1)
        est = dict(zip(
            feats["horse_number"].astype(int),
            feats["estimated_popularity"].astype(int),
        ))
        estimates[rid] = est

        actual = race.actual_popularity
        for no, ep in est.items():
            if no in actual:
                abs_errors.append(abs(ep - actual[no]))
        est_top3 = {no for no, p in est.items() if p <= 3}
        act_top3 = {no for no, p in actual.items() if p <= 3}
        top3_overlaps.append(len(est_top3 & act_top3) / 3.0)

    # Final coefficients for future generalization (trained on all 72 races).
    final_beta = fit_ridge(teacher)
    metrics = {
        "method": "leave-one-race-out for historical estimates",
        "ridge": 2.0,
        "features": FEATURE_COLS,
        "meanAbsolutePopularityRankError": round(float(np.mean(abs_errors)), 4),
        "meanTop3OverlapRate": round(float(np.mean(top3_overlaps)), 4),
        "futureModelCoefficients": {
            "intercept": float(final_beta[0]),
            **{name: float(v) for name, v in zip(FEATURE_COLS, final_beta[1:])},
        },
        "teacherRows": int(len(teacher)),
        "teacherRaces": int(teacher["race_id"].nunique()),
    }
    return estimates, metrics


def prediction_target_count(field_size: int) -> int:
    return min((field_size + 1) // 2, 7)


def build_prediction(race: RaceModel, estimated_popularity: dict[int, int]) -> tuple[dict, int]:
    horses = sorted(race.total_internal)

    # 危険: 想定1〜3番人気のうち総合評価最下位.
    top3 = [h for h in horses if estimated_popularity[h] <= 3]
    if len(top3) != 3:
        raise ValueError(f"{race.race_id}: estimated top3 count={len(top3)}")
    danger = min(
        top3,
        key=lambda h: (
            race.total_internal[h],
            -race.tie_order.get(h, 999),
            -h,
        ),
    )

    target_n = prediction_target_count(len(horses))
    eligible = [h for h in horses if h != danger]
    selected = sorted(
        eligible,
        key=lambda h: (
            -race.total_internal[h],
            race.tie_order.get(h, 999),
            h,
        ),
    )[:target_n]
    if len(selected) != target_n:
        raise ValueError(f"{race.race_id}: selected {len(selected)} != {target_n}")

    main = selected[0]
    # 対抗: selection target with the lowest estimated popularity (= largest rank).
    second = max(
        (h for h in selected if h != main),
        key=lambda h: (estimated_popularity[h], race.total_internal[h], -h),
    )
    opponents = [
        h for h in selected if h not in {main, second}
    ]
    opponents.sort(
        key=lambda h: (-race.total_internal[h], race.tie_order.get(h, 999), h)
    )

    return {"axes": [main, second], "opponents": opponents}, danger


def build_result(race: RaceModel) -> tuple[dict, list[int]]:
    r = race.result.copy()
    r["finish_num"] = pd.to_numeric(r["finish_position"], errors="coerce")
    r["horse_num"] = pd.to_numeric(r["horse_number"], errors="coerce")
    top = r[r["finish_num"].notna() & (r["finish_num"] <= 3)].copy()
    if top.empty:
        raise ValueError(f"{race.race_id}: no top-3 result")

    places = []
    for pos in sorted(top["finish_num"].astype(int).unique()):
        group = top[top["finish_num"].astype(int) == pos]["horse_num"].astype(int).tolist()
        if group:
            places.append(group)

    p = race.payout.copy()
    tri = p[p["bet_type"].astype(str).isin(["三連単", "3連単"])].copy()
    if tri.empty:
        raise ValueError(f"{race.race_id}: no trifecta payout row")

    trifectas = []
    for _, row in tri.iterrows():
        nums = [int(x) for x in re.findall(r"\d+", clean_str(row["combination"]))]
        if len(nums) != 3:
            raise ValueError(f"{race.race_id}: bad trifecta combination {row['combination']}")
        amount = int(pd.to_numeric(pd.Series([row["amount"]]), errors="raise").iloc[0])
        trifectas.append({"horses": nums, "payout": amount})

    return {"places": places, "trifectas": trifectas}, [t["payout"] for t in trifectas]


def covered(prediction: dict, horses: list[int]) -> bool:
    combo = set(horses)
    axes = set(prediction.get("axes", []))
    opponents = set(prediction.get("opponents", []))
    return axes.issubset(combo) and bool(opponents & combo)


def normalize_old_result(race: dict) -> tuple:
    result = race.get("result") or {}
    places = tuple(tuple(int(x) for x in g) for g in result.get("places", []))
    tris = tuple(sorted(
        (tuple(int(x) for x in t.get("horses", [])), int(t.get("payout", 0)))
        for t in result.get("trifectas", [])
    ))
    return places, tris


def normalize_new_result(result: dict) -> tuple:
    places = tuple(tuple(int(x) for x in g) for g in result.get("places", []))
    tris = tuple(sorted(
        (tuple(int(x) for x in t["horses"]), int(t["payout"]))
        for t in result.get("trifectas", [])
    ))
    return places, tris


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--data-path", default="data/races.json", type=Path)
    parser.add_argument("--audit-path", default="data/rebuild_audit_20260815_16.json", type=Path)
    parser.add_argument("--pop-model-path", default="data/popularity_model_20260815_16.json", type=Path)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    if not source_root.exists():
        raise FileNotFoundError(source_root)
    if not args.data_path.exists():
        raise FileNotFoundError(args.data_path)

    source_sha = source_commit(source_root)
    races: dict[str, RaceModel] = {}

    for date_s in TARGET_DATES:
        pred_dir = source_root / "data" / "predictions" / date_s.replace("-", "")
        files = sorted(pred_dir.glob("*.csv"))
        if len(files) != EXPECTED_RACES_PER_DAY:
            raise RuntimeError(
                f"{date_s}: expected {EXPECTED_RACES_PER_DAY} prediction files, got {len(files)}"
            )
        for f in files:
            race = build_race_model(source_root, date_s, f)
            if race.race_id in races:
                raise RuntimeError(f"duplicate race {race.race_id}")
            races[race.race_id] = race

    if len(races) != EXPECTED_TOTAL_RACES:
        raise RuntimeError(f"expected {EXPECTED_TOTAL_RACES} races, got {len(races)}")

    estimated_pops, pop_metrics = estimate_popularities(races)

    data = json.loads(args.data_path.read_text(encoding="utf-8"))
    existing_by_date = {
        d.get("date"): d for d in data.get("days", [])
    }

    audit_races = []
    daily = {}
    old_result_mismatches = []

    for date_s in TARGET_DATES:
        day = existing_by_date.get(date_s)
        if day is None:
            day = {"date": date_s, "races": []}
            data.setdefault("days", []).append(day)
            existing_by_date[date_s] = day

        old_by_id = {r.get("raceId"): r for r in day.get("races", [])}
        rebuilt = []

        day_races = [r for r in races.values() if r.date == date_s]
        day_races.sort(key=lambda x: (int(x.race_id[4:6]), int(x.race_id[-2:])))

        day_hits = 0
        day_payout = 0
        day_stake = 0

        for race in day_races:
            rid = race.race_id
            est_pop = estimated_pops[rid]
            prediction, danger = build_prediction(race, est_pop)
            result, tri_payouts = build_result(race)

            old = old_by_id.get(rid, {})
            if old.get("result") and normalize_old_result(old) != normalize_new_result(result):
                old_result_mismatches.append(rid)

            winners = [
                t for t in result["trifectas"]
                if covered(prediction, t["horses"])
            ]
            payout_return = sum(int(t["payout"]) for t in winners)
            stake = len(prediction["opponents"]) * 6 * 100
            status = "hit" if winners else "miss"

            if status == "hit":
                day_hits += 1
            day_payout += payout_return
            day_stake += stake

            card = race.card.copy()
            card["horse_number"] = pd.to_numeric(card["horse_number"], errors="raise").astype(int)
            frames = {
                str(int(row["horse_number"])): int(row["waku"])
                for _, row in card.iterrows()
                if pd.notna(row.get("waku"))
            }
            names = {
                str(int(row["horse_number"])): clean_str(row["horse_name"])
                for _, row in card.iterrows()
            }

            updated = dict(old)
            updated.pop("seedNote", None)
            updated.update({
                "raceId": rid,
                "venue": TRACKS.get(rid[4:6], rid[4:6]),
                "raceNo": int(rid[-2:]),
                "horseCount": int(len(card)),
                "horseFrames": frames,
                "horseNames": names,
                "prediction": prediction,
                "danger": [int(danger)],
                "result": result,
                "status": status,
                "payout": int(payout_return),
                "trifectaPayouts": [int(x) for x in tri_payouts],
                "stake": int(stake),
                "modelMeta": {
                    "version": REBUILD_VERSION,
                    "estimatedPopularity": {str(k): int(v) for k, v in est_pop.items()},
                    "totalIndex": {str(k): int(v) for k, v in race.total_display.items()},
                    "performanceSource": (
                        "detailed-index-pilot"
                        if rid == SAPPORO11_ID
                        else "pre-race historical reconstruction"
                    ),
                    "popularityMethod": "leave-one-race-out calibrated model",
                },
                "dataSources": {
                    "preRaceSnapshot": f"{SOURCE_REPO}@{source_sha or SOURCE_REF}",
                    "resultArchive": f"{SOURCE_REPO}@{source_sha or SOURCE_REF}",
                    "payoutArchive": f"{SOURCE_REPO}@{source_sha or SOURCE_REF}",
                },
            })
            rebuilt.append(updated)

            actual_pop = race.actual_popularity
            mae = float(np.mean([
                abs(est_pop[h] - actual_pop[h]) for h in est_pop if h in actual_pop
            ]))
            audit_races.append({
                "date": date_s,
                "raceId": rid,
                "venue": updated["venue"],
                "raceNo": updated["raceNo"],
                "horseCount": updated["horseCount"],
                "prediction": prediction,
                "danger": danger,
                "result": result["places"],
                "trifectas": result["trifectas"],
                "status": status,
                "return": payout_return,
                "stake": stake,
                "recoveryRate": round(payout_return / stake * 100, 1) if stake else 0.0,
                "popularityMAE": round(mae, 3),
                "oldResultMatchedArchive": rid not in old_result_mismatches,
            })

        if len(rebuilt) != EXPECTED_RACES_PER_DAY:
            raise RuntimeError(f"{date_s}: rebuilt {len(rebuilt)} races")

        day["races"] = rebuilt
        daily[date_s] = {
            "races": len(rebuilt),
            "hits": day_hits,
            "return": day_payout,
            "stake": day_stake,
            "recoveryRate": round(day_payout / day_stake * 100, 1) if day_stake else 0.0,
        }

    data["updatedAt"] = datetime.now(JST).isoformat(timespec="seconds")
    data["days"] = sorted(data.get("days", []), key=lambda d: d.get("date", ""), reverse=True)
    for day in data["days"]:
        day["races"] = sorted(
            day.get("races", []),
            key=lambda r: (
                TRACK_ORDER.get(r.get("venue", ""), 99),
                int(r.get("raceNo", 99)),
            ),
        )

    # Final hard validations before writing.
    for date_s in TARGET_DATES:
        day = next(d for d in data["days"] if d.get("date") == date_s)
        if len(day["races"]) != EXPECTED_RACES_PER_DAY:
            raise RuntimeError(f"{date_s}: final race count invalid")
        ids = [r["raceId"] for r in day["races"]]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"{date_s}: duplicate race IDs")
        for r in day["races"]:
            p = r["prediction"]
            target_n = prediction_target_count(r["horseCount"])
            if 2 + len(p["opponents"]) != target_n:
                raise RuntimeError(f"{r['raceId']}: selected count invalid")
            if set(p["axes"]) & set(p["opponents"]):
                raise RuntimeError(f"{r['raceId']}: axes/opponents overlap")
            if r.get("danger", [None])[0] in set(p["axes"] + p["opponents"]):
                raise RuntimeError(f"{r['raceId']}: danger horse selected")
            expected_stake = len(p["opponents"]) * 600
            if r["stake"] != expected_stake:
                raise RuntimeError(f"{r['raceId']}: stake invalid")

    audit = {
        "version": REBUILD_VERSION,
        "generatedAt": datetime.now(JST).isoformat(timespec="seconds"),
        "sourceRepository": SOURCE_REPO,
        "sourceRef": SOURCE_REF,
        "sourceCommit": source_sha,
        "strictInputRules": {
            "currentOddsUsed": False,
            "currentPopularityUsedAsPredictionFeature": False,
            "horseWeightOrChangeUsed": False,
            "raceResultUsedAsPerformanceFeature": False,
            "actualPopularityUse": (
                "teacher label only; each race excluded from its own estimator training"
            ),
        },
        "raceCount": len(audit_races),
        "oldResultMismatchCount": len(old_result_mismatches),
        "oldResultMismatches": old_result_mismatches,
        "popularityModel": pop_metrics,
        "daily": daily,
        "races": audit_races,
    }

    pop_model = {
        "version": "predictjra-popularity-calibration-20260815-16-v1",
        "trainedAt": datetime.now(JST).isoformat(timespec="seconds"),
        "teacherDates": list(TARGET_DATES),
        "teacherRaces": pop_metrics["teacherRaces"],
        "teacherRows": pop_metrics["teacherRows"],
        "features": pop_metrics["features"],
        "ridge": pop_metrics["ridge"],
        "coefficients": pop_metrics["futureModelCoefficients"],
        "validation": {
            "method": pop_metrics["method"],
            "meanAbsolutePopularityRankError": pop_metrics["meanAbsolutePopularityRankError"],
            "meanTop3OverlapRate": pop_metrics["meanTop3OverlapRate"],
        },
        "prohibitedInputs": [
            "current odds", "current actual popularity", "horse bodyweight", "horse bodyweight change"
        ],
    }

    args.data_path.parent.mkdir(parents=True, exist_ok=True)
    args.data_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.pop_model_path.write_text(
        json.dumps(pop_model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps({
        "raceCount": len(audit_races),
        "daily": daily,
        "oldResultMismatchCount": len(old_result_mismatches),
        "popularityMAE": pop_metrics["meanAbsolutePopularityRankError"],
        "top3Overlap": pop_metrics["meanTop3OverlapRate"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
