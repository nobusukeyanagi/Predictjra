#!/usr/bin/env python3
"""YS4 D2: leakage-safe single-win (単勝) main-pick model.

This module is intentionally isolated from production selection logic.  It consumes only
fields already persisted in each race's pre-race ``modelMeta.indexDetail`` plus the
estimated-popularity rank.  Current-race odds / actual popularity / bodyweight are never
features.

The learning target is decomposed into three parts:
  1. P(win)
  2. P(top3) -- used as a stability guard, not as the return objective
  3. E(win payout multiple | win)

Expected single-win value is then
    EV = normalized P(win) * E(payout multiple | win)

A policy layer selects the main horse only inside the already-selected YS4 horse set, so
危険馬 and the selected-horse count are left untouched.  The policy is tuned on historical
out-of-fold predictions and checked on a later untouched holdout period.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

MODEL_VERSION = "predictjra-single-win-d2-v1"

# Every feature below is available before the race and is present/derivable from indexDetail.
FEATURE_COLS = [
    "recent_index",
    "current_run",
    "current_flow",
    "current_power",
    "today_index",
    "total_index",
    "total_rank_strength",
    "estimated_popularity_strength",
    "total_gap_strength",
    "recent_gap_strength",
    "today_gap_strength",
    "field_size_strength",
    "single_ev_legacy_strength",
    "pace_fast",
    "pace_slow",
    "surface_turf",
    "surface_dirt",
    "surface_jump",
    "distance_strength",
]


@dataclass(frozen=True)
class Policy:
    top_k: int = 4
    max_total_gap: float = 8.0
    min_win_ratio: float = 0.50
    min_top3_ratio: float = 0.50
    top3_power: float = 0.20
    ability_power: float = 0.15
    legacy_power: float = 0.15

    def to_dict(self) -> dict:
        return asdict(self)


def _float(value, default: float = 0.0) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _surface_flags(surface: str) -> tuple[float, float, float]:
    s = str(surface or "")
    if "障" in s:
        return 0.0, 0.0, 1.0
    if "ダ" in s:
        return 0.0, 1.0, 0.0
    return 1.0, 0.0, 0.0


def _pace_flags(pace: str) -> tuple[float, float]:
    p = str(pace or "").lower()
    return float(p == "fast"), float(p == "slow")


def _winner_map(result: dict) -> dict[int, int]:
    """Return horse->100-yen win payout. Handles dead heats."""
    out: dict[int, int] = {}
    for item in (result or {}).get("winPayouts", []) or []:
        payout = int(_float(item.get("payout"), 0.0))
        for no in item.get("horses", []) or []:
            out[int(no)] = payout
    return out


def _top3_set(result: dict) -> set[int]:
    out: set[int] = set()
    for group in ((result or {}).get("places", []) or [])[:3]:
        out.update(int(x) for x in group)
    return out


def _winner_set(result: dict) -> set[int]:
    places = ((result or {}).get("places", []) or [])
    return set(int(x) for x in places[0]) if places else set()


def selected_set(race: dict) -> list[int]:
    p = race.get("prediction") or {}
    nums = [int(x) for x in (p.get("axes", []) or [])]
    nums += [int(x) for x in (p.get("opponents", []) or [])]
    # Preserve order while removing accidental duplicates.
    return list(dict.fromkeys(nums))


def race_feature_rows(date_s: str, race: dict) -> list[dict]:
    """Build one all-pre-race feature row per non-scratched horse."""
    if race.get("predictionDisabled") is True:
        return []
    detail = ((race.get("modelMeta") or {}).get("indexDetail") or {})
    horses = detail.get("horses") or []
    if not horses:
        return []

    nonstarters = set(int(x) for x in ((race.get("modelMeta") or {}).get("nonStarters") or []))
    usable = [h for h in horses if int(h.get("no", -1)) not in nonstarters]
    if len(usable) < 2:
        return []

    result = race.get("result") or {}
    winners = _winner_set(result)
    top3 = _top3_set(result)
    payouts = _winner_map(result)
    selected = set(selected_set(race))
    danger = set(int(x) for x in (race.get("danger") or []))

    max_total = max(_float(h.get("total"), 50.0) for h in usable)
    max_recent = max(_float(h.get("recentIndex"), 50.0) for h in usable)
    max_today = max(_float(h.get("today"), 50.0) for h in usable)
    field = len(usable)
    denom = max(field - 1, 1)
    surface = ((detail.get("raceConditions") or {}).get("surface") or "")
    distance = _float((detail.get("raceConditions") or {}).get("distanceM"), 1600.0)
    pace_fast, pace_slow = _pace_flags(detail.get("paceRegime"))
    turf, dirt, jump = _surface_flags(surface)

    rows: list[dict] = []
    for h in usable:
        no = int(h["no"])
        total = _float(h.get("total"), 50.0)
        recent = _float(h.get("recentIndex"), 50.0)
        today = _float(h.get("today"), 50.0)
        rank = max(1, int(_float(h.get("rank"), field)))
        ep = max(1, int(_float(h.get("expectedPopularity"), rank)))
        legacy_ev = _float(h.get("singleEV"), 50.0)
        payout = int(payouts.get(no, 0))
        rows.append({
            "date": str(date_s),
            "race_id": str(race.get("raceId") or ""),
            "horse_number": no,
            "selected": int(no in selected),
            "danger": int(no in danger),
            "is_winner": int(no in winners),
            "is_top3": int(no in top3),
            "win_payout": payout,
            "win_payout_multiple": payout / 100.0 if payout > 0 else 0.0,
            "recent_index": _clip01(recent / 100.0),
            "current_run": _clip01(_float(h.get("currentRun"), 50.0) / 100.0),
            "current_flow": _clip01(_float(h.get("currentFlow"), 50.0) / 100.0),
            "current_power": _clip01(_float(h.get("currentPower"), 50.0) / 100.0),
            "today_index": _clip01(today / 100.0),
            "total_index": _clip01(total / 100.0),
            "total_rank_strength": _clip01(1.0 - (rank - 1) / denom),
            "estimated_popularity_strength": _clip01(1.0 - (ep - 1) / denom),
            "total_gap_strength": _clip01(1.0 - (max_total - total) / 25.0),
            "recent_gap_strength": _clip01(1.0 - (max_recent - recent) / 30.0),
            "today_gap_strength": _clip01(1.0 - (max_today - today) / 30.0),
            "field_size_strength": _clip01(field / 18.0),
            "single_ev_legacy_strength": _clip01(legacy_ev / 99.0),
            "pace_fast": pace_fast,
            "pace_slow": pace_slow,
            "surface_turf": turf,
            "surface_dirt": dirt,
            "surface_jump": jump,
            "distance_strength": _clip01(distance / 3600.0),
            # Raw values retained only for policy guards/reporting.
            "_total": total,
            "_recent": recent,
            "_today": today,
            "_rank": rank,
            "_expected_popularity": ep,
            "_legacy_ev": legacy_ev,
            "_field": field,
        })
    return rows


def rows_from_history(data: dict) -> list[dict]:
    rows: list[dict] = []
    for day in data.get("days", []) or []:
        date_s = str(day.get("date") or "")
        for race in day.get("races", []) or []:
            rows.extend(race_feature_rows(date_s, race))
    return rows


def _x(rows: Iterable[dict]) -> np.ndarray:
    return np.asarray([[float(r.get(c, 0.5)) for c in FEATURE_COLS] for r in rows], dtype=float)


class D2Model:
    """Three-model decomposition used by D2."""

    def __init__(self) -> None:
        self.win_model = HistGradientBoostingClassifier(
            max_iter=160,
            max_leaf_nodes=10,
            learning_rate=0.035,
            l2_regularization=8.0,
            min_samples_leaf=55,
            random_state=17,
        )
        self.top3_model = HistGradientBoostingClassifier(
            max_iter=140,
            max_leaf_nodes=12,
            learning_rate=0.035,
            l2_regularization=7.0,
            min_samples_leaf=50,
            random_state=19,
        )
        self.payout_model = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=140,
            max_leaf_nodes=8,
            learning_rate=0.035,
            l2_regularization=10.0,
            min_samples_leaf=18,
            random_state=23,
        )
        self.fitted = False
        self.payout_fitted = False
        self.rank_log_payout_prior: dict[int, float] = {}
        self.global_log_payout_prior = math.log(4.0)

    def fit(self, rows: list[dict]) -> "D2Model":
        if not rows:
            raise ValueError("D2Model.fit: no rows")
        X = _x(rows)
        y_win = np.asarray([int(r["is_winner"]) for r in rows], dtype=int)
        y_top3 = np.asarray([int(r["is_top3"]) for r in rows], dtype=int)
        if y_win.sum() < 40 or len(np.unique(y_win)) < 2:
            raise ValueError("D2Model.fit: insufficient winner history")
        if y_top3.sum() < 100 or len(np.unique(y_top3)) < 2:
            raise ValueError("D2Model.fit: insufficient top3 history")

        # Race-balanced weights: one large field must not dominate one small field.
        sample_weight = np.asarray([1.0 / max(int(r.get("_field", 1)), 1) for r in rows], dtype=float)
        sample_weight *= len(sample_weight) / sample_weight.sum()
        self.win_model.fit(X, y_win, sample_weight=sample_weight)
        self.top3_model.fit(X, y_top3, sample_weight=sample_weight)
        self.fitted = True

        winners = [r for r in rows if int(r.get("is_winner", 0)) == 1 and _float(r.get("win_payout_multiple"), 0.0) > 0]
        if winners:
            logs = [math.log(max(_float(r["win_payout_multiple"], 1.0), 1.01)) for r in winners]
            self.global_log_payout_prior = float(np.median(logs))
            by_rank: dict[int, list[float]] = {}
            for r, logp in zip(winners, logs):
                rank = min(8, max(1, int(r.get("_expected_popularity", 8))))
                by_rank.setdefault(rank, []).append(logp)
            self.rank_log_payout_prior = {
                rank: float(np.median(vals)) for rank, vals in by_rank.items() if vals
            }

        if len(winners) >= 80:
            Xp = _x(winners)
            yp = np.asarray(
                [math.log(max(float(r["win_payout_multiple"]), 1.01)) for r in winners],
                dtype=float,
            )
            self.payout_model.fit(Xp, yp)
            self.payout_fitted = True
        return self

    def score_race(self, rows: list[dict]) -> list[dict]:
        if not rows:
            return []
        if not self.fitted:
            raise ValueError("D2Model.score_race before fit")
        X = _x(rows)
        raw_win = np.clip(self.win_model.predict_proba(X)[:, 1], 1e-6, 1.0)
        # A race has one winner in normal cases. Normalize to a race-level probability mass.
        p_win = raw_win / max(float(raw_win.sum()), 1e-9)
        p_top3 = np.clip(self.top3_model.predict_proba(X)[:, 1], 1e-5, 1.0)

        if self.payout_fitted:
            model_log_payout = self.payout_model.predict(X)
        else:
            model_log_payout = np.full(len(rows), self.global_log_payout_prior, dtype=float)

        out: list[dict] = []
        for i, row in enumerate(rows):
            rank = min(8, max(1, int(row.get("_expected_popularity", 8))))
            prior = self.rank_log_payout_prior.get(rank, self.global_log_payout_prior)
            # Shrink the heavy-tailed payout model toward a rank-bucket median.
            log_payout = 0.68 * float(model_log_payout[i]) + 0.32 * float(prior)
            payout_multiple = float(np.clip(math.exp(log_payout), 1.2, 80.0))
            ev = float(p_win[i] * payout_multiple)
            scored = dict(row)
            scored.update({
                "d2_win_prob": float(p_win[i]),
                "d2_top3_prob": float(p_top3[i]),
                "d2_expected_payout_multiple": payout_multiple,
                "d2_ev": ev,
                # 80% is a natural neutral point for a pari-mutuel market with ~20% takeout.
                "singleD2": int(round(max(0.0, min(99.0, 50.0 + 30.0 * math.log(max(ev, 1e-6) / 0.80))))),
            })
            out.append(scored)
        return out


def legacy_fallback_scores(rows: list[dict]) -> list[dict]:
    """Cold-start scoring before enough older races exist for D2 training."""
    if not rows:
        return []
    weights = np.asarray([math.exp((_float(r.get("_total"), 50.0) - 50.0) / 8.0) for r in rows])
    probs = weights / max(float(weights.sum()), 1e-9)
    out = []
    for row, p in zip(rows, probs):
        legacy = _clip01(_float(row.get("_legacy_ev"), 50.0) / 99.0)
        # Not an ROI estimate; only a deterministic ranking fallback.
        ev = 0.55 + 0.70 * legacy
        x = dict(row)
        x.update({
            "d2_win_prob": float(p),
            "d2_top3_prob": _clip01(0.25 + 0.70 * _float(row.get("total_rank_strength"), 0.5)),
            "d2_expected_payout_multiple": float(max(1.2, ev / max(float(p), 1e-4))),
            "d2_ev": float(ev),
            "singleD2": int(round(_float(row.get("_legacy_ev"), 50.0))),
        })
        out.append(x)
    return out


def choose_main(scored_rows: list[dict], selected_numbers: Iterable[int], policy: Policy) -> int:
    selected = {int(x) for x in selected_numbers}
    candidates = [r for r in scored_rows if int(r["horse_number"]) in selected]
    if not candidates:
        raise ValueError("choose_main: selected set has no scored horses")

    ability_order = sorted(candidates, key=lambda r: (-_float(r.get("_total")), -_float(r.get("_recent")), int(r["horse_number"])))
    best_total = _float(ability_order[0].get("_total"), 50.0)
    best_win = max(_float(r.get("d2_win_prob"), 0.0) for r in candidates)
    best_top3 = max(_float(r.get("d2_top3_prob"), 0.0) for r in candidates)
    topk = {int(r["horse_number"]) for r in ability_order[: max(1, int(policy.top_k))]}

    safe = [
        r for r in candidates
        if int(r["horse_number"]) in topk
        and best_total - _float(r.get("_total"), 50.0) <= float(policy.max_total_gap)
        and _float(r.get("d2_win_prob"), 0.0) >= best_win * float(policy.min_win_ratio)
        and _float(r.get("d2_top3_prob"), 0.0) >= best_top3 * float(policy.min_top3_ratio)
    ]
    if not safe:
        safe = [ability_order[0]]

    def score(r: dict) -> float:
        ev = max(_float(r.get("d2_ev"), 0.0), 1e-9)
        top3_rel = max(_float(r.get("d2_top3_prob"), 0.0) / max(best_top3, 1e-9), 1e-6)
        gap = max(0.0, best_total - _float(r.get("_total"), 50.0))
        ability_rel = math.exp(-gap / 10.0)
        legacy_rel = math.exp((_float(r.get("_legacy_ev"), 50.0) - 50.0) / 30.0)
        return (
            ev
            * (top3_rel ** float(policy.top3_power))
            * (ability_rel ** float(policy.ability_power))
            * (legacy_rel ** float(policy.legacy_power))
        )

    best = max(
        safe,
        key=lambda r: (score(r), _float(r.get("d2_win_prob")), _float(r.get("_total")), -int(r["horse_number"])),
    )
    return int(best["horse_number"])


def choose_second(selected_numbers: Iterable[int], main: int, horse_rows: list[dict]) -> int:
    selected = {int(x) for x in selected_numbers if int(x) != int(main)}
    candidates = [r for r in horse_rows if int(r["horse_number"]) in selected]
    if not candidates:
        raise ValueError("choose_second: no candidate")
    # Same rule as current YS4: lowest estimated popularity (largest rank) excluding main.
    return int(max(
        candidates,
        key=lambda r: (int(r.get("_expected_popularity", 1)), _float(r.get("_total")), -int(r["horse_number"])),
    )["horse_number"])


def trifecta_return(race: dict, selected_numbers: Iterable[int], main: int, second: int) -> int:
    selected = set(int(x) for x in selected_numbers)
    opponents = selected - {int(main), int(second)}
    if not opponents:
        return 0
    total = 0
    for item in ((race.get("result") or {}).get("trifectas", []) or []):
        horses = set(int(x) for x in (item.get("horses") or []))
        if {int(main), int(second)}.issubset(horses) and bool(opponents & horses):
            total += int(_float(item.get("payout"), 0.0))
    return total
