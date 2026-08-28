#!/usr/bin/env python3
"""Predictjra D3: leakage-safe single-win value reranker.

D3 deliberately separates *ability* from the market proxy:

  ability P(win)  <- horse/index features only (NO expected popularity)
  market P(win)   <- historical prior from expected-popularity rank + field bucket
  payout prior    <- robust historical winner payout by expected-popularity rank

The core value signals are:

  edge = ability P(win) / market P(win)
  payoutEV = ability P(win) * robust expected payout multiple
  d3EV = geometric blend of edge-based 80%-takeout proxy and payoutEV

Current-race odds, actual popularity, bodyweight and bodyweight change are never inputs.
Historical payouts/results are labels used only on dates strictly before a target date.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from typing import Iterable

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from single_win_d2 import (
    _clip01,
    _float,
    legacy_fallback_scores as d2_legacy_fallback_scores,
)

MODEL_VERSION = "predictjra-single-win-d3-v1.1-value-gate-80floor"

# Critical: expected popularity and legacy singleEV are intentionally excluded here.
# This model estimates horse ability/win chance independently of the market proxy.
ABILITY_FEATURE_COLS = [
    "recent_index",
    "current_run",
    "current_flow",
    "current_power",
    "today_index",
    "total_index",
    "total_rank_strength",
    "total_gap_strength",
    "recent_gap_strength",
    "today_gap_strength",
    "field_size_strength",
    "pace_fast",
    "pace_slow",
    "surface_turf",
    "surface_dirt",
    "surface_jump",
    "distance_strength",
]

TOP3_FEATURE_COLS = ABILITY_FEATURE_COLS


@dataclass(frozen=True)
class D3Policy:
    top_k: int = 4
    max_total_gap: float = 9.0
    min_win_ratio: float = 0.50
    min_top3_ratio: float = 0.45
    edge_power: float = 1.00
    ev_power: float = 0.50
    top3_power: float = 0.15
    ability_power: float = 0.15
    favorite_penalty: float = 0.00
    min_edge_to_switch: float = 1.08
    min_ev_to_switch: float = 0.72
    switch_margin: float = 1.08

    def to_dict(self) -> dict:
        return asdict(self)


def _x(rows: Iterable[dict], cols: list[str]) -> np.ndarray:
    return np.asarray([[float(r.get(c, 0.5)) for c in cols] for r in rows], dtype=float)


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except Exception:
        return None


def _field_bucket(field: int) -> int:
    f = int(field)
    if f <= 10:
        return 10
    if f <= 13:
        return 13
    if f <= 16:
        return 16
    return 18


def _rank_bucket(rank: int) -> int:
    r = max(1, int(rank))
    return min(r, 10)


def _weighted_mean(values: list[float], weights: list[float], default: float) -> float:
    if not values:
        return float(default)
    a = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    s = float(w.sum())
    if s <= 0:
        return float(default)
    return float(np.dot(a, w) / s)


class D3Model:
    """Ability/market-decoupled single-win model with recency adaptation."""

    def __init__(
        self,
        *,
        recency_half_life_days: float = 120.0,
        recent_window_days: int = 120,
        recent_blend: float = 0.30,
    ) -> None:
        self.recency_half_life_days = float(recency_half_life_days)
        self.recent_window_days = int(recent_window_days)
        self.recent_blend = float(recent_blend)

        self.win_model = HistGradientBoostingClassifier(
            max_iter=180,
            max_leaf_nodes=12,
            learning_rate=0.035,
            l2_regularization=10.0,
            min_samples_leaf=50,
            random_state=31,
        )
        self.win_recent_model = HistGradientBoostingClassifier(
            max_iter=150,
            max_leaf_nodes=10,
            learning_rate=0.035,
            l2_regularization=12.0,
            min_samples_leaf=42,
            random_state=37,
        )
        self.top3_model = HistGradientBoostingClassifier(
            max_iter=160,
            max_leaf_nodes=12,
            learning_rate=0.035,
            l2_regularization=9.0,
            min_samples_leaf=48,
            random_state=41,
        )

        self.fitted = False
        self.recent_fitted = False
        self.trained_through: str | None = None

        # Market prior: empirical win rates by expected-popularity rank.
        self.rank_market_rate: dict[int, float] = {}
        self.rank_field_market_rate: dict[tuple[int, int], float] = {}
        self.global_market_rate = 1.0 / 14.0

        # Robust payout priors among winners. Arithmetic mean is used after clipping,
        # because the target is expected yen return, not median payout.
        self.rank_payout_mean: dict[int, float] = {}
        self.rank_field_payout_mean: dict[tuple[int, int], float] = {}
        self.global_payout_mean = 4.0

    def _recency_weights(self, rows: list[dict]) -> np.ndarray:
        parsed = [_parse_date(str(r.get("date", ""))) for r in rows]
        valid = [d for d in parsed if d is not None]
        anchor = max(valid) if valid else None
        weights = []
        for row, d in zip(rows, parsed):
            race_balance = 1.0 / max(int(row.get("_field", 1)), 1)
            if anchor is None or d is None:
                decay = 1.0
            else:
                age = max((anchor - d).days, 0)
                decay = 0.5 ** (age / max(self.recency_half_life_days, 1.0))
                # Keep older races informative; avoid abrupt regime forgetting.
                decay = max(decay, 0.20)
            weights.append(race_balance * decay)
        arr = np.asarray(weights, dtype=float)
        arr *= len(arr) / max(float(arr.sum()), 1e-12)
        return arr

    def _fit_market_priors(self, rows: list[dict], recency_weights: np.ndarray) -> None:
        # Weighted sufficient statistics.
        rank_starts: dict[int, float] = defaultdict(float)
        rank_wins: dict[int, float] = defaultdict(float)
        bucket_starts: dict[tuple[int, int], float] = defaultdict(float)
        bucket_wins: dict[tuple[int, int], float] = defaultdict(float)

        winner_payouts: dict[int, list[float]] = defaultdict(list)
        winner_payout_weights: dict[int, list[float]] = defaultdict(list)
        winner_bucket_payouts: dict[tuple[int, int], list[float]] = defaultdict(list)
        winner_bucket_weights: dict[tuple[int, int], list[float]] = defaultdict(list)
        global_payouts: list[float] = []
        global_payout_weights: list[float] = []

        total_weight = 0.0
        winner_weight = 0.0
        for row, w in zip(rows, recency_weights):
            rank = _rank_bucket(int(row.get("_expected_popularity", row.get("_rank", 10))))
            field_bucket = _field_bucket(int(row.get("_field", 14)))
            key = (rank, field_bucket)
            is_win = int(row.get("is_winner", 0)) == 1

            rank_starts[rank] += float(w)
            bucket_starts[key] += float(w)
            total_weight += float(w)
            if is_win:
                rank_wins[rank] += float(w)
                bucket_wins[key] += float(w)
                winner_weight += float(w)
                payout = _float(row.get("win_payout_multiple"), 0.0)
                if payout > 0:
                    # Robust expected-value target: retain longshot information but cap the
                    # influence of extremely rare payouts.
                    payout = float(np.clip(payout, 1.0, 50.0))
                    winner_payouts[rank].append(payout)
                    winner_payout_weights[rank].append(float(w))
                    winner_bucket_payouts[key].append(payout)
                    winner_bucket_weights[key].append(float(w))
                    global_payouts.append(payout)
                    global_payout_weights.append(float(w))

        self.global_market_rate = winner_weight / max(total_weight, 1e-9)

        # Hierarchical beta-style shrinkage. Rank-level priors are strong enough to be
        # stable, while rank+field buckets get pulled toward rank-only estimates.
        for rank, starts in rank_starts.items():
            wins = rank_wins.get(rank, 0.0)
            prior_n = 35.0
            rate = (wins + prior_n * self.global_market_rate) / (starts + prior_n)
            self.rank_market_rate[rank] = max(float(rate), 1e-5)

        for key, starts in bucket_starts.items():
            rank = key[0]
            wins = bucket_wins.get(key, 0.0)
            rank_prior = self.rank_market_rate.get(rank, self.global_market_rate)
            prior_n = 22.0
            rate = (wins + prior_n * rank_prior) / (starts + prior_n)
            self.rank_field_market_rate[key] = max(float(rate), 1e-5)

        self.global_payout_mean = _weighted_mean(
            global_payouts,
            global_payout_weights,
            default=4.0,
        )
        self.global_payout_mean = float(np.clip(self.global_payout_mean, 1.2, 25.0))

        for rank, vals in winner_payouts.items():
            ws = winner_payout_weights[rank]
            raw = _weighted_mean(vals, ws, self.global_payout_mean)
            eff_n = max(sum(ws), 0.0)
            prior_n = 10.0
            shrunk = (eff_n * raw + prior_n * self.global_payout_mean) / (eff_n + prior_n)
            self.rank_payout_mean[rank] = float(np.clip(shrunk, 1.2, 35.0))

        for key, vals in winner_bucket_payouts.items():
            ws = winner_bucket_weights[key]
            rank_prior = self.rank_payout_mean.get(key[0], self.global_payout_mean)
            raw = _weighted_mean(vals, ws, rank_prior)
            eff_n = max(sum(ws), 0.0)
            prior_n = 8.0
            shrunk = (eff_n * raw + prior_n * rank_prior) / (eff_n + prior_n)
            self.rank_field_payout_mean[key] = float(np.clip(shrunk, 1.2, 40.0))

    def fit(self, rows: list[dict]) -> "D3Model":
        if not rows:
            raise ValueError("D3Model.fit: no rows")
        y_win = np.asarray([int(r.get("is_winner", 0)) for r in rows], dtype=int)
        y_top3 = np.asarray([int(r.get("is_top3", 0)) for r in rows], dtype=int)
        if y_win.sum() < 40 or len(np.unique(y_win)) < 2:
            raise ValueError("D3Model.fit: insufficient winner history")
        if y_top3.sum() < 100 or len(np.unique(y_top3)) < 2:
            raise ValueError("D3Model.fit: insufficient top3 history")

        Xw = _x(rows, ABILITY_FEATURE_COLS)
        weights = self._recency_weights(rows)
        self.win_model.fit(Xw, y_win, sample_weight=weights)
        self.top3_model.fit(_x(rows, TOP3_FEATURE_COLS), y_top3, sample_weight=weights)

        dates = [_parse_date(str(r.get("date", ""))) for r in rows]
        valid_dates = [d for d in dates if d is not None]
        anchor = max(valid_dates) if valid_dates else None
        recent_idx: list[int] = []
        if anchor is not None:
            for i, d in enumerate(dates):
                if d is not None and (anchor - d).days <= self.recent_window_days:
                    recent_idx.append(i)
        if len({rows[i]["race_id"] for i in recent_idx}) >= 300:
            recent_rows = [rows[i] for i in recent_idx]
            recent_y = y_win[recent_idx]
            if recent_y.sum() >= 30 and len(np.unique(recent_y)) == 2:
                recent_w = self._recency_weights(recent_rows)
                self.win_recent_model.fit(_x(recent_rows, ABILITY_FEATURE_COLS), recent_y, sample_weight=recent_w)
                self.recent_fitted = True

        self._fit_market_priors(rows, weights)
        self.fitted = True
        self.trained_through = max(str(r.get("date", "")) for r in rows)
        return self

    def score_race(self, rows: list[dict]) -> list[dict]:
        if not rows:
            return []
        if not self.fitted:
            raise ValueError("D3Model.score_race before fit")

        X = _x(rows, ABILITY_FEATURE_COLS)
        raw_global = np.clip(self.win_model.predict_proba(X)[:, 1], 1e-8, 1.0)
        if self.recent_fitted:
            raw_recent = np.clip(self.win_recent_model.predict_proba(X)[:, 1], 1e-8, 1.0)
            raw_win = (1.0 - self.recent_blend) * raw_global + self.recent_blend * raw_recent
        else:
            raw_win = raw_global
        # Exactly one winner's probability mass per race.
        p_win = raw_win / max(float(raw_win.sum()), 1e-12)

        raw_top3 = np.clip(self.top3_model.predict_proba(_x(rows, TOP3_FEATURE_COLS))[:, 1], 1e-8, 1.0)
        # Calibrate the race-level top3 mass to approximately three places.
        top3_mass = min(3.0, float(len(rows)))
        p_top3 = np.clip(raw_top3 * top3_mass / max(float(raw_top3.sum()), 1e-12), 1e-6, 0.999)

        market_raw = []
        payout_priors = []
        for row in rows:
            rank = _rank_bucket(int(row.get("_expected_popularity", row.get("_rank", 10))))
            fb = _field_bucket(int(row.get("_field", len(rows))))
            market = self.rank_field_market_rate.get(
                (rank, fb),
                self.rank_market_rate.get(rank, self.global_market_rate),
            )
            payout = self.rank_field_payout_mean.get(
                (rank, fb),
                self.rank_payout_mean.get(rank, self.global_payout_mean),
            )
            market_raw.append(max(float(market), 1e-6))
            payout_priors.append(float(payout))
        market_raw_arr = np.asarray(market_raw, dtype=float)
        p_market = market_raw_arr / max(float(market_raw_arr.sum()), 1e-12)

        out: list[dict] = []
        for i, row in enumerate(rows):
            edge = float(p_win[i] / max(float(p_market[i]), 1e-7))
            payout_ev = float(p_win[i] * payout_priors[i])
            # If the market proxy were perfectly efficient, 20% takeout would imply
            # ~0.80 * edge as expected return. Blend that with robust payout history.
            edge_ev = float(0.80 * edge)
            d3_ev = float(math.sqrt(max(edge_ev, 1e-9) * max(payout_ev, 1e-9)))
            scored = dict(row)
            scored.update({
                "d3_win_prob": float(p_win[i]),
                "d3_top3_prob": float(p_top3[i]),
                "d3_market_prob": float(p_market[i]),
                "d3_edge": edge,
                "d3_expected_payout_multiple": float(payout_priors[i]),
                "d3_payout_ev": payout_ev,
                "d3_edge_ev": edge_ev,
                "d3_ev": d3_ev,
                "singleD3": int(round(max(0.0, min(99.0, 50.0 + 30.0 * math.log(max(d3_ev, 1e-6) / 0.80))))),
            })
            out.append(scored)
        return out


def legacy_fallback_scores(rows: list[dict]) -> list[dict]:
    """Cold-start fallback; deterministic and market-decoupled enough for early history."""
    base = d2_legacy_fallback_scores(rows)
    if not base:
        return []
    # Rank-power market proxy. Only used before enough older races exist for training.
    market_weights = np.asarray([
        1.0 / ((max(1, int(r.get("_expected_popularity", r.get("_rank", 1)))) + 0.35) ** 1.05)
        for r in base
    ], dtype=float)
    p_market = market_weights / max(float(market_weights.sum()), 1e-12)
    out = []
    for r, pm in zip(base, p_market):
        pwin = max(_float(r.get("d2_win_prob"), 0.0), 1e-8)
        edge = pwin / max(float(pm), 1e-7)
        d3_ev = 0.80 * edge
        x = dict(r)
        x.update({
            "d3_win_prob": pwin,
            "d3_top3_prob": _float(r.get("d2_top3_prob"), 0.0),
            "d3_market_prob": float(pm),
            "d3_edge": float(edge),
            "d3_expected_payout_multiple": 0.80 / max(float(pm), 1e-4),
            "d3_payout_ev": float(d3_ev),
            "d3_edge_ev": float(d3_ev),
            "d3_ev": float(d3_ev),
            "singleD3": int(round(max(0.0, min(99.0, 50.0 + 30.0 * math.log(max(d3_ev, 1e-6) / 0.80))))),
        })
        out.append(x)
    return out


def choose_main(scored_rows: list[dict], selected_numbers: Iterable[int], policy: D3Policy) -> int:
    selected = {int(x) for x in selected_numbers}
    candidates = [r for r in scored_rows if int(r["horse_number"]) in selected]
    if not candidates:
        raise ValueError("D3 choose_main: selected set has no scored horses")

    ability_order = sorted(
        candidates,
        key=lambda r: (-_float(r.get("_total")), -_float(r.get("_recent")), int(r["horse_number"])),
    )
    best_total = _float(ability_order[0].get("_total"), 50.0)
    best_win = max(_float(r.get("d3_win_prob"), 0.0) for r in candidates)
    best_top3 = max(_float(r.get("d3_top3_prob"), 0.0) for r in candidates)
    topk = {int(r["horse_number"]) for r in ability_order[: max(1, int(policy.top_k))]}

    safe = [
        r for r in candidates
        if int(r["horse_number"]) in topk
        and best_total - _float(r.get("_total"), 50.0) <= float(policy.max_total_gap)
        and _float(r.get("d3_win_prob"), 0.0) >= best_win * float(policy.min_win_ratio)
        and _float(r.get("d3_top3_prob"), 0.0) >= best_top3 * float(policy.min_top3_ratio)
    ]
    if not safe:
        safe = [ability_order[0]]

    def reliability_score(r: dict) -> float:
        """Stable anchor used when no sufficiently strong value challenger exists.

        Because every eligible race must still buy one 100-yen win ticket, D3.1 avoids
        forcing a weak value bet. The anchor is driven mainly by model win probability,
        with top3/ability used only as modest stabilizers.
        """
        win = max(_float(r.get("d3_win_prob"), 0.0), 1e-9)
        top3 = max(_float(r.get("d3_top3_prob"), 0.0), 1e-9)
        gap = max(0.0, best_total - _float(r.get("_total"), 50.0))
        ability_rel = math.exp(-gap / 18.0)
        return win * (top3 ** 0.20) * (ability_rel ** 0.10)

    def value_score(r: dict) -> float:
        edge = max(_float(r.get("d3_edge"), 0.0), 1e-8)
        ev = max(_float(r.get("d3_ev"), 0.0), 1e-8)
        win_rel = max(_float(r.get("d3_win_prob"), 0.0) / max(best_win, 1e-9), 1e-6)
        top3_rel = max(_float(r.get("d3_top3_prob"), 0.0) / max(best_top3, 1e-9), 1e-6)
        gap = max(0.0, best_total - _float(r.get("_total"), 50.0))
        ability_rel = math.exp(-gap / 10.0)
        ep = max(1, int(r.get("_expected_popularity", 1)))
        favorite_factor = math.exp(-float(policy.favorite_penalty)) if ep == 1 else 1.0
        return (
            (edge ** float(policy.edge_power))
            * (ev ** float(policy.ev_power))
            * (win_rel ** 0.20)
            * (top3_rel ** float(policy.top3_power))
            * (ability_rel ** float(policy.ability_power))
            * favorite_factor
        )

    # D3.1 value gate: start from the safest model pick. Switch only when a challenger
    # has a real estimated market edge AND its value score clears a margin. This is
    # especially important when one bet is compulsory in every race.
    anchor = max(
        safe,
        key=lambda r: (
            reliability_score(r),
            _float(r.get("d3_win_prob")),
            _float(r.get("d3_top3_prob")),
            _float(r.get("_total")),
            -int(r["horse_number"]),
        ),
    )
    anchor_value = max(value_score(anchor), 1e-9)

    challengers = []
    for r in safe:
        if int(r["horse_number"]) == int(anchor["horse_number"]):
            continue
        edge = _float(r.get("d3_edge"), 0.0)
        ev = _float(r.get("d3_ev"), 0.0)
        if edge < float(policy.min_edge_to_switch):
            continue
        if ev < float(policy.min_ev_to_switch):
            continue

        # The farther down the expected-popularity ladder, the more evidence is required
        # before abandoning the stable anchor. This is not an odds input; expected
        # popularity is the pre-race model output already allowed by the project rules.
        ep = max(1, int(r.get("_expected_popularity", 1)))
        longshot_extra = max(0, ep - 4) * 0.025
        required_margin = float(policy.switch_margin) + longshot_extra
        if value_score(r) >= anchor_value * required_margin:
            challengers.append(r)

    if not challengers:
        return int(anchor["horse_number"])

    best = max(
        challengers,
        key=lambda r: (
            value_score(r),
            _float(r.get("d3_edge")),
            _float(r.get("d3_win_prob")),
            _float(r.get("_total")),
            -int(r["horse_number"]),
        ),
    )
    return int(best["horse_number"])
