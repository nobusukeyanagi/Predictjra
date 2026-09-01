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

MODEL_VERSION = "predictjra-single-win-d3-v2.1-asymmetric-challenger-support"

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
    min_win_ratio: float = 0.55
    min_top3_ratio: float = 0.56
    edge_power: float = 1.00
    ev_power: float = 0.50
    top3_power: float = 0.15
    ability_power: float = 0.15
    favorite_penalty: float = 0.00
    min_edge_to_switch: float = 1.05
    min_ev_to_switch: float = 0.72
    switch_margin: float = 1.11
    min_relative_edge: float = 1.04
    min_anchor_win_support: float = 0.61
    max_challenger_expected_popularity: int = 8
    min_tail_anchor_win_support: float = 0.80
    max_value_switch_win_ratio: float = 1.00
    max_value_switch_distance_m: float = 2000.0
    avoid_equal_total_value_switch: bool = True
    max_near_tie_today_deficit: float = 1.0
    avoid_total_recent_disagreement_switch: bool = True
    avoid_total_run_double_deficit_switch: bool = True
    min_total_deficit_for_run_guard: float = 1.0
    min_current_run_deficit_for_guard: float = 0.11

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
            prior_n = 50.0
            rate = (wins + prior_n * self.global_market_rate) / (starts + prior_n)
            self.rank_market_rate[rank] = max(float(rate), 1e-5)

        # Expected-popularity ranks should be monotone in win chance. Sparse history can
        # otherwise create false "value" when a lower rank accidentally has a higher
        # empirical win rate. Enforce only the economically obvious ordering.
        prev = None
        for rank in sorted(self.rank_market_rate):
            cur = self.rank_market_rate[rank]
            if prev is not None:
                cur = min(cur, prev)
                self.rank_market_rate[rank] = cur
            prev = cur

        for key, starts in bucket_starts.items():
            rank = key[0]
            wins = bucket_wins.get(key, 0.0)
            rank_prior = self.rank_market_rate.get(rank, self.global_market_rate)
            prior_n = 80.0
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
            prior_n = 16.0
            shrunk = (eff_n * raw + prior_n * self.global_payout_mean) / (eff_n + prior_n)
            self.rank_payout_mean[rank] = float(np.clip(shrunk, 1.2, 35.0))

        for key, vals in winner_bucket_payouts.items():
            ws = winner_bucket_weights[key]
            rank_prior = self.rank_payout_mean.get(key[0], self.global_payout_mean)
            raw = _weighted_mean(vals, ws, rank_prior)
            eff_n = max(sum(ws), 0.0)
            prior_n = 24.0
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
            rank_market = self.rank_market_rate.get(rank, self.global_market_rate)
            bucket_market = self.rank_field_market_rate.get((rank, fb), rank_market)
            # Field buckets are useful but noisier than rank-only history. Keep them as a
            # modest adjustment rather than letting a small bucket manufacture an edge.
            market = 0.75 * float(rank_market) + 0.25 * float(bucket_market)

            rank_payout = self.rank_payout_mean.get(rank, self.global_payout_mean)
            bucket_payout = self.rank_field_payout_mean.get((rank, fb), rank_payout)
            payout = 0.80 * float(rank_payout) + 0.20 * float(bucket_payout)
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
            # D3.2 trusts the structurally stable ability-vs-market edge more than the
            # noisier payout prior. A 70/30 geometric blend reduces jackpot overfitting.
            d3_ev = float(
                (max(edge_ev, 1e-9) ** 0.70)
                * (max(payout_ev, 1e-9) ** 0.30)
            )
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
    #
    # D3.11 uses asymmetric challenger support. After zero-base and feature-ablation
    # checks, the time/pace/ability (時・展・実) signals remain useful; the remaining
    # weakness is the one-size-fits-all support floor. Expected-popularity ranks 1-7 may
    # challenge with 61% of the anchor's win probability, but the deepest allowed rank 8
    # must retain 80%. At the same time the safe candidate pool itself is strengthened to
    # 55% of the race-best win probability and 56% of the best top3 probability. This
    # opens credible mid-ranked value while making tail promotions more conservative.
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
    anchor_edge = max(_float(anchor.get("d3_edge"), 0.0), 1e-9)
    anchor_win = max(_float(anchor.get("d3_win_prob"), 0.0), 1e-9)

    challengers = []
    for r in safe:
        if int(r["horse_number"]) == int(anchor["horse_number"]):
            continue
        edge = _float(r.get("d3_edge"), 0.0)
        ev = _float(r.get("d3_ev"), 0.0)
        if edge < float(policy.min_edge_to_switch):
            continue
        # An absolute edge is not enough: the challenger must also be better value than
        # the stable anchor and retain a meaningful share of the anchor's win chance.
        if edge < anchor_edge * float(policy.min_relative_edge):
            continue

        # D3.4 tail win-support: the deepest still-allowed value challenger (expected
        # popularity rank 8+) must preserve substantially more of the stable anchor's
        # model win probability. This targets the high-variance edge of the v73 tail
        # without banning a rank-8 horse when it is itself the reliability anchor.
        ep = max(1, int(r.get("_expected_popularity", 1)))
        required_win_support = float(policy.min_anchor_win_support)
        if ep >= 8:
            required_win_support = max(
                required_win_support,
                float(policy.min_tail_anchor_win_support),
            )
        if _float(r.get("d3_win_prob"), 0.0) < anchor_win * required_win_support:
            continue
        if ev < float(policy.min_ev_to_switch):
            continue

        # The farther down the expected-popularity ladder, the more evidence is required
        # before abandoning the stable anchor. This is not an odds input; expected
        # popularity is the pre-race model output already allowed by the project rules.
        # D3.3 tail-risk cap: deep expected-popularity ranks have sparse, noisy priors.
        # They may still be selected when they are the reliability anchor; this gate only
        # blocks speculative value-driven promotions away from that anchor.
        if ep > int(policy.max_challenger_expected_popularity):
            continue
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

    # D3.5 model-consensus guard: D3 is a *value reranker*. If the strongest value
    # challenger already exceeds the reliability anchor in raw P(win), yet still loses
    # the multi-signal reliability score, the win model is disagreeing with top3/ability.
    # Tune-only temporal blocks showed these near/over-anchor switches to be unstable, so
    # keep the anchor instead of letting the market-edge prior act as a second ability
    # selector. A ratio of 1.00 means only true over-anchor cases are vetoed.
    challenger_win_ratio = _float(best.get("d3_win_prob"), 0.0) / max(anchor_win, 1e-9)
    if challenger_win_ratio > float(policy.max_value_switch_win_ratio):
        return int(anchor["horse_number"])

    # D3.5 long-distance regime guard: the current feature/model stack is not distance-
    # specialized enough for 2100m+ races. Across all three tune-only chronological
    # blocks, value-driven switches above 2000m underperformed the reliability anchor.
    # Until a dedicated long-distance model has enough history, avoid the speculative
    # rerank while still allowing the same horse when it is itself the anchor.
    distance_m = _float(best.get("distance_strength"), 0.0) * 3600.0
    if distance_m > float(policy.max_value_switch_distance_m):
        return int(anchor["horse_number"])

    # D3.6 near-tie anchor guard: after the broad v73-v75 risk controls, the remaining
    # unstable switches are concentrated in races where the challenger is not clearly
    # better on the core pre-race ability indices.  Tune-only chronological blocks show
    # that market-value reranking is especially noisy when total ability is exactly tied,
    # or when the reliability anchor is only one Today-index point ahead.  In these
    # near-tie states, keep the multi-signal anchor and demand that value reranking prove
    # itself in less ambiguous ability configurations.
    anchor_total = _float(anchor.get("_total"), 0.0)
    challenger_total = _float(best.get("_total"), 0.0)
    if bool(policy.avoid_equal_total_value_switch) and abs(anchor_total - challenger_total) <= 1e-9:
        return int(anchor["horse_number"])

    anchor_today = _float(anchor.get("_today"), 0.0)
    challenger_today = _float(best.get("_today"), 0.0)
    today_deficit = anchor_today - challenger_today
    near_tie_limit = max(0.0, float(policy.max_near_tie_today_deficit))
    if near_tie_limit > 0.0 and 0.0 < today_deficit <= near_tie_limit + 1e-9:
        return int(anchor["horse_number"])

    # D3.7 recent-consensus guard: a challenger whose total ability index is above the
    # anchor but whose recent index is below it is being promoted despite conflicting
    # ability signals.  On tune-only chronological checks, these value-driven switches
    # underperformed the reliability anchor and also reduced top3 stability.  Keep the
    # anchor in this disagreement state; a horse can still be selected normally when it
    # is itself the reliability anchor.
    anchor_recent = _float(anchor.get("_recent"), 0.0)
    challenger_recent = _float(best.get("_recent"), 0.0)
    if (
        bool(policy.avoid_total_recent_disagreement_switch)
        and challenger_total > anchor_total + 1e-9
        and challenger_recent < anchor_recent - 1e-9
    ):
        return int(anchor["horse_number"])

    # D3.8 run-consensus guard: after the v77 recent-index disagreement filter, a small
    # residual loss cluster remained where the value challenger was weaker on BOTH the
    # total ability index and the immediate currentRun component.  The 10-point boundary
    # was unstable, so the guard deliberately starts at an 11-point currentRun deficit.
    # This keeps borderline 10-point cases under the existing value gate while preventing
    # market-value evidence from overriding a clear two-signal ability deficit.
    anchor_run = _float(anchor.get("current_run"), 0.0)
    challenger_run = _float(best.get("current_run"), 0.0)
    total_deficit = anchor_total - challenger_total
    run_deficit = anchor_run - challenger_run
    if (
        bool(policy.avoid_total_run_double_deficit_switch)
        and total_deficit >= float(policy.min_total_deficit_for_run_guard) - 1e-9
        and run_deficit >= float(policy.min_current_run_deficit_for_guard) - 1e-9
    ):
        return int(anchor["horse_number"])

    return int(best["horse_number"])
