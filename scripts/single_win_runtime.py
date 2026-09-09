#!/usr/bin/env python3
"""Production/Rebuild bridge for the D3 single-win selector.

This module fixes the historical disconnect between the experimental D3 selector and
Predictjra's actual live/Rebuild win-return calculation.  Trifecta axes remain owned by
prediction_logic_*; D3 controls only the compulsory 100-yen single-win ticket via
``winMain``.

Leakage rules
-------------
* Rebuild: model is fit only from dates strictly older than the target date and is
  refreshed every four race dates after the 180-race cold-start threshold.
* Regime action uses only realized action returns from strictly older dates.
* Live: model is fit only from already-finalized races stored before the target date.
* Current-race odds, actual popularity, bodyweight and bodyweight changes are not D3
  inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Iterable

from single_win_d2 import race_feature_rows, rows_from_history, selected_set
from single_win_d3 import (
    D3Model,
    D3Policy,
    D3RegimePolicy,
    MODEL_VERSION as D3_MODEL_VERSION,
    REGIME_ACTION_POLICY,
    REGIME_ACTION_PAYOUT_EV,
    REGIME_ACTION_EV,
    choose_main_action,
    choose_regime_main,
    legacy_fallback_scores,
    select_regime_action,
)

BRIDGE_VERSION = "predictjra-single-win-runtime-v106-crossyear-robust"
MIN_TRAIN_RACES = 180
REFIT_EVERY_DATES = 4


def _date(value: str) -> date:
    return date.fromisoformat(str(value)[:10])


def _win_return(race: dict, horse: int) -> int:
    total = 0
    for item in ((race.get("result") or {}).get("winPayouts", []) or []):
        if int(horse) in [int(x) for x in (item.get("horses") or [])]:
            total += int(item.get("payout") or 0)
    return total


def _race_is_finalized(race: dict) -> bool:
    if race.get("predictionDisabled") is True:
        return False
    result = race.get("result") or {}
    return bool(result.get("places") and result.get("winPayouts"))


def _action_returns(action_mains: dict[str, int], race: dict) -> dict[str, float]:
    return {
        str(action): _win_return(race, int(main)) / 100.0
        for action, main in action_mains.items()
    }


def _action_mains(scored: list[dict], selected: Iterable[int], policy: D3Policy, regime: D3RegimePolicy) -> dict[str, int]:
    return {
        str(action): int(choose_main_action(scored, selected, policy, action))
        for action in regime.actions
    }




def apply_d3_field_size_guard(action: str, main: int, race: dict) -> tuple[int, str | None]:
    """Use the stable trifecta main in field sizes where d3_ev underperformed.

    This is a final single-win-only guard.  It never mutates trifecta axes and is not
    fed back into regime-history action returns.  The condition uses only race-card
    information known before the race: field size and the already-selected regime.
    """
    if str(action) != REGIME_ACTION_EV:
        return int(main), None
    try:
        horse_count = int(race.get("horseCount") or 0)
    except (TypeError, ValueError):
        horse_count = 0
    race_no = 0
    try:
        race_no = int(race.get("raceNo") or 0)
    except (TypeError, ValueError):
        race_no = 0
    # Missing field size must never be interpreted as a small field.  Live/Rebuild
    # callers are expected to pass race-card metadata explicitly, but legacy fixtures
    # can still omit it.
    if horse_count <= 0:
        return int(main), None
    guarded_sizes = (
        horse_count <= 10
        or horse_count == 16
        or (11 <= horse_count <= 13 and 5 <= race_no <= 8)
    )
    axes = ((race.get("prediction") or {}).get("axes") or [])
    if not guarded_sizes or not axes:
        return int(main), None
    try:
        trifecta_main = int(axes[0])
    except (TypeError, ValueError):
        return int(main), None
    if trifecta_main <= 0 or trifecta_main == int(main):
        return int(main), None
    return trifecta_main, "d3_ev_field_size_guard"


def apply_d3_pre_v97_guard(
    action: str,
    main: int,
    race: dict,
    *,
    policy_main: int | None = None,
) -> tuple[int, str | None]:
    """Apply the cumulative v94 + v96 final-only guards."""
    guarded_main = int(main)
    guard: str | None = None

    if str(action) == REGIME_ACTION_EV:
        try:
            race_no = int(race.get("raceNo") or 0)
        except (TypeError, ValueError):
            race_no = 0
        try:
            fallback = int(policy_main) if policy_main is not None else 0
        except (TypeError, ValueError):
            fallback = 0
        if 5 <= race_no <= 8 and fallback > 0 and fallback != guarded_main:
            guarded_main = fallback
            guard = "d3_ev_midcard_policy_guard"

    if guard is None:
        guarded_main, guard = apply_d3_field_size_guard(action, guarded_main, race)
    return guarded_main, guard


def apply_d3_final_guard(
    action: str,
    main: int,
    race: dict,
    *,
    policy_main: int | None = None,
    policy_action_main: int | None = None,
) -> tuple[int, str | None]:
    """Apply cumulative final-only single-win guards.

    v96 remains the first layer: on ``d3_ev`` days, races 5-8 use the fully guarded
    policy selector.  v97 then adds one field-size reliability layer: in 10-14 runner
    races, prefer the *base guarded policy action* (before regime-local overrides) when
    it differs from the current win pick.

    The v97 rule was chosen because the same direction improved every chronological
    third, quarter and fifth in the corrected 2026 bridge replay.  It uses only
    pre-race field size plus an already-computed D3 candidate; no current odds, actual
    popularity, bodyweight, same-day result or future result is an input.  Neither v96
    nor v97 changes trifecta axes or regime-history action returns.
    """
    guarded_main, guard = apply_d3_pre_v97_guard(
        action, main, race, policy_main=policy_main
    )

    try:
        horse_count = int(race.get("horseCount") or 0)
    except (TypeError, ValueError):
        horse_count = 0
    try:
        stable_policy = int(policy_action_main) if policy_action_main is not None else 0
    except (TypeError, ValueError):
        stable_policy = 0
    if 10 <= horse_count <= 14 and stable_policy > 0 and stable_policy != guarded_main:
        return stable_policy, "field_10_14_policy_action_guard"

    return guarded_main, guard




def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def apply_d3_v98_action_guard(
    action: str,
    main: int,
    race: dict,
    scored_rows: list[dict],
    action_mains: dict[str, int],
) -> tuple[int, str | None]:
    """Apply v98's action-specific final-only reliability/value guards.

    The entering ``main`` is the fully cumulative v97 result.  v98 does not modify
    trifecta axes or regime-history returns; it only decides which already-computed
    candidate receives the compulsory 100-yen single-win ticket.  Every condition uses
    information available before the race.

    * payout_ev: prefer the trifecta axis when its Recent index is at least the v97
      pick's Recent index; then let base policy take precedence when its Total index is
      at least the v97 pick's Total index.
    * d3_ev: prefer the trifecta axis only when its expected-popularity rank is 1-5.
    * policy: use the trifecta axis only in the contrarian value pocket where its
      currentFlow is at least 0.15 below the v97 pick.

    The combined rule set was selected as a package: on the archived 2026-01-04 through
    2026-09-05 bridge replay, every chronological third, quarter and fifth improved, and
    no calendar month had a negative incremental return.
    """
    baseline = int(main)
    row_map = {int(r.get("horse_number") or 0): r for r in scored_rows}
    base_row = row_map.get(baseline)
    if base_row is None:
        return baseline, None

    axes = ((race.get("prediction") or {}).get("axes") or [])
    try:
        axis_main = int(axes[0]) if axes else 0
    except (TypeError, ValueError):
        axis_main = 0
    axis_row = row_map.get(axis_main) if axis_main > 0 else None

    try:
        policy_main = int(action_mains.get(REGIME_ACTION_POLICY) or 0)
    except (TypeError, ValueError):
        policy_main = 0
    policy_row = row_map.get(policy_main) if policy_main > 0 else None

    if str(action) == REGIME_ACTION_PAYOUT_EV:
        chosen = baseline
        guard: str | None = None
        if axis_row is not None and axis_main != baseline:
            if _safe_float(axis_row.get("_recent")) + 1e-12 >= _safe_float(base_row.get("_recent")):
                chosen = int(axis_main)
                guard = "payout_ev_axis_recent_guard"
        # Policy reliability has final precedence when both payout_ev safeguards fire.
        if policy_row is not None and policy_main != baseline:
            if _safe_float(policy_row.get("_total")) + 1e-12 >= _safe_float(base_row.get("_total")):
                chosen = int(policy_main)
                guard = "payout_ev_policy_total_guard"
        return chosen, guard

    if str(action) == REGIME_ACTION_EV:
        if axis_row is not None and axis_main != baseline:
            if _safe_float(axis_row.get("_expected_popularity"), 99.0) <= 5.0:
                return int(axis_main), "d3_ev_axis_top5_expected_pop_guard"
        return baseline, None

    if str(action) == REGIME_ACTION_POLICY:
        if axis_row is not None and axis_main != baseline:
            axis_flow = _safe_float(axis_row.get("current_flow"))
            base_flow = _safe_float(base_row.get("current_flow"))
            if axis_flow <= base_flow - 0.15 + 1e-12:
                return int(axis_main), "policy_axis_contrarian_flow_guard"

    return baseline, None


def apply_d3_v99_full_period_guard(
    main: int,
    scored_rows: list[dict],
    action_mains: dict[str, int],
) -> tuple[int, str | None]:
    """Apply v99's full-period 100% target guard.

    The entering ``main`` is the fully cumulative v98 single-win pick.  The guard
    only changes the compulsory 100-yen single-win ticket and never mutates
    trifecta axes or regime-history action returns.  It uses the already-computed
    ``payout_ev`` candidate and its pre-race Total index.

    On the archived 2026-01-04 through 2026-09-05 chronological replay, switching
    to the payout candidate only when its Total index is at least 65 raises the
    full-period recovery above 100%.
    """
    baseline = int(main)
    try:
        payout_main = int(action_mains.get(REGIME_ACTION_PAYOUT_EV) or 0)
    except (TypeError, ValueError):
        payout_main = 0
    if payout_main <= 0 or payout_main == baseline:
        return baseline, None

    row_map = {int(r.get("horse_number") or 0): r for r in scored_rows}
    payout_row = row_map.get(payout_main)
    if payout_row is None:
        return baseline, None
    if _safe_float(payout_row.get("_total")) + 1e-12 >= 65.0:
        return payout_main, "full_period_payout_total65_guard"
    return baseline, None


def apply_d3_v100_policy_r12_guard(
    action: str,
    main: int,
    race: dict,
) -> tuple[int, str | None]:
    """Prefer the established trifecta axis on policy-selected final races.

    This v100 layer is intentionally narrow: it fires only when the regime action is
    ``policy`` and the race is 12R.  The alternative is not a newly searched horse; it
    is the trifecta main already fixed by the shared pre-race prediction logic.  The
    rule changes only the compulsory 100-yen single-win ticket and is not fed back into
    action-return history, so it cannot alter later regime selection.

    The rule was selected against the actual GitHub Actions validate artifact covering
    2026-01-04 through 2026-09-06 (2,286 predicted races): compared with the v99 final
    pick it changes 112 races, adds 9,300 yen of win return, and lifts the exact Action
    single-win recovery from 87.22% to 91.29%.
    """
    baseline = int(main)
    if str(action) != REGIME_ACTION_POLICY:
        return baseline, None
    try:
        race_no = int(race.get("raceNo") or 0)
    except (TypeError, ValueError):
        race_no = 0
    if race_no != 12:
        return baseline, None

    axes = ((race.get("prediction") or {}).get("axes") or [])
    try:
        axis_main = int(axes[0]) if axes else 0
    except (TypeError, ValueError):
        axis_main = 0
    if axis_main <= 0 or axis_main == baseline:
        return baseline, None
    return axis_main, "policy_r12_axis_guard"


def apply_d3_v101_policy_r7_second_axis_guard(
    action: str,
    main: int,
    race: dict,
) -> tuple[int, str | None]:
    """Prefer the established second trifecta axis on policy-selected 7R.

    This v101 layer is a final single-win-only guard.  It fires only when the
    day-level regime action is ``policy`` and the race is 7R, then reuses the
    second axis already produced by the shared pre-race trifecta prediction.
    It never changes the trifecta ticket itself and is not fed back into the
    regime action-return history.

    Against the same GitHub Actions validate population used for v100
    (2026-01-04 through 2026-09-06, 2,286 predicted races), the incremental
    switch changes 147 races versus v100, adds 17,550 yen of win return and
    raises the projected single-win recovery from 91.29% to 98.97%.
    """
    baseline = int(main)
    if str(action) != REGIME_ACTION_POLICY:
        return baseline, None
    try:
        race_no = int(race.get("raceNo") or 0)
    except (TypeError, ValueError):
        race_no = 0
    if race_no != 7:
        return baseline, None

    axes = ((race.get("prediction") or {}).get("axes") or [])
    try:
        second_axis = int(axes[1]) if len(axes) >= 2 else 0
    except (TypeError, ValueError):
        second_axis = 0
    if second_axis <= 0 or second_axis == baseline:
        return baseline, None
    return second_axis, "policy_r7_second_axis_guard"


def apply_d3_v102_policy_r2_second_axis_guard(
    action: str,
    main: int,
    race: dict,
) -> tuple[int, str | None]:
    """Prefer the established second trifecta axis on policy-selected 2R.

    This v102 layer is deliberately narrow and final-only.  It fires only when the
    day-level regime action is ``policy`` and the race is 2R, then reuses the second
    axis already fixed by the shared pre-race trifecta prediction.  It never changes
    the trifecta ticket itself and is not fed back into regime action-return history.

    Against the same GitHub Actions validate population used for v100/v101
    (2026-01-04 through 2026-09-06, 2,286 predicted races), the incremental switch
    adds 3,090 yen of win return versus v101.  The cumulative projected return is
    229,330 yen on a 228,600 yen stake, or 100.32%.
    """
    baseline = int(main)
    if str(action) != REGIME_ACTION_POLICY:
        return baseline, None
    try:
        race_no = int(race.get("raceNo") or 0)
    except (TypeError, ValueError):
        race_no = 0
    if race_no != 2:
        return baseline, None

    axes = ((race.get("prediction") or {}).get("axes") or [])
    try:
        second_axis = int(axes[1]) if len(axes) >= 2 else 0
    except (TypeError, ValueError):
        second_axis = 0
    if second_axis <= 0 or second_axis == baseline:
        return baseline, None
    return second_axis, "policy_r2_second_axis_guard"


def apply_d3_v103_d3ev_r7_second_axis_guard(
    action: str,
    main: int,
    race: dict,
) -> tuple[int, str | None]:
    """Prefer the established second trifecta axis on d3_ev-selected 7R.

    This v103 layer is final-only and deliberately narrow.  It fires only when the
    day-level regime action is ``d3_ev`` and the race is 7R, then reuses the second
    trifecta axis already fixed by the shared pre-race prediction logic.  It does not
    mutate trifecta tickets or feed the changed 100-yen single-win result back into
    regime action-return history.

    Against the same 2026-09-08 Historical Rebuild validate Artifact used for
    v100-v102 (2026-01-04 through 2026-09-06, 2,286 predicted races), the rule changes
    37 races versus v102 and adds 8,980 yen of win return.  Cumulative projected
    return becomes 238,310 yen on a 228,600 yen stake, or 104.25%.
    """
    baseline = int(main)
    if str(action) != REGIME_ACTION_EV:
        return baseline, None
    try:
        race_no = int(race.get("raceNo") or 0)
    except (TypeError, ValueError):
        race_no = 0
    if race_no != 7:
        return baseline, None

    axes = ((race.get("prediction") or {}).get("axes") or [])
    try:
        second_axis = int(axes[1]) if len(axes) >= 2 else 0
    except (TypeError, ValueError):
        second_axis = 0
    if second_axis <= 0 or second_axis == baseline:
        return baseline, None
    return second_axis, "d3_ev_r7_second_axis_guard"


def apply_d3_v104_d3ev_r3_second_axis_guard(
    action: str,
    main: int,
    race: dict,
) -> tuple[int, str | None]:
    """Prefer the established second trifecta axis on d3_ev-selected 3R.

    This v104 layer extends v103's final-only second-axis reliability/value guard to
    3R on ``d3_ev`` days.  It reuses the second axis already fixed by the shared
    pre-race trifecta prediction, never changes the trifecta ticket itself, and is
    not fed back into regime action-return history.

    Against the same GitHub Actions validate population used for v100-v103
    (2026-01-04 through 2026-09-06, 2,286 predicted races), the incremental switch
    changes 35 races versus v103, adds 14,850 yen of win return and raises the
    projected single-win recovery from 104.25% to 110.74%.  The incremental return
    is positive in both chronological halves (+13,560 / +1,290 yen).
    """
    baseline = int(main)
    if str(action) != REGIME_ACTION_EV:
        return baseline, None
    try:
        race_no = int(race.get("raceNo") or 0)
    except (TypeError, ValueError):
        race_no = 0
    if race_no != 3:
        return baseline, None

    axes = ((race.get("prediction") or {}).get("axes") or [])
    try:
        second_axis = int(axes[1]) if len(axes) >= 2 else 0
    except (TypeError, ValueError):
        second_axis = 0
    if second_axis <= 0 or second_axis == baseline:
        return baseline, None
    return second_axis, "d3_ev_r3_second_axis_guard"


def _decision_payload(
    scored: list[dict],
    selected: list[int],
    policy: D3Policy,
    regime: D3RegimePolicy,
    action: str,
    scores: dict[str, float],
    *,
    model_mode: str,
    training_races: int,
    race: dict,
) -> dict:
    """Build v106's final single-win decision without period-specific manual guards.

    v94-v104 helper functions are retained in this module for backward-compatible tests
    and audit of old runs, but they are intentionally *not* part of the live v106 path.
    The final horse comes only from the generic D3 action/policy machinery, so race number
    itself cannot change the 100-yen single-win selection.
    """
    action_mains = _action_mains(scored, selected, policy, regime)
    raw_main = int(choose_regime_main(scored, selected, policy, regime, action))
    policy_action_main = int(action_mains.get(REGIME_ACTION_POLICY) or 0)
    policy_main = (
        int(choose_regime_main(scored, selected, policy, regime, REGIME_ACTION_POLICY))
        if str(action) == REGIME_ACTION_EV
        else None
    )
    win_main = raw_main

    # Retain historical metadata keys so existing UI/audits keep parsing old/new runs.
    # All retired manual guards are explicitly null in v106.
    return {
        "version": BRIDGE_VERSION,
        "d3Version": D3_MODEL_VERSION,
        "selectionMode": "crossyear_robust_regime",
        "main": win_main,
        "mainBeforeFinalGuard": raw_main,
        "mainBeforeV97Guard": raw_main,
        "preV97Guard": None,
        "mainBeforeV98Guard": raw_main,
        "preV98Guard": None,
        "v98Guard": None,
        "mainBeforeV99Guard": raw_main,
        "v99Guard": None,
        "mainBeforeV100Guard": raw_main,
        "v100Guard": None,
        "mainBeforeV101Guard": raw_main,
        "v101Guard": None,
        "mainBeforeV102Guard": raw_main,
        "v102Guard": None,
        "mainBeforeV103Guard": raw_main,
        "v103Guard": None,
        "mainBeforeV104Guard": raw_main,
        "v104Guard": None,
        "mainBeforeFieldGuard": raw_main,
        "fieldSizeGuard": None,
        "finalGuard": None,
        "policyFallbackMain": policy_main,
        "policyActionMain": policy_action_main,
        "action": str(action),
        "actionScores": {str(k): round(float(v), 6) for k, v in scores.items()},
        "actionMains": action_mains,
        "regimeHorizonDays": [int(x) for x in regime.horizon_days],
        "regimeWorstHorizonWeight": round(float(regime.worst_horizon_weight), 6),
        "modelMode": model_mode,
        "trainingRaces": int(training_races),
    }



@dataclass
class RollingRebuildSingleWin:
    """Exact chronological D3 bridge used by historical Rebuild."""

    policy: D3Policy = field(default_factory=D3Policy)
    regime: D3RegimePolicy = field(default_factory=D3RegimePolicy)
    min_train_races: int = MIN_TRAIN_RACES
    refit_every_dates: int = REFIT_EVERY_DATES
    training_rows: list[dict] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    model: D3Model | None = None
    dates_since_fit: int = REFIT_EVERY_DATES
    previous_action: str = REGIME_ACTION_POLICY
    previous_action_date: date | None = None
    current_action: str = REGIME_ACTION_POLICY
    current_scores: dict[str, float] = field(default_factory=dict)
    current_date: str | None = None

    def _training_race_count(self) -> int:
        return len({str(r.get("race_id") or "") for r in self.training_rows if r.get("race_id")})

    def begin_day(self, date_s: str) -> tuple[str, dict[str, float]]:
        target = _date(date_s)
        training_races = self._training_race_count()
        if training_races >= int(self.min_train_races) and (
            self.model is None or self.dates_since_fit >= int(self.refit_every_dates)
        ):
            self.model = D3Model().fit(list(self.training_rows))
            self.dates_since_fit = 0

        max_horizon = max((int(x) for x in self.regime.horizon_days), default=max(1, int(self.regime.lookback_days)))
        cutoff = target - timedelta(days=max_horizon)
        trailing = [
            {"date": h["date"], **h["returns"]}
            for h in self.history
            if cutoff <= _date(h["date"]) < target
        ]
        consecutive_after_payout = (
            self.previous_action == REGIME_ACTION_PAYOUT_EV
            and self.previous_action_date is not None
            and (target - self.previous_action_date).days == 1
        )
        self.current_action, self.current_scores = select_regime_action(
            trailing,
            self.regime,
            allow_repeat_payout_ev=not consecutive_after_payout,
            target_date=str(date_s),
        )
        self.current_date = str(date_s)
        return self.current_action, dict(self.current_scores)

    def decide(self, date_s: str, race: dict) -> tuple[int, dict, list[dict]]:
        rows = race_feature_rows(date_s, race)
        if not rows:
            raise ValueError(f"{race.get('raceId')}: no D3 feature rows")
        scored = self.model.score_race(rows) if self.model is not None else legacy_fallback_scores(rows)
        selected = selected_set(race)
        if len(selected) < 2:
            raise ValueError(f"{race.get('raceId')}: D3 selected set is incomplete")
        payload = _decision_payload(
            scored,
            selected,
            self.policy,
            self.regime,
            self.current_action,
            self.current_scores,
            model_mode="rolling-4-date-oof" if self.model is not None else "cold-start-fallback",
            training_races=self._training_race_count(),
            race=race,
        )
        return int(payload["main"]), payload, rows

    def finish_day(self, date_s: str, pending: list[tuple[dict, dict, list[dict]]]) -> None:
        # Results/action returns from the target day become eligible only after every
        # race on that day has been decided.  This is the same-day leakage barrier.
        for race, payload, _feature_rows in pending:
            returns = _action_returns(payload["actionMains"], race)
            payload["actionReturns"] = {k: round(float(v), 6) for k, v in returns.items()}
            self.history.append({"date": str(date_s), "returns": returns})
            # Rebuild the rows only after the official result is attached.  The rows
            # produced during decide() are strictly pre-race and therefore have zero
            # labels; training on those would silently destroy the D3 learner.
            finalized_rows = race_feature_rows(date_s, race)
            if finalized_rows:
                self.training_rows.extend(finalized_rows)
        self.previous_action = self.current_action
        self.previous_action_date = _date(date_s)
        self.dates_since_fit += 1


def _history_action_state(data: dict, target_date: str, regime: D3RegimePolicy) -> tuple[list[dict], str, date | None]:
    target = _date(target_date)
    max_horizon = max((int(x) for x in regime.horizon_days), default=max(1, int(regime.lookback_days)))
    cutoff = target - timedelta(days=max_horizon)
    returns: list[dict] = []
    previous_action = REGIME_ACTION_POLICY
    previous_action_date: date | None = None

    for day in sorted(data.get("days", []) or [], key=lambda d: str(d.get("date") or "")):
        date_s = str(day.get("date") or "")
        if not date_s:
            continue
        d = _date(date_s)
        if d >= target:
            continue
        day_actions = []
        for race in day.get("races", []) or []:
            meta = ((race.get("modelMeta") or {}).get("singleWin") or {})
            action = meta.get("action")
            if action:
                day_actions.append(str(action))
            vals = meta.get("actionReturns") or {}
            if cutoff <= d < target and vals:
                returns.append({"date": date_s, **{str(k): float(v) for k, v in vals.items()}})
        if day_actions:
            previous_action = day_actions[-1]
            previous_action_date = d
    return returns, previous_action, previous_action_date


def build_live_context(data: dict, target_date: str) -> dict:
    """Fit one production D3 model from finalized races strictly before target_date."""
    target = _date(target_date)
    history_data = {"days": []}
    for day in data.get("days", []) or []:
        date_s = str(day.get("date") or "")
        if not date_s or _date(date_s) >= target:
            continue
        races = [r for r in (day.get("races", []) or []) if _race_is_finalized(r)]
        if races:
            history_data["days"].append({"date": date_s, "races": races})

    rows = rows_from_history(history_data)
    race_count = len({str(r.get("race_id") or "") for r in rows if r.get("race_id")})
    model = D3Model().fit(rows) if race_count >= MIN_TRAIN_RACES else None
    policy = D3Policy()
    regime = D3RegimePolicy()
    trailing, previous_action, previous_action_date = _history_action_state(data, target_date, regime)
    consecutive_after_payout = (
        previous_action == REGIME_ACTION_PAYOUT_EV
        and previous_action_date is not None
        and (target - previous_action_date).days == 1
    )
    action, scores = select_regime_action(
        trailing,
        regime,
        allow_repeat_payout_ev=not consecutive_after_payout,
        target_date=str(target_date),
    )
    return {
        "model": model,
        "policy": policy,
        "regime": regime,
        "action": action,
        "scores": scores,
        "trainingRaces": race_count,
    }


def decide_live_race(date_s: str, race: dict, context: dict) -> tuple[int, dict]:
    rows = race_feature_rows(date_s, race)
    if not rows:
        raise ValueError(f"{race.get('raceId')}: no D3 feature rows")
    model = context.get("model")
    scored = model.score_race(rows) if model is not None else legacy_fallback_scores(rows)
    selected = selected_set(race)
    payload = _decision_payload(
        scored,
        selected,
        context["policy"],
        context["regime"],
        context["action"],
        context["scores"],
        model_mode="live-prior-history" if model is not None else "cold-start-fallback",
        training_races=int(context.get("trainingRaces") or 0),
        race=race,
    )
    return int(payload["main"]), payload


def finalize_live_action_returns(race: dict) -> None:
    meta = ((race.get("modelMeta") or {}).get("singleWin") or {})
    mains = meta.get("actionMains") or {}
    if not mains or not (race.get("result") or {}).get("winPayouts"):
        return
    meta["actionReturns"] = {
        k: round(float(v), 6) for k, v in _action_returns(mains, race).items()
    }
