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
    choose_main_action,
    choose_regime_main,
    legacy_fallback_scores,
    select_regime_action,
)

BRIDGE_VERSION = "predictjra-single-win-runtime-v91"
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
) -> dict:
    action_mains = _action_mains(scored, selected, policy, regime)
    win_main = int(choose_regime_main(scored, selected, policy, regime, action))
    return {
        "version": BRIDGE_VERSION,
        "d3Version": D3_MODEL_VERSION,
        "main": win_main,
        "action": str(action),
        "actionScores": {str(k): round(float(v), 6) for k, v in scores.items()},
        "actionMains": action_mains,
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

        cutoff = target - timedelta(days=max(1, int(self.regime.lookback_days)))
        trailing = [
            h["returns"]
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


def _history_action_state(data: dict, target_date: str, regime: D3RegimePolicy) -> tuple[list[dict[str, float]], str, date | None]:
    target = _date(target_date)
    cutoff = target - timedelta(days=max(1, int(regime.lookback_days)))
    returns: list[dict[str, float]] = []
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
                returns.append({str(k): float(v) for k, v in vals.items()})
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
