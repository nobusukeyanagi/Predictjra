#!/usr/bin/env python3
from __future__ import annotations

from datetime import date, timedelta

from single_win_d3 import (
    ABILITY_FEATURE_COLS,
    D3Model,
    D3Policy,
    D3RegimePolicy,
    REGIME_ACTION_EV,
    REGIME_ACTION_PAYOUT_EV,
    REGIME_ACTION_POLICY,
    choose_main,
    choose_main_action,
    select_regime_action,
)


def row(day: str, rid: str, no: int, rank: int, ep: int, winner: bool, top3: bool, payout: int = 0) -> dict:
    field = 8
    total = 88 - (rank - 1) * 4
    base = {
        "date": day,
        "race_id": rid,
        "horse_number": no,
        "selected": int(no <= 4),
        "danger": 0,
        "is_winner": int(winner),
        "is_top3": int(top3),
        "win_payout": payout,
        "win_payout_multiple": payout / 100.0 if payout else 0.0,
        "recent_index": (total - 2) / 100,
        "current_run": 0.65 + 0.02 * (5 - rank),
        "current_flow": 0.62 + 0.02 * (5 - rank),
        "current_power": 0.66 + 0.02 * (5 - rank),
        "today_index": (total - 1) / 100,
        "total_index": total / 100,
        "total_rank_strength": 1 - (rank - 1) / 7,
        "estimated_popularity_strength": 1 - (ep - 1) / 7,
        "total_gap_strength": 1 - (88 - total) / 25,
        "recent_gap_strength": 1 - (90 - (total - 2)) / 30,
        "today_gap_strength": 1 - (89 - (total - 1)) / 30,
        "field_size_strength": field / 18,
        "single_ev_legacy_strength": 0.5,
        "pace_fast": 0.0,
        "pace_slow": 0.0,
        "surface_turf": 1.0,
        "surface_dirt": 0.0,
        "surface_jump": 0.0,
        "distance_strength": 1600 / 3600,
        "_total": total,
        "_recent": total - 2,
        "_today": total - 1,
        "_rank": rank,
        "_expected_popularity": ep,
        "_legacy_ev": 50,
        "_field": field,
    }
    return base


def synthetic_history() -> list[dict]:
    rows = []
    start = date(2026, 1, 1)
    # 90 races x 8 horses = enough for model contracts.
    for i in range(90):
        day = str(start + timedelta(days=i))
        rid = f"R{i:03d}"
        # Alternate winner between ability rank 1 and rank 2; rank 2 is often market rank 4,
        # which creates a learnable ability-vs-market edge without using current odds.
        winner_rank = 1 if i % 3 else 2
        for rank in range(1, 9):
            ep = rank
            if rank == 2:
                ep = 4
            elif rank in (3, 4):
                ep = rank - 1
            is_win = rank == winner_rank
            payout = (180 if winner_rank == 1 else 620) if is_win else 0
            rows.append(row(day, rid, rank, rank, ep, is_win, rank <= 3, payout))
    return rows


def main() -> None:
    # D3.11 contract: asymmetric challenger support. Mid-ranked value horses get a
    # slightly wider path, while the rank-8 tail and the safe candidate pool are stricter.
    # These defaults are cumulative with every v73-v80 guard.
    assert abs(D3Policy().min_edge_to_switch - 1.05) < 1e-12
    assert abs(D3Policy().switch_margin - 1.11) < 1e-12
    assert abs(D3Policy().min_anchor_win_support - 0.61) < 1e-12
    assert abs(D3Policy().min_tail_anchor_win_support - 0.80) < 1e-12
    assert abs(D3Policy().min_win_ratio - 0.55) < 1e-12
    assert abs(D3Policy().min_top3_ratio - 0.56) < 1e-12

    assert "estimated_popularity_strength" not in ABILITY_FEATURE_COLS
    assert "single_ev_legacy_strength" not in ABILITY_FEATURE_COLS
    forbidden = {"current_odds", "actual_popularity", "bodyweight", "bodyweight_change"}
    assert not forbidden.intersection(ABILITY_FEATURE_COLS)

    hist = synthetic_history()
    model = D3Model(recent_window_days=45).fit(hist)
    target = [r for r in hist if r["race_id"] == "R089"]
    scored = model.score_race(target)
    assert len(scored) == 8
    assert abs(sum(r["d3_win_prob"] for r in scored) - 1.0) < 1e-8
    assert abs(sum(r["d3_market_prob"] for r in scored) - 1.0) < 1e-8
    assert all(r["d3_edge"] > 0 for r in scored)
    assert all(0 <= r["singleD3"] <= 99 for r in scored)
    assert sum(r["d3_top3_prob"] for r in scored) <= 3.001

    selected = [1, 2, 3, 4]
    main_pick = choose_main(scored, selected, D3Policy(top_k=4, max_total_gap=20))
    assert main_pick in selected

    # D3.1 value-gate contract: an absurdly high switch threshold must keep the stable
    # reliability anchor rather than forcing a speculative value challenger.
    locked_policy = D3Policy(
        top_k=4,
        max_total_gap=20,
        min_edge_to_switch=99.0,
        min_ev_to_switch=99.0,
        switch_margin=99.0,
    )
    locked_pick = choose_main(scored, selected, locked_policy)
    safe_scored = [r for r in scored if r["horse_number"] in selected]
    best_win = max(r["d3_win_prob"] for r in safe_scored)
    locked_row = next(r for r in safe_scored if r["horse_number"] == locked_pick)
    assert locked_row["d3_win_prob"] >= best_win * 0.80

    # Market decoupling contract: rank-2 horse is assigned weaker market rank, so if its
    # ability probability is comparable, its edge must exceed a similarly capable favorite.
    by_no = {r["horse_number"]: r for r in scored}
    assert by_no[2]["d3_market_prob"] < by_no[1]["d3_market_prob"]

    # D3.2 relative-edge guard: a challenger that has a decent absolute edge but does not
    # beat the anchor's edge by the requested ratio must not force a switch.
    strict_relative = D3Policy(
        top_k=4,
        max_total_gap=20,
        min_edge_to_switch=0.0,
        min_ev_to_switch=0.0,
        switch_margin=0.0,
        min_relative_edge=9.0,
        min_anchor_win_support=0.0,
    )
    relative_locked = choose_main(scored, selected, strict_relative)
    assert relative_locked == locked_pick

    # D3.3 tail-risk cap: a deep expected-popularity challenger must not replace a
    # reliable anchor only because its value score is extreme. Relaxing the cap proves
    # the gate, rather than the base score, is what blocks the switch.
    manual = [
        {
            "horse_number": 1, "_total": 90.0, "_recent": 90.0,
            "_expected_popularity": 1, "d3_win_prob": 0.35,
            "d3_top3_prob": 0.70, "d3_edge": 1.00, "d3_ev": 0.80,
        },
        {
            "horse_number": 2, "_total": 88.0, "_recent": 88.0,
            "_expected_popularity": 9, "d3_win_prob": 0.30,
            "d3_top3_prob": 0.65, "d3_edge": 2.50, "d3_ev": 2.00,
        },
    ]
    permissive = dict(
        top_k=2, max_total_gap=20, min_win_ratio=0.0, min_top3_ratio=0.0,
        edge_power=1.0, ev_power=1.0, top3_power=0.0, ability_power=0.0,
        min_edge_to_switch=0.0, min_ev_to_switch=0.0, switch_margin=0.0,
        min_relative_edge=0.0, min_anchor_win_support=0.0,
    )
    assert choose_main(manual, [1, 2], D3Policy(**permissive)) == 1
    assert choose_main(
        manual, [1, 2],
        D3Policy(**permissive, max_challenger_expected_popularity=10, min_tail_anchor_win_support=0.0),
    ) == 2


    # D3.4 tail win-support: the deepest still-allowed value challenger (rank 8)
    # must retain 76% of the anchor's model win chance before a value switch.
    # Relaxing only this new floor proves that the D3.4 gate blocks the switch.
    tail_support = [
        {
            "horse_number": 1, "_total": 90.0, "_recent": 90.0,
            "_expected_popularity": 1, "d3_win_prob": 0.35,
            "d3_top3_prob": 0.70, "d3_edge": 1.00, "d3_ev": 0.80,
        },
        {
            "horse_number": 2, "_total": 88.0, "_recent": 88.0,
            "_expected_popularity": 8, "d3_win_prob": 0.26,
            "d3_top3_prob": 0.65, "d3_edge": 2.80, "d3_ev": 2.20,
        },
    ]
    assert choose_main(tail_support, [1, 2], D3Policy(**permissive)) == 1
    assert choose_main(
        tail_support, [1, 2],
        D3Policy(**permissive, min_tail_anchor_win_support=0.0),
    ) == 2


    # D3.5 model-consensus guard: if a value challenger has raw P(win) above the
    # reliability anchor but still loses the multi-signal reliability score, the value
    # reranker must not override the anchor. Relaxing only this guard proves the veto.
    consensus = [
        {
            "horse_number": 1, "_total": 90.0, "_recent": 90.0,
            "_expected_popularity": 1, "d3_win_prob": 0.35,
            "d3_top3_prob": 0.75, "d3_edge": 1.00, "d3_ev": 0.80,
            "distance_strength": 1600 / 3600,
        },
        {
            "horse_number": 2, "_total": 87.0, "_recent": 87.0,
            "_expected_popularity": 4, "d3_win_prob": 0.36,
            "d3_top3_prob": 0.30, "d3_edge": 3.00, "d3_ev": 2.50,
            "distance_strength": 1600 / 3600,
        },
    ]
    assert choose_main(consensus, [1, 2], D3Policy(**permissive)) == 1
    assert choose_main(
        consensus, [1, 2],
        D3Policy(**permissive, max_value_switch_win_ratio=9.0),
    ) == 2

    # D3.5 long-distance regime guard: above 2000m, a value-driven promotion is
    # suppressed, but relaxing only the distance guard restores the same challenger.
    long_distance = [
        {
            "horse_number": 1, "_total": 90.0, "_recent": 90.0,
            "_expected_popularity": 1, "d3_win_prob": 0.35,
            "d3_top3_prob": 0.70, "d3_edge": 1.00, "d3_ev": 0.80,
            "distance_strength": 2200 / 3600,
        },
        {
            "horse_number": 2, "_total": 88.0, "_recent": 88.0,
            "_expected_popularity": 6, "d3_win_prob": 0.30,
            "d3_top3_prob": 0.65, "d3_edge": 2.80, "d3_ev": 2.20,
            "distance_strength": 2200 / 3600,
        },
    ]
    assert choose_main(long_distance, [1, 2], D3Policy(**permissive)) == 1
    assert choose_main(
        long_distance, [1, 2],
        D3Policy(**permissive, max_value_switch_distance_m=9999.0),
    ) == 2

    # D3.6 equal-total near-tie guard: when the reliability anchor and value challenger
    # have exactly the same total ability index, keep the anchor. Disabling only this
    # guard must restore the otherwise-valid value switch.
    equal_total = [
        {
            "horse_number": 1, "_total": 90.0, "_recent": 90.0, "_today": 90.0,
            "_expected_popularity": 1, "d3_win_prob": 0.35,
            "d3_top3_prob": 0.70, "d3_edge": 1.00, "d3_ev": 0.80,
            "distance_strength": 1600 / 3600,
        },
        {
            "horse_number": 2, "_total": 90.0, "_recent": 88.0, "_today": 91.0,
            "_expected_popularity": 6, "d3_win_prob": 0.30,
            "d3_top3_prob": 0.64, "d3_edge": 2.80, "d3_ev": 2.20,
            "distance_strength": 1600 / 3600,
        },
    ]
    assert choose_main(equal_total, [1, 2], D3Policy(**permissive)) == 1
    assert choose_main(
        equal_total, [1, 2],
        D3Policy(**permissive, avoid_equal_total_value_switch=False),
    ) == 2

    # D3.6 Today-index near-tie guard: if the anchor is only one point ahead on Today,
    # avoid a market-value-only promotion. Setting the guard width to zero restores the
    # value challenger and proves that this new veto is the deciding condition.
    today_near_tie = [
        {
            "horse_number": 1, "_total": 90.0, "_recent": 90.0, "_today": 90.0,
            "_expected_popularity": 1, "d3_win_prob": 0.35,
            "d3_top3_prob": 0.70, "d3_edge": 1.00, "d3_ev": 0.80,
            "distance_strength": 1600 / 3600,
        },
        {
            "horse_number": 2, "_total": 88.0, "_recent": 88.0, "_today": 89.0,
            "_expected_popularity": 6, "d3_win_prob": 0.30,
            "d3_top3_prob": 0.64, "d3_edge": 2.80, "d3_ev": 2.20,
            "distance_strength": 1600 / 3600,
        },
    ]
    assert choose_main(today_near_tie, [1, 2], D3Policy(**permissive)) == 1
    assert choose_main(
        today_near_tie, [1, 2],
        D3Policy(**permissive, max_near_tie_today_deficit=0.0),
    ) == 2

    # D3.7 recent-consensus guard: if a value challenger is stronger on total ability
    # but weaker on the recent index, keep the reliability anchor rather than letting
    # market value override conflicting ability signals. Disabling only this guard must
    # restore the challenger.
    recent_disagreement = [
        {
            "horse_number": 1, "_total": 88.0, "_recent": 92.0, "_today": 88.0,
            "_expected_popularity": 1, "d3_win_prob": 0.35,
            "d3_top3_prob": 0.72, "d3_edge": 1.00, "d3_ev": 0.80,
            "distance_strength": 1600 / 3600,
        },
        {
            "horse_number": 2, "_total": 90.0, "_recent": 88.0, "_today": 92.0,
            "_expected_popularity": 5, "d3_win_prob": 0.30,
            "d3_top3_prob": 0.64, "d3_edge": 2.80, "d3_ev": 2.20,
            "distance_strength": 1600 / 3600,
        },
    ]
    assert choose_main(recent_disagreement, [1, 2], D3Policy(**permissive)) == 1
    assert choose_main(
        recent_disagreement, [1, 2],
        D3Policy(**permissive, avoid_total_recent_disagreement_switch=False),
    ) == 2

    # D3.8 run-consensus guard: if the challenger is at least one total-index point
    # weaker AND its currentRun component is at least 11 points weaker, keep the anchor.
    # A challenger only 10 currentRun points behind remains eligible, which pins the
    # deliberately conservative boundary selected by chronological checks.
    run_double_deficit = [
        {
            "horse_number": 1, "_total": 90.0, "_recent": 88.0, "_today": 88.0,
            "_expected_popularity": 1, "d3_win_prob": 0.35,
            "d3_top3_prob": 0.70, "d3_edge": 1.00, "d3_ev": 0.80,
            "distance_strength": 1600 / 3600, "current_run": 0.70,
        },
        {
            "horse_number": 2, "_total": 88.0, "_recent": 90.0, "_today": 92.0,
            "_expected_popularity": 5, "d3_win_prob": 0.30,
            "d3_top3_prob": 0.64, "d3_edge": 2.80, "d3_ev": 2.20,
            "distance_strength": 1600 / 3600, "current_run": 0.59,
        },
    ]
    assert choose_main(run_double_deficit, [1, 2], D3Policy(**permissive)) == 1
    assert choose_main(
        run_double_deficit, [1, 2],
        D3Policy(**permissive, avoid_total_run_double_deficit_switch=False),
    ) == 2

    run_ten_point_boundary = [dict(run_double_deficit[0]), dict(run_double_deficit[1])]
    run_ten_point_boundary[1]["current_run"] = 0.60
    assert choose_main(run_ten_point_boundary, [1, 2], D3Policy(**permissive)) == 2

    # D3.10 controlled-challenger expansion: a challenger retaining about 67% of the
    # anchor's model win probability is now eligible when the other value/stability
    # evidence is strong. The former 70% floor still blocks the exact same setup,
    # proving that this is a deliberate boundary relaxation rather than a guard bypass.
    support_expansion = [
        {
            "horse_number": 1, "_total": 90.0, "_recent": 90.0, "_today": 90.0,
            "_expected_popularity": 1, "d3_win_prob": 0.35,
            "d3_top3_prob": 0.70, "d3_edge": 1.00, "d3_ev": 0.80,
            "distance_strength": 1600 / 3600, "current_run": 0.70,
        },
        {
            "horse_number": 2, "_total": 88.0, "_recent": 90.0, "_today": 92.0,
            "_expected_popularity": 5, "d3_win_prob": 0.235,
            "d3_top3_prob": 0.62, "d3_edge": 3.00, "d3_ev": 2.40,
            "distance_strength": 1600 / 3600, "current_run": 0.70,
        },
    ]
    support_base = dict(
        top_k=2, max_total_gap=20, min_win_ratio=0.0, min_top3_ratio=0.48,
        edge_power=1.0, ev_power=1.0, top3_power=0.0, ability_power=0.0,
        min_edge_to_switch=0.0, min_ev_to_switch=0.0, switch_margin=0.0,
        min_relative_edge=0.0, min_tail_anchor_win_support=0.0,
        max_value_switch_win_ratio=9.0, max_value_switch_distance_m=9999.0,
        avoid_equal_total_value_switch=False, max_near_tie_today_deficit=0.0,
        avoid_total_recent_disagreement_switch=False,
        avoid_total_run_double_deficit_switch=False,
    )
    assert choose_main(
        support_expansion, [1, 2],
        D3Policy(**support_base, min_anchor_win_support=0.70),
    ) == 1
    assert choose_main(
        support_expansion, [1, 2],
        D3Policy(**support_base, min_anchor_win_support=0.66),
    ) == 2

    # D3.11 asymmetric boundary: the same 61.25%-support challenger may be promoted
    # at expected-popularity rank 7, but rank 8 must satisfy the stricter 80% tail floor.
    asymmetric = [dict(support_expansion[0]), dict(support_expansion[1])]
    asymmetric[1]["d3_win_prob"] = 0.245  # 70% of 0.35; above 61%, below 80%.
    asymmetric[1]["d3_top3_prob"] = 0.66
    asymmetric[1]["_expected_popularity"] = 7
    asym_base = dict(support_base)
    asym_base.update(min_win_ratio=0.55, min_top3_ratio=0.56,
                     min_anchor_win_support=0.61, min_tail_anchor_win_support=0.80)
    assert choose_main(asymmetric, [1, 2], D3Policy(**asym_base)) == 2
    asymmetric[1]["_expected_popularity"] = 8
    assert choose_main(asymmetric, [1, 2], D3Policy(**asym_base)) == 1

    # D3.14 robust-regime + anti-chase defaults: the selector stays short-memory but freezes the
    # meta-hyperparameters validated across multiple fixed-origin future blocks. It may
    # switch the compulsory 100-yen ticket only at a 5% shrunk advantage over D3.11.
    regime = D3RegimePolicy()
    assert regime.lookback_days == 6
    assert abs(regime.prior_races - 250.0) < 1e-12
    assert abs(regime.neutral_return_multiple - 0.80) < 1e-12
    assert abs(regime.return_cap_multiple - 6.0) < 1e-12
    assert abs(regime.switch_margin - 1.05) < 1e-12
    assert abs(regime.payout_ev_min_advantage_vs_ev - 1.02) < 1e-12
    assert regime.avoid_consecutive_payout_ev is True
    assert regime.actions == (REGIME_ACTION_POLICY, REGIME_ACTION_EV, REGIME_ACTION_PAYOUT_EV)

    action_rows = [
        {"horse_number": 1, "d3_ev": 0.90, "d3_payout_ev": 0.70, "d3_win_prob": 0.35, "_total": 90},
        {"horse_number": 2, "d3_ev": 1.40, "d3_payout_ev": 0.80, "d3_win_prob": 0.25, "_total": 86},
        {"horse_number": 3, "d3_ev": 0.80, "d3_payout_ev": 2.20, "d3_win_prob": 0.18, "_total": 84},
    ]
    assert choose_main_action(action_rows, [1, 2, 3], D3Policy(), REGIME_ACTION_EV) == 2
    assert choose_main_action(action_rows, [1, 2, 3], D3Policy(), REGIME_ACTION_PAYOUT_EV) == 3

    action, scores = select_regime_action([], regime)
    assert action == REGIME_ACTION_POLICY
    assert all(abs(v - 0.80) < 1e-12 for v in scores.values())

    # Sustained payout-EV strength clears the margin despite the 250-race neutral prior.
    history = [
        {REGIME_ACTION_POLICY: 0.8, REGIME_ACTION_EV: 0.6, REGIME_ACTION_PAYOUT_EV: 4.0}
        for _ in range(100)
    ]
    action, scores = select_regime_action(history, regime)
    assert action == REGIME_ACTION_PAYOUT_EV
    assert scores[REGIME_ACTION_PAYOUT_EV] > scores[REGIME_ACTION_POLICY] * regime.switch_margin

    # A historical jackpot is capped at 6x for regime detection, even though actual ROI
    # reporting elsewhere remains uncapped.
    _, jackpot_scores = select_regime_action([
        {REGIME_ACTION_POLICY: 0.0, REGIME_ACTION_EV: 0.0, REGIME_ACTION_PAYOUT_EV: 100.0}
    ], regime)
    expected_capped = (6.0 + 250.0 * 0.80) / 251.0
    assert abs(jackpot_scores[REGIME_ACTION_PAYOUT_EV] - expected_capped) < 1e-12


    # D3.14 payout-vs-EV tie-break: if payout-EV clears the policy switch margin but
    # is less than 2% above pure EV, choose pure EV when pure EV at least matches policy.
    near_tie_history = [
        {REGIME_ACTION_POLICY: 0.70, REGIME_ACTION_EV: 1.10, REGIME_ACTION_PAYOUT_EV: 1.115}
        for _ in range(300)
    ]
    near_action, near_scores = select_regime_action(near_tie_history, regime)
    assert near_scores[REGIME_ACTION_PAYOUT_EV] > near_scores[REGIME_ACTION_POLICY] * regime.switch_margin
    assert near_action == REGIME_ACTION_EV

    # D3.14 consecutive payout-EV breaker: callers may suppress a second payout-EV
    # race day.  Pure EV is used as the de-jackpot fallback when it is not below policy.
    repeat_history = [
        {REGIME_ACTION_POLICY: 0.70, REGIME_ACTION_EV: 0.90, REGIME_ACTION_PAYOUT_EV: 1.50}
        for _ in range(300)
    ]
    repeat_action, _ = select_regime_action(
        repeat_history, regime, allow_repeat_payout_ev=False
    )
    assert repeat_action == REGIME_ACTION_EV

    print("OK: single-win D3.14 synthetic contract tests passed")


if __name__ == "__main__":
    main()
