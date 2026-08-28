#!/usr/bin/env python3
from __future__ import annotations

from datetime import date, timedelta

from single_win_d3 import ABILITY_FEATURE_COLS, D3Model, D3Policy, choose_main


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

    print("OK: single-win D3 synthetic contract tests passed")


if __name__ == "__main__":
    main()
