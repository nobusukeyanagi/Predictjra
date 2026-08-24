#!/usr/bin/env python3
"""Regression contract for v55 single-win/trifecta summaries and display labels."""
from __future__ import annotations

from pathlib import Path

from build_discord_notification import result_summary

ROOT = Path(__file__).resolve().parents[1]


def predicted(rid: str, *, win_return: int, tri_return: int, tri_stake: int = 3000) -> dict:
    return {
        "raceId": rid,
        "prediction": {"axes": [1, 2], "opponents": [3, 4, 5, 6, 7]},
        "result": {
            "places": [[1 if win_return else 8], [2], [3]],
            "winPayouts": [{"horses": [1 if win_return else 8], "payout": win_return or 220}],
            "trifectas": [{"horses": [1, 2, 3], "payout": 5000}],
        },
        "status": "hit" if tri_return else "miss",
        "winReturn": win_return,
        "winStake": 100,
        "payout": tri_return,
        "stake": tri_stake,
    }


def debut(rid: str) -> dict:
    return {
        "raceId": rid,
        "raceName": "2歳新馬",
        "prediction": None,
        "predictionDisabled": True,
        "predictionDisabledReason": "新馬戦",
        "result": {
            "places": [[9], [4], [5]],
            "winPayouts": [{"horses": [9], "payout": 280}],
            "trifectas": [{"horses": [9, 4, 5], "payout": 37030}],
        },
        "status": "result-only",
        "winReturn": 0,
        "winStake": 0,
        "payout": 0,
        "stake": 0,
    }


def test_summary_counts_any_ticket_hit_and_excludes_debut() -> None:
    day = {"races": [
        predicted("A", win_return=380, tri_return=0),
        predicted("B", win_return=0, tri_return=6000),
        debut("C"),
    ]}
    hits, total, win_recovery, tri_recovery = result_summary(day)
    assert (hits, total) == (2, 2)
    assert win_recovery == 190.0  # 380 / (2 * 100)
    assert tri_recovery == 100.0  # 6000 / (2 * 3000)


def test_frontend_contract() -> None:
    app = (ROOT / "app.js").read_text(encoding="utf-8")
    css = (ROOT / "styles.css").read_text(encoding="utf-8")
    for text in ("<th>単勝</th><th>三連単</th>", "単回収率", "三回収率", "予想無し", "総合成績"):
        assert text in app, text
    assert "単対" not in app and "本対" not in app
    for klass in ("result-role-main", "result-role-second", "result-role-danger", "payout-rate"):
        assert klass in css, klass


if __name__ == "__main__":
    tests = [test_summary_counts_any_ticket_hit_and_excludes_debut, test_frontend_contract]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"OK: {len(tests)} v55 return/display regression tests passed")
