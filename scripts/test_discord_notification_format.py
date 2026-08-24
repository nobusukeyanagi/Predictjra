#!/usr/bin/env python3
from __future__ import annotations

from build_discord_notification import PAGE_URL, build_message


def synthetic_day() -> dict:
    # 36 predicted races, one ticket hit. Returns are chosen to produce 91.6%.
    races = []
    for i in range(36):
        races.append({
            "prediction": {"axes": [1, 2], "opponents": [3]},
            "status": "hit" if i == 0 else "miss",
            "result": {"finish": [1, 2, 3]},
            "winReturn": 3298 if i == 0 else 0,
            "payout": 98928 if i == 0 else 0,
            "stake": 3000,
        })
    return {"date": "2026-08-23", "races": races}


def main() -> None:
    day = synthetic_day()
    expected_result = (
        "🏇2026-08-23(日)のJRA予想結果\n"
        "**的中数 1 / 36**\n"
        "**単回収率 91.6%**\n"
        "**三回収率 91.6%**\n"
        f"{PAGE_URL}"
    )
    assert build_message("result", "2026-08-23", day) == expected_result

    expected_prepare = (
        "🏇2026-08-23(日)のJRA予想を公開しました\n"
        f"{PAGE_URL}"
    )
    assert build_message("prepare", "2026-08-23", day) == expected_prepare
    print("Discord notification format OK")


if __name__ == "__main__":
    main()
