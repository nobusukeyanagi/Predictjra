#!/usr/bin/env python3
"""Build a Discord webhook payload for a race-data publication."""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "races.json"
PAGE_URL = "https://nobusukeyanagi.github.io/Predictjra/"
WEEKDAYS = ["月", "火", "水", "木", "金", "土", "日"]
DEFAULT_STAKE = 3000


def load_day(target: str) -> dict:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    for day in data.get("days", []):
        if day.get("date") == target:
            return day
    raise SystemExit(f"Target day not found in {DATA_PATH}: {target}")


def label(target: str) -> str:
    d = date.fromisoformat(target)
    return f"{target}({WEEKDAYS[d.weekday()]})"


def is_debut_race(race: dict) -> bool:
    title = str(((race.get("modelMeta") or {}).get("indexDetail") or {}).get("title") or "")
    return "新馬" in title


def result_summary(day: dict) -> tuple[int, int, int, float]:
    races = [r for r in day.get("races", []) if not is_debut_race(r)]
    finished = [r for r in races if r.get("status") in {"hit", "miss"}]
    hits = sum(1 for r in finished if r.get("status") == "hit")
    payout = sum(int(r.get("payout") or 0) for r in finished)
    stake = sum(int(r.get("stake") or DEFAULT_STAKE) for r in finished)
    recovery = payout / stake * 100 if stake else 0.0
    return hits, len(races), payout, recovery


def build_message(mode: str, target: str, day: dict) -> str:
    d = label(target)
    if mode == "prepare":
        return f"🏇{d}のJRA予想を公開しました\n{PAGE_URL}"

    hits, total, payout, recovery = result_summary(day)
    return (
        f"🏇{d}のJRA予想結果\n"
        f"**的中数{hits} / {total}**\n"
        f"**払戻総額{payout:,}円**\n"
        f"**総回収率{recovery:.1f}%**\n"
        f"{PAGE_URL}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["prepare", "result"], required=True)
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    day = load_day(args.date)
    content = build_message(args.mode, args.date, day)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"send": True, "content": content}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
