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
    if race.get("predictionDisabledReason") == "新馬戦" or race.get("predictionDisabled") is True:
        return True
    title = str(((race.get("modelMeta") or {}).get("indexDetail") or {}).get("title") or "")
    race_name = str(race.get("raceName") or "")
    return "新馬" in title or "新馬" in race_name


def result_summary(day: dict) -> tuple[int, int, float, float]:
    races = [
        r for r in day.get("races", [])
        if not is_debut_race(r) and r.get("prediction")
    ]
    finished = [r for r in races if r.get("status") in {"hit", "miss"} and r.get("result")]
    hits = sum(
        1 for r in finished
        if int(r.get("winReturn") or 0) > 0 or int(r.get("payout") or 0) > 0
    )
    win_return = sum(int(r.get("winReturn") or 0) for r in finished)
    win_stake = len(finished) * 100
    tri_return = sum(int(r.get("payout") or 0) for r in finished)
    tri_stake = sum(int(r.get("stake") or DEFAULT_STAKE) for r in finished)
    win_recovery = win_return / win_stake * 100 if win_stake else 0.0
    tri_recovery = tri_return / tri_stake * 100 if tri_stake else 0.0
    return hits, len(races), win_recovery, tri_recovery

def build_message(mode: str, target: str, day: dict) -> str:
    d = label(target)
    if mode == "prepare":
        return f"🏇{d}のJRA予想を公開しました\n{PAGE_URL}"

    hits, total, win_recovery, tri_recovery = result_summary(day)
    return (
        f"🏇{d}のJRA予想結果\n"
        f"**的中数{hits} / {total}**\n"
        f"**単回収率{win_recovery:.1f}%**\n"
        f"**三回収率{tri_recovery:.1f}%**\n"
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
