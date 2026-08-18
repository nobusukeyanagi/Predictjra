#!/usr/bin/env python3
"""Fetch JRA race cards/results from netkeiba and update data/races.json.

prepare: previous-day 15:00 JST run; targets tomorrow, creates predictions once.
result: race-day 18:00 JST run; targets today, records results, official trifecta payout, and prediction return.

The scraper deliberately uses conservative delays and has several selector fallbacks,
because netkeiba markup can change over time.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "races.json"
JST = ZoneInfo("Asia/Tokyo")
BASE = "https://race.netkeiba.com"
UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
)
TRACKS = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
}
TRACK_ORDER = {name: i for i, name in enumerate(TRACKS.values(), start=1)}

session = requests.Session()
session.headers.update({
    "User-Agent": UA,
    "Accept-Language": "ja,en-US;q=0.8,en;q=0.6",
    "Referer": "https://race.netkeiba.com/",
})


def clean(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def int_money(text: str) -> int | None:
    m = re.search(r"([\d,]+)円", text or "")
    return int(m.group(1).replace(",", "")) if m else None


def request_html(url: str, *, pause: float = 0.7) -> str:
    last = None
    for attempt in range(3):
        try:
            r = session.get(url, timeout=25)
            r.raise_for_status()
            if not r.encoding or r.encoding.lower() == "iso-8859-1":
                r.encoding = r.apparent_encoding
            time.sleep(pause)
            return r.text
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"GET failed: {url}: {last}")


def selenium_html(url: str, wait_seconds: float = 3.0) -> str:
    # Ubuntu GitHub-hosted runners include Chrome; Selenium Manager resolves driver.
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument(f"--user-agent={UA}")
    driver = webdriver.Chrome(options=opts)
    try:
        driver.set_page_load_timeout(35)
        driver.get(url)
        time.sleep(wait_seconds)
        return driver.page_source
    finally:
        driver.quit()


def discover_race_ids(target: date) -> list[str]:
    ds = target.strftime("%Y%m%d")
    url = f"{BASE}/top/race_list.html?kaisai_date={ds}"
    # Try lightweight HTML first. The current race-list page may be JS-rendered,
    # so fall back to a browser if no race links are present.
    try:
        html = request_html(url, pause=0.4)
        ids = re.findall(r"race_id=(\d{12})", html)
    except Exception:
        ids = []
    if not ids:
        html = selenium_html(url, wait_seconds=4.0)
        ids = re.findall(r"race_id=(\d{12})", html)

    # Dedupe and keep only central JRA tracks/year. Race number must be 01..12.
    unique = []
    seen = set()
    for rid in ids:
        if rid in seen or rid[:4] != str(target.year) or rid[4:6] not in TRACKS:
            continue
        try:
            race_no = int(rid[-2:])
        except ValueError:
            continue
        if 1 <= race_no <= 12:
            seen.add(rid)
            unique.append(rid)
    unique.sort(key=lambda rid: (int(rid[4:6]), int(rid[-2:])))
    return unique


def parse_entries(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    entries: list[dict] = []

    rows = soup.select("tr.HorseList") or soup.select("table.Shutuba_Table tr")
    for row in rows:
        horse_node = (
            row.select_one("td.Umaban")
            or row.select_one(".Umaban")
            or row.select_one("td[class*='Umaban']")
        )
        if not horse_node:
            continue
        hm = re.search(r"\b(\d{1,2})\b", clean(horse_node.get_text(" ", strip=True)))
        if not hm:
            continue
        horse = int(hm.group(1))
        if not 1 <= horse <= 18:
            continue

        frame = None
        frame_node = row.select_one("td.Waku") or row.select_one(".Waku") or row.select_one("td[class*='Waku']")
        if frame_node:
            fm = re.search(r"\b([1-8])\b", clean(frame_node.get_text(" ", strip=True)))
            if fm:
                frame = int(fm.group(1))
            if frame is None:
                class_text = " ".join(frame_node.get("class", []))
                fm = re.search(r"Waku[_-]?([1-8])", class_text, re.I)
                if fm:
                    frame = int(fm.group(1))
        entries.append({"horse": horse, "frame": frame})

    if not entries:
        # Fallback for markup variants. Frame may be unavailable, but horse number is enough to predict.
        for node in soup.select(".HorseList td.Num, .HorseList .Num"):
            t = clean(node.get_text(" ", strip=True))
            if t.isdigit() and 1 <= int(t) <= 18:
                entries.append({"horse": int(t), "frame": None})

    by_horse = {}
    for e in entries:
        by_horse[e["horse"]] = e
    return [by_horse[h] for h in sorted(by_horse)]


def fetch_entries(race_id: str) -> list[dict]:
    url = f"{BASE}/race/shutuba_past.html?race_id={race_id}"
    html = request_html(url)
    entries = parse_entries(html)
    if len(entries) < 2:
        entries = parse_entries(selenium_html(url, wait_seconds=2.0))
    return entries

def parse_result_places(soup: BeautifulSoup) -> list[list[int]]:
    table = soup.select_one("table.RaceTable01") or soup.select_one(".ResultTableWrap table")
    if not table:
        return []

    place_map: dict[int, list[int]] = {}
    for row in table.select("tr"):
        cells = row.find_all(["th", "td"], recursive=False)
        if len(cells) < 3:
            continue
        first = clean(cells[0].get_text(" ", strip=True))
        pm = re.match(r"^(\d+)", first)
        if not pm:
            continue
        place = int(pm.group(1))
        if place > 3:
            continue

        horse_node = row.select_one("td.Num") or row.select_one(".Num")
        candidates = []
        if horse_node:
            candidates = re.findall(r"\b(\d{1,2})\b", clean(horse_node.get_text(" ", strip=True)))
        if not candidates:
            # In netkeiba's result table, horse number is normally the 3rd direct cell.
            candidates = re.findall(r"\b(\d{1,2})\b", clean(cells[2].get_text(" ", strip=True)))
        if candidates:
            h = int(candidates[0])
            if 1 <= h <= 18:
                place_map.setdefault(place, []).append(h)

    return [place_map[p] for p in sorted(place_map) if p <= 3]


def parse_trifectas(soup: BeautifulSoup) -> list[dict]:
    rows = soup.select("table.Payout_Detail_Table tr")
    if not rows:
        rows = soup.find_all("tr")

    for row in rows:
        label = row.find("th") or row.find("td")
        if not label or "3連単" not in clean(label.get_text(" ", strip=True)):
            continue

        result_cell = row.select_one("td.Result")
        payout_cell = row.select_one("td.Payout")
        cells = row.find_all("td", recursive=False)
        if result_cell is None and cells:
            result_cell = cells[0]
        if payout_cell is None and len(cells) >= 2:
            payout_cell = cells[1]
        if result_cell is None or payout_cell is None:
            continue

        nums = [int(x) for x in re.findall(r"\d{1,2}", result_cell.get_text(" ", strip=True))]
        payouts = [int(x.replace(",", "")) for x in re.findall(r"([\d,]+)円", payout_cell.get_text(" ", strip=True))]
        if not payouts:
            # Some markup separates the yen symbol from the numeric span.
            payouts = [int(x.replace(",", "")) for x in re.findall(r"\b([\d,]{3,})\b", payout_cell.get_text(" ", strip=True))]

        combos = []
        for i, p in enumerate(payouts):
            chunk = nums[i * 3:(i + 1) * 3]
            if len(chunk) == 3:
                combos.append({"horses": chunk, "payout": p})
        if combos:
            return combos
    return []


def fetch_result(race_id: str) -> dict | None:
    url = f"{BASE}/race/result.html?race_id={race_id}"
    html = request_html(url)
    soup = BeautifulSoup(html, "lxml")
    places = parse_result_places(soup)
    trifectas = parse_trifectas(soup)
    if not places or not trifectas:
        html = selenium_html(url, wait_seconds=2.0)
        soup = BeautifulSoup(html, "lxml")
        places = parse_result_places(soup)
        trifectas = parse_trifectas(soup)
    if not places or not trifectas:
        return None
    return {"places": places, "trifectas": trifectas}


def default_data() -> dict:
    return {
        "updatedAt": datetime.now(JST).isoformat(timespec="seconds"),
        "bet": {
            "type": "3連単2頭軸マルチ", "axes": 2, "opponents": 5,
            "unitYen": 100, "combinations": 30, "stakePerRace": 3000,
        },
        "days": [],
    }


def load_data() -> dict:
    if not DATA_PATH.exists():
        return default_data()
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def save_data(data: dict) -> None:
    data["updatedAt"] = datetime.now(JST).isoformat(timespec="seconds")
    data["days"] = sorted(data.get("days", []), key=lambda d: d["date"], reverse=True)
    for day in data["days"]:
        day["races"] = sorted(
            day.get("races", []),
            key=lambda r: (TRACK_ORDER.get(r.get("venue", ""), 99), int(r.get("raceNo", 99))),
        )
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def find_day(data: dict, target: date, create: bool = False) -> dict | None:
    iso = target.isoformat()
    for day in data.get("days", []):
        if day.get("date") == iso:
            return day
    if create:
        day = {"date": iso, "races": []}
        data.setdefault("days", []).append(day)
        return day
    return None


def race_meta(race_id: str) -> tuple[str, int]:
    return TRACKS.get(race_id[4:6], race_id[4:6]), int(race_id[-2:])


def create_prediction(horses: list[int]) -> dict:
    if len(horses) < 7:
        raise ValueError(f"Need at least 7 starters, got {len(horses)}")
    picks = random.SystemRandom().sample(horses, 7)
    return {"axes": picks[:2], "opponents": picks[2:]}


def combo_is_covered(prediction: dict, combo: Iterable[int]) -> bool:
    combo_set = set(combo)
    axes = set(prediction.get("axes", []))
    opponents = set(prediction.get("opponents", []))
    return axes.issubset(combo_set) and bool(opponents & combo_set)


def prepare_day(data: dict, target: date) -> int:
    ids = discover_race_ids(target)
    if not ids:
        print(f"No JRA race IDs found for {target}; no changes.")
        return 0

    day = find_day(data, target, create=True)
    existing = {r["raceId"]: r for r in day["races"]}
    changed = 0
    for race_id in ids:
        if race_id in existing and existing[race_id].get("prediction"):
            continue
        venue, race_no = race_meta(race_id)
        try:
            entries = fetch_entries(race_id)
            horses = [e["horse"] for e in entries]
            if len(horses) < 7:
                print(f"SKIP {race_id}: only {len(horses)} horse numbers parsed")
                continue
            frames = {str(e["horse"]): e["frame"] for e in entries if e.get("frame")}
            race = existing.get(race_id, {})
            race.update({
                "raceId": race_id,
                "venue": venue,
                "raceNo": race_no,
                "horseCount": len(horses),
                "horseFrames": frames,
                "prediction": race.get("prediction") or create_prediction(horses),
                "result": race.get("result"),
                "status": race.get("status", "pending"),
                # payout = return from this prediction (0 when missed).
                "payout": int(race.get("payout", 0)),
                # trifectaPayouts = official 3連単 payout(s), regardless of hit/miss.
                "trifectaPayouts": race.get("trifectaPayouts", []),
                "stake": 3000,
            })
            if race_id not in existing:
                day["races"].append(race)
                existing[race_id] = race
            changed += 1
            print(f"PREPARED {target} {venue}{race_no}R {race_id}")
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR prepare {race_id}: {exc}")
    return changed


def result_day(data: dict, target: date) -> int:
    day = find_day(data, target, create=False)
    if not day or not day.get("races"):
        print(f"No prepared races for {target}; attempting preparation first.")
        prepare_day(data, target)
        day = find_day(data, target, create=False)
    if not day:
        return 0

    changed = 0
    for race in day.get("races", []):
        race_id = race["raceId"]
        try:
            result = fetch_result(race_id)
            if not result:
                print(f"PENDING {race_id}: result/trifecta not parsed yet")
                continue
            prediction = race.get("prediction") or {"axes": [], "opponents": []}
            winning = [t for t in result["trifectas"] if combo_is_covered(prediction, t["horses"])]
            payout = sum(int(t["payout"]) for t in winning)
            race["result"] = result
            race["status"] = "hit" if winning else "miss"
            # Keep the prediction return separate from the official race payout.
            # This lets the page show the real 3連単 payout even when the prediction missed.
            race["payout"] = payout
            race["trifectaPayouts"] = [int(t["payout"]) for t in result["trifectas"]]
            race["stake"] = 3000
            changed += 1
            print(f"RESULT {race_id}: {race['status']} payout={payout}")
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR result {race_id}: {exc}")
    return changed


def resolve_target(mode: str, explicit: str | None) -> date:
    if explicit:
        return date.fromisoformat(explicit)
    today = datetime.now(JST).date()
    return today + timedelta(days=1) if mode == "prepare" else today


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["prepare", "result"], required=True)
    parser.add_argument("--date", help="YYYY-MM-DD. Omit for scheduled default.")
    args = parser.parse_args()

    target = resolve_target(args.mode, args.date)
    data = load_data()
    print(f"mode={args.mode} target={target}")
    changed = prepare_day(data, target) if args.mode == "prepare" else result_day(data, target)
    if changed:
        save_data(data)
        print(f"Saved {DATA_PATH} ({changed} race updates)")
    else:
        print("No data changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
