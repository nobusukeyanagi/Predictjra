#!/usr/bin/env python3
"""Fetch JRA race cards/results with multi-source fallback and update data/races.json.

prepare: previous-day 13:00 JST run; targets tomorrow, creates predictions once.
result: race-day 18:00 JST run; targets today, records results and trifecta payout.

Source priority
---------------
Race discovery / entries:
  1. JBIS Search
  2. SportsNavi
  3. netkeiba

Results:
  1. JBIS Search
  2. SportsNavi
  3. netkeiba

JRA official remains the authoritative reference, but its public result URL contains
non-stable parameters and is therefore not used as an automatically generated fetch URL.
The scraper records the source actually used for each race in dataSources.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "races.json"
JST = ZoneInfo("Asia/Tokyo")

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
)

NETKEIBA_BASE = "https://race.netkeiba.com"
SPORTSNAVI_BASE = "https://sports.yahoo.co.jp/keiba/race"
JBIS_BASE = "https://www.jbis.or.jp/race"

TRACKS = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
}
JBIS_TRACK_CODES = {
    "01": "101", "02": "102", "03": "103", "04": "104", "05": "105",
    "06": "106", "07": "107", "08": "108", "09": "109", "10": "110",
}
TRACK_ORDER = {name: i for i, name in enumerate(TRACKS.values(), start=1)}

session = requests.Session()
session.headers.update({
    "User-Agent": UA,
    "Accept-Language": "ja,en-US;q=0.8,en;q=0.6",
})


def clean(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def request_html(url: str, *, pause: float = 0.35, referer: str | None = None) -> str:
    last = None
    headers = {"Referer": referer} if referer else {}
    for attempt in range(3):
        try:
            r = session.get(url, timeout=25, headers=headers)
            r.raise_for_status()
            if not r.encoding or r.encoding.lower() == "iso-8859-1":
                r.encoding = r.apparent_encoding
            html = r.text
            time.sleep(pause)
            return html
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.3 * (attempt + 1))
    raise RuntimeError(f"GET failed: {url}: {last}")


def selenium_html(url: str, wait_seconds: float = 2.5) -> str:
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


def sports_id(race_id: str) -> str:
    # netkeiba-style 202601010811 -> SportsNavi 2601010811
    return f"{int(race_id[:4]) % 100:02d}{race_id[4:]}"


def jbis_url(target: date, race_id: str, *, result: bool = False) -> str:
    track = JBIS_TRACK_CODES[race_id[4:6]]
    race_no = int(race_id[-2:])
    prefix = f"{JBIS_BASE}/result" if result else JBIS_BASE
    return f"{prefix}/{target:%Y%m%d}/{track}/{race_no:02d}/"


def sports_url(race_id: str, *, result: bool = False) -> str:
    page = "result" if result else "denma"
    return f"{SPORTSNAVI_BASE}/{page}/{sports_id(race_id)}"


def netkeiba_url(race_id: str, *, result: bool = False) -> str:
    page = "result.html" if result else "shutuba_past.html"
    return f"{NETKEIBA_BASE}/race/{page}?race_id={race_id}"


def build_race_id(target: date, track_code: str, meeting: int, day_no: int, race_no: int) -> str:
    return f"{target.year:04d}{track_code}{meeting:02d}{day_no:02d}{race_no:02d}"


# ---------------------------------------------------------------------------
# Race discovery
# ---------------------------------------------------------------------------

def discover_race_ids_jbis(target: date) -> list[str]:
    """Discover active JRA tracks via JBIS and construct netkeiba-style race IDs.

    JBIS race URLs are date/track/race based. The page title contains e.g.
    "1回 札幌 8日", allowing us to reconstruct the conventional race ID
    without asking netkeiba for the race list.
    """
    found: list[str] = []
    for track_code, venue in TRACKS.items():
        jbis_track = JBIS_TRACK_CODES[track_code]
        url = f"{JBIS_BASE}/{target:%Y%m%d}/{jbis_track}/01/"
        try:
            html = request_html(url, pause=0.15, referer="https://www.jbis.or.jp/")
        except Exception:
            continue

        soup = BeautifulSoup(html, "lxml")
        title_text = " ".join([
            soup.title.get_text(" ", strip=True) if soup.title else "",
            soup.get_text(" ", strip=True)[:2000],
        ])
        m = re.search(rf"(\d+)回\s*{re.escape(venue)}\s*(\d+)日", title_text)
        if not m:
            continue

        meeting = int(m.group(1))
        day_no = int(m.group(2))
        # Central JRA cards have at most 12 races; individual cancelled races can
        # later fail safely at the entry-fetch stage.
        for race_no in range(1, 13):
            found.append(build_race_id(target, track_code, meeting, day_no, race_no))

    return sorted(set(found), key=lambda rid: (int(rid[4:6]), int(rid[-2:])))


def discover_race_ids_netkeiba(target: date) -> list[str]:
    ds = target.strftime("%Y%m%d")
    url = f"{NETKEIBA_BASE}/top/race_list.html?kaisai_date={ds}"
    try:
        html = request_html(url, pause=0.3, referer="https://race.netkeiba.com/")
        ids = re.findall(r"race_id=(\d{12})", html)
    except Exception:
        ids = []
    if not ids:
        try:
            html = selenium_html(url, wait_seconds=3.0)
            ids = re.findall(r"race_id=(\d{12})", html)
        except Exception:
            ids = []

    unique = []
    seen = set()
    for rid in ids:
        if rid in seen or rid[:4] != str(target.year) or rid[4:6] not in TRACKS:
            continue
        race_no = int(rid[-2:])
        if 1 <= race_no <= 12:
            seen.add(rid)
            unique.append(rid)
    return sorted(unique, key=lambda rid: (int(rid[4:6]), int(rid[-2:])))


def discover_race_ids(target: date) -> tuple[list[str], str]:
    ids = discover_race_ids_jbis(target)
    if ids:
        print(f"DISCOVERY source=JBIS races={len(ids)}")
        return ids, "jbis"

    ids = discover_race_ids_netkeiba(target)
    if ids:
        print(f"DISCOVERY source=netkeiba races={len(ids)}")
        return ids, "netkeiba"

    return [], "none"


# ---------------------------------------------------------------------------
# Entry parsers
# ---------------------------------------------------------------------------

def dedupe_entries(entries: list[dict]) -> list[dict]:
    by_horse: dict[int, dict] = {}
    for entry in entries:
        horse = int(entry["horse"])
        if 1 <= horse <= 18:
            by_horse[horse] = {
                "horse": horse,
                "frame": entry.get("frame"),
                **({"name": entry["name"]} if entry.get("name") else {}),
            }
    return [by_horse[h] for h in sorted(by_horse)]


def parse_entries_jbis(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    entries: list[dict] = []

    for row in soup.find_all("tr"):
        cells = row.find_all(["th", "td"], recursive=False)
        if not cells:
            continue
        texts = [clean(c.get_text(" ", strip=True)) for c in cells]

        horse_idx = None
        horse = None
        for i, text in enumerate(texts):
            m = re.fullmatch(r"(\d{1,2})番", text)
            if m and 1 <= int(m.group(1)) <= 18:
                horse_idx = i
                horse = int(m.group(1))
                break
        if horse is None:
            continue

        frame = None
        for text in reversed(texts[:horse_idx]):
            if re.fullmatch(r"[1-8]", text):
                frame = int(text)
                break

        name = None
        if horse_idx + 1 < len(texts):
            candidate = re.sub(r"ブラックタイプ.*$", "", texts[horse_idx + 1])
            if candidate and not re.fullmatch(r"\d+", candidate):
                name = candidate[:40]

        entries.append({"horse": horse, "frame": frame, "name": name})

    return dedupe_entries(entries)


def parse_entries_sportsnavi(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    entries: list[dict] = []

    # Prefer tables whose header explicitly contains 馬番.
    tables = [t for t in soup.find_all("table") if "馬番" in clean(t.get_text(" ", strip=True))]
    for table in tables or soup.find_all("table"):
        for row in table.find_all("tr"):
            cells = row.find_all(["th", "td"], recursive=False)
            if len(cells) < 2:
                continue
            texts = [clean(c.get_text(" ", strip=True)) for c in cells]

            # SportsNavi usually renders 枠番, 馬番 as the first two numeric cells.
            numeric = [(i, int(t)) for i, t in enumerate(texts) if re.fullmatch(r"\d{1,2}", t)]
            numeric = [(i, n) for i, n in numeric if 1 <= n <= 18]
            if not numeric:
                continue

            horse_idx, horse = numeric[1] if len(numeric) >= 2 and numeric[0][1] <= 8 else numeric[0]
            frame = numeric[0][1] if len(numeric) >= 2 and numeric[0][1] <= 8 else None

            # Avoid result/odds helper rows: an entry row should contain non-numeric text after horse number.
            name = None
            for text in texts[horse_idx + 1:]:
                if text and not re.fullmatch(r"[\d.()+\-]+", text):
                    name = text[:40]
                    break
            if not name:
                continue

            entries.append({"horse": horse, "frame": frame, "name": name})

        if len(dedupe_entries(entries)) >= 2:
            break

    return dedupe_entries(entries)


def parse_entries_netkeiba(html: str) -> list[dict]:
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

        name_node = row.select_one(".HorseName a") or row.select_one("td.HorseInfo a")
        name = clean(name_node.get_text(" ", strip=True)) if name_node else None
        entries.append({"horse": horse, "frame": frame, "name": name})

    return dedupe_entries(entries)


def fetch_entries(race_id: str, target: date) -> tuple[list[dict], str]:
    attempts = [
        ("jbis", jbis_url(target, race_id), parse_entries_jbis, "https://www.jbis.or.jp/"),
        ("sportsnavi", sports_url(race_id), parse_entries_sportsnavi, "https://sports.yahoo.co.jp/keiba/"),
        ("netkeiba", netkeiba_url(race_id), parse_entries_netkeiba, "https://race.netkeiba.com/"),
    ]

    errors: list[str] = []
    for source, url, parser, referer in attempts:
        try:
            html = request_html(url, referer=referer)
            entries = parser(html)
            if len(entries) >= 2:
                print(f"ENTRIES {race_id} source={source} horses={len(entries)}")
                return entries, source
            errors.append(f"{source}: parsed {len(entries)}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{source}: {exc}")

    # Last browser retry for netkeiba only.
    try:
        html = selenium_html(netkeiba_url(race_id), wait_seconds=2.0)
        entries = parse_entries_netkeiba(html)
        if len(entries) >= 2:
            print(f"ENTRIES {race_id} source=netkeiba-selenium horses={len(entries)}")
            return entries, "netkeiba-selenium"
    except Exception as exc:  # noqa: BLE001
        errors.append(f"netkeiba-selenium: {exc}")

    raise RuntimeError("entry fetch failed; " + " | ".join(errors))


# ---------------------------------------------------------------------------
# Result parsers
# ---------------------------------------------------------------------------

def table_header_indices(table) -> tuple[int | None, int | None]:
    for row in table.find_all("tr"):
        cells = row.find_all(["th", "td"], recursive=False)
        if not cells:
            continue
        headers = [clean(c.get_text(" ", strip=True)) for c in cells]
        place_i = next((i for i, h in enumerate(headers) if "着順" in h), None)
        horse_i = next((i for i, h in enumerate(headers) if "馬番" in h), None)
        if place_i is not None and horse_i is not None:
            return place_i, horse_i
    return None, None


def parse_result_places_generic(soup: BeautifulSoup) -> list[list[int]]:
    place_map: dict[int, list[int]] = {}

    for table in soup.find_all("table"):
        place_i, horse_i = table_header_indices(table)
        if place_i is None or horse_i is None:
            continue

        for row in table.find_all("tr"):
            cells = row.find_all(["th", "td"], recursive=False)
            if len(cells) <= max(place_i, horse_i):
                continue

            place_text = clean(cells[place_i].get_text(" ", strip=True))
            horse_text = clean(cells[horse_i].get_text(" ", strip=True))
            pm = re.match(r"^(\d+)", place_text)
            hm = re.search(r"(\d{1,2})", horse_text)
            if not pm or not hm:
                continue

            place = int(pm.group(1))
            horse = int(hm.group(1))
            if 1 <= place <= 3 and 1 <= horse <= 18:
                place_map.setdefault(place, []).append(horse)

        if place_map:
            break

    return [place_map[p] for p in sorted(place_map)]


def parse_trifectas_generic(soup: BeautifulSoup) -> list[dict]:
    combos: list[dict] = []
    seen = set()

    for row in soup.find_all("tr"):
        text = row.get_text(" ", strip=True)
        if "3連単" not in text and "三連単" not in text:
            continue

        normalized = re.sub(r"\s+", " ", text)
        horses = re.findall(
            r"(?<!\d)(\d{1,2})\s*(?:-|－|→|＞|>)\s*(\d{1,2})\s*(?:-|－|→|＞|>)\s*(\d{1,2})(?!\d)",
            normalized,
        )
        payouts = [int(x.replace(",", "")) for x in re.findall(r"([\d,]+)\s*円", normalized)]

        if horses and payouts:
            for h, payout in zip(horses, payouts):
                nums = [int(x) for x in h]
                key = (*nums, payout)
                if key not in seen:
                    seen.add(key)
                    combos.append({"horses": nums, "payout": payout})

    return combos


def parse_result_generic(html: str) -> dict | None:
    soup = BeautifulSoup(html, "lxml")
    places = parse_result_places_generic(soup)
    trifectas = parse_trifectas_generic(soup)
    if not places or not trifectas:
        return None
    return {"places": places, "trifectas": trifectas}


def parse_result_netkeiba(html: str) -> dict | None:
    # Generic parser handles most current netkeiba markup. Keep a fallback for its
    # Result/Payout classes in case header markup changes.
    generic = parse_result_generic(html)
    if generic:
        return generic

    soup = BeautifulSoup(html, "lxml")
    table = soup.select_one("table.RaceTable01") or soup.select_one(".ResultTableWrap table")
    if not table:
        return None

    place_map: dict[int, list[int]] = {}
    for row in table.select("tr"):
        cells = row.find_all(["th", "td"], recursive=False)
        if len(cells) < 3:
            continue
        pm = re.match(r"^(\d+)", clean(cells[0].get_text(" ", strip=True)))
        if not pm:
            continue
        place = int(pm.group(1))
        if place > 3:
            continue

        horse_node = row.select_one("td.Num") or row.select_one(".Num")
        horse_text = clean(horse_node.get_text(" ", strip=True)) if horse_node else clean(cells[2].get_text(" ", strip=True))
        hm = re.search(r"\b(\d{1,2})\b", horse_text)
        if hm:
            place_map.setdefault(place, []).append(int(hm.group(1)))

    places = [place_map[p] for p in sorted(place_map)]
    trifectas = parse_trifectas_generic(soup)
    return {"places": places, "trifectas": trifectas} if places and trifectas else None


def fetch_result(race_id: str, target: date) -> tuple[dict | None, str]:
    attempts = [
        ("jbis", jbis_url(target, race_id, result=True), parse_result_generic, "https://www.jbis.or.jp/"),
        ("sportsnavi", sports_url(race_id, result=True), parse_result_generic, "https://sports.yahoo.co.jp/keiba/"),
        ("netkeiba", netkeiba_url(race_id, result=True), parse_result_netkeiba, "https://race.netkeiba.com/"),
    ]

    errors: list[str] = []
    for source, url, parser, referer in attempts:
        try:
            html = request_html(url, referer=referer)
            result = parser(html)
            if result:
                print(f"RESULT_FETCH {race_id} source={source}")
                return result, source
            errors.append(f"{source}: result not complete")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{source}: {exc}")

    try:
        html = selenium_html(netkeiba_url(race_id, result=True), wait_seconds=2.0)
        result = parse_result_netkeiba(html)
        if result:
            print(f"RESULT_FETCH {race_id} source=netkeiba-selenium")
            return result, "netkeiba-selenium"
    except Exception as exc:  # noqa: BLE001
        errors.append(f"netkeiba-selenium: {exc}")

    print(f"RESULT_FETCH_FAILED {race_id}: {' | '.join(errors)}")
    return None, "none"


# ---------------------------------------------------------------------------
# Data / prediction
# ---------------------------------------------------------------------------

def default_data() -> dict:
    return {
        "updatedAt": datetime.now(JST).isoformat(timespec="seconds"),
        "bet": {
            "type": "3連単2頭軸マルチ",
            "axes": 2,
            "selectionRule": "min(ceil(horseCount/2), 7)",
            "maxOpponents": 5,
            "unitYen": 100,
            "maxCombinations": 30,
            "maxStakePerRace": 3000,
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


def prediction_target_count(horse_count: int) -> int:
    if horse_count < 5:
        raise ValueError(f"Need at least 5 starters, got {horse_count}")
    return min((horse_count + 1) // 2, 7)


def create_prediction(horses: list[int]) -> dict:
    # Existing fallback. The index-based prediction engine can replace this when
    # race index data is generated server-side for every race.
    pick_count = prediction_target_count(len(horses))
    picks = random.SystemRandom().sample(horses, pick_count)
    return {"axes": picks[:2], "opponents": picks[2:]}


def stake_for_prediction(prediction: dict, unit_yen: int = 100) -> int:
    return len(prediction.get("opponents", [])) * 6 * unit_yen


def combo_is_covered(prediction: dict, combo: Iterable[int]) -> bool:
    combo_set = set(combo)
    axes = set(prediction.get("axes", []))
    opponents = set(prediction.get("opponents", []))
    return axes.issubset(combo_set) and bool(opponents & combo_set)


def prepare_day(data: dict, target: date) -> int:
    ids, discovery_source = discover_race_ids(target)
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
            entries, entry_source = fetch_entries(race_id, target)
            horses = [e["horse"] for e in entries]
            if len(horses) < 5:
                print(f"SKIP {race_id}: only {len(horses)} horse numbers parsed")
                continue

            frames = {str(e["horse"]): e["frame"] for e in entries if e.get("frame")}
            names = {str(e["horse"]): e["name"] for e in entries if e.get("name")}

            race = existing.get(race_id, {})
            prediction = race.get("prediction") or create_prediction(horses)

            race.update({
                "raceId": race_id,
                "venue": venue,
                "raceNo": race_no,
                "horseCount": len(horses),
                "horseFrames": frames,
                "horseNames": names,
                "prediction": prediction,
                "result": race.get("result"),
                "status": race.get("status", "pending"),
                "payout": int(race.get("payout", 0)),
                "trifectaPayouts": race.get("trifectaPayouts", []),
                "stake": stake_for_prediction(prediction),
                "dataSources": {
                    **race.get("dataSources", {}),
                    "discovery": discovery_source,
                    "entries": entry_source,
                },
            })

            if race_id not in existing:
                day["races"].append(race)
                existing[race_id] = race

            changed += 1
            print(f"PREPARED {target} {venue}{race_no}R {race_id} source={entry_source}")
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
            result, result_source = fetch_result(race_id, target)
            if not result:
                print(f"PENDING {race_id}: result/trifecta not parsed yet")
                continue

            prediction = race.get("prediction") or {"axes": [], "opponents": []}
            winning = [t for t in result["trifectas"] if combo_is_covered(prediction, t["horses"])]
            payout = sum(int(t["payout"]) for t in winning)

            race["result"] = result
            race["status"] = "hit" if winning else "miss"
            race["payout"] = payout
            race["trifectaPayouts"] = [int(t["payout"]) for t in result["trifectas"]]
            race["stake"] = stake_for_prediction(prediction)
            race["dataSources"] = {
                **race.get("dataSources", {}),
                "result": result_source,
            }

            changed += 1
            print(f"RESULT {race_id}: {race['status']} payout={payout} source={result_source}")
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
