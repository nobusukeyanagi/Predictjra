#!/usr/bin/env python3
"""Build a durable leakage-safe historical facts cache for Predictjra.

Target races are discovered from canonical 12-digit archived race-card files rather than
from the old prediction-output directory.  This matters for dates such as 2026-07-25 and
2026-07-26 where all JRA race cards/results exist but only some legacy prediction CSVs were
saved.

For every completed target race the cache stores:
- a sanitized race card (current odds / actual popularity / bodyweight removed),
- a runner-set snapshot under data/predictions/ (sanitized legacy snapshot when usable,
  otherwise synthesized from the sanitized card),
- final result and trifecta payout,
- historical result files needed for the target horses' pre-race histories.

The runner-set snapshot never carries old model scores/ranks.  Predictjra indices and
selections are always recalculated by the current candidate logic during Rebuild.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from bs4 import BeautifulSoup

JST = ZoneInfo("Asia/Tokyo")
CACHE_VERSION = "predictjra-historical-facts-v5-web-repair"
SOURCE_REPO = "sugaimo15/keibayosoku"
SOURCE_REF = "claude/horse-racing-predictor-ak6crm"

PROHIBITED_CURRENT_COLUMNS = {"win_odds", "horse_weight", "popularity"}
LEGACY_DERIVED_COLUMNS = {"score", "predicted_rank", "ml_win_prob", "ml_ev", "ml_rank"}
CANONICAL_RACE_FILE = re.compile(r"\d{12}\.csv")


def clean_str(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def source_commit(source_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def prediction_columns_to_strip(pred: pd.DataFrame) -> list[str]:
    return sorted(
        c for c in pred.columns
        if c in PROHIBITED_CURRENT_COLUMNS
        or c in LEGACY_DERIVED_COLUMNS
        or c.startswith("ml_")
    )


def sanitize_prediction_snapshot(pred: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    removed = prediction_columns_to_strip(pred)
    return pred.drop(columns=removed, errors="ignore").copy(), removed


def sanitize_card(card: pd.DataFrame) -> pd.DataFrame:
    return card.drop(columns=sorted(PROHIBITED_CURRENT_COLUMNS), errors="ignore").copy()


def runner_numbers(df: pd.DataFrame) -> set[int]:
    if "horse_number" not in df.columns:
        return set()
    vals = pd.to_numeric(df["horse_number"], errors="coerce").dropna().astype(int)
    return set(vals.tolist())


def valid_race_frame(df: pd.DataFrame, race_id: str) -> tuple[bool, str]:
    missing = {"race_id", "horse_number"} - set(df.columns)
    if missing:
        return False, "missing core columns: " + ", ".join(sorted(missing))
    if df.empty:
        return False, "empty file"
    race_ids = {clean_str(x) for x in df["race_id"].tolist() if clean_str(x)}
    if race_ids != {race_id}:
        return False, f"invalid race_id values: {sorted(race_ids)}"
    nums = runner_numbers(df)
    if len(nums) < 5:
        return False, f"only {len(nums)} runner numbers"
    return True, ""


def synthesize_runner_snapshot(card: pd.DataFrame, race_id: str) -> pd.DataFrame:
    """Create the smallest leakage-safe runner snapshot required by Rebuild."""
    nums = sorted(runner_numbers(card))
    if len(nums) < 5:
        raise ValueError(f"{race_id}: cannot synthesize runner snapshot from {len(nums)} horses")
    return pd.DataFrame({"race_id": [race_id] * len(nums), "horse_number": nums})


def inspect_card_date(source_root: Path, card_dir: Path) -> dict:
    canonical: list[str] = []
    ignored: list[str] = []
    schema_errors: list[dict] = []
    missing_archives: list[dict] = []

    for path in sorted(card_dir.glob("*.csv")):
        if not CANONICAL_RACE_FILE.fullmatch(path.name):
            ignored.append(path.name)
            continue
        rid = path.stem
        try:
            card = read_csv(path)
        except Exception as exc:  # noqa: BLE001
            schema_errors.append({"file": path.name, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        ok, reason = valid_race_frame(card, rid)
        if not ok:
            schema_errors.append({"file": path.name, "reason": reason})
            continue

        result_path = source_root / "data" / "race_results" / rid[:4] / f"{rid}.csv"
        payout_path = source_root / "data" / "race_payouts" / f"{rid}.csv"
        missing = [
            label for label, req in (("result", result_path), ("payout", payout_path))
            if not req.exists()
        ]
        if missing:
            missing_archives.append({"raceId": rid, "missing": missing})
            continue
        canonical.append(path.name)

    # Do not silently build a partial day: every canonical race card in the directory must
    # have a valid schema plus final result/payout before the date is declared complete.
    canonical_seen = sum(1 for p in card_dir.glob("*.csv") if CANONICAL_RACE_FILE.fullmatch(p.name))
    safe = bool(canonical) and not schema_errors and not missing_archives and len(canonical) == canonical_seen
    if schema_errors:
        reason = f"{len(schema_errors)} canonical race cards have unsupported schema/read errors"
    elif missing_archives:
        reason = f"result/payout missing for {len(missing_archives)} canonical races"
    elif not canonical:
        reason = "no valid 12-digit canonical race cards"
    else:
        reason = ""
    return {
        "safe": safe,
        "raceFiles": canonical,
        "ignoredFiles": ignored,
        "schemaErrors": schema_errors,
        "missingArchives": missing_archives,
        "reason": reason,
    }




RESULT_CARD_COLUMNS = [
    "race_id", "race_name", "surface", "distance_m", "waku", "horse_number",
    "horse_name", "sex_age", "weight_carried", "jockey", "trainer",
    "horse_id", "jockey_id", "trainer_id",
]
RESULT_REQUIRED_COLUMNS = set(RESULT_CARD_COLUMNS) | {
    "date", "finish_position", "popularity", "win_odds", "time",
}
BACKFILL_START = date(2026, 1, 1)
EXPECTED_FIRST_JRA_DATE = date(2026, 1, 4)


def _one_value(df: pd.DataFrame, column: str, race_id: str) -> str:
    if column not in df.columns:
        raise ValueError(f"{race_id}: result missing {column}")
    vals = [clean_str(x) for x in df[column].tolist() if clean_str(x)]
    unique = sorted(set(vals))
    if len(unique) != 1:
        raise ValueError(f"{race_id}: result {column} is not constant: {unique[:5]}")
    return unique[0]


def result_date(result: pd.DataFrame, race_id: str) -> str:
    raw = _one_value(result, "date", race_id)
    try:
        return pd.Timestamp(raw).date().isoformat()
    except Exception as exc:
        raise ValueError(f"{race_id}: invalid result date {raw!r}") from exc


def synthesize_card_from_result(result: pd.DataFrame, race_id: str) -> pd.DataFrame:
    """Recreate only immutable/pre-race program fields from a final result archive.

    Finish, clock, margin, final popularity, final odds and bodyweight are intentionally
    excluded.  This allows old dates without an archived race-card file to be rebuilt
    without leaking the target-race outcome into Predictjra.
    """
    missing = RESULT_REQUIRED_COLUMNS - set(result.columns)
    if missing:
        raise ValueError(f"{race_id}: result missing required columns: {sorted(missing)}")
    ok, reason = valid_race_frame(result, race_id)
    if not ok:
        raise ValueError(f"{race_id}: invalid result frame: {reason}")

    # Fields that define the race itself must be consistent across every runner row.
    _one_value(result, "race_name", race_id)
    _one_value(result, "surface", race_id)
    _one_value(result, "distance_m", race_id)
    result_date(result, race_id)

    nums = pd.to_numeric(result["horse_number"], errors="coerce")
    if nums.isna().any() or nums.astype(int).duplicated().any():
        raise ValueError(f"{race_id}: duplicate/invalid horse_number in result")

    card = result[RESULT_CARD_COLUMNS].copy()
    # The result archive may spell trainers more fully than the old race-card archive.
    # trainer_id is retained, so entity matching remains stable; no post-race field is kept.
    card = sanitize_card(card)
    ok, reason = valid_race_frame(card, race_id)
    if not ok:
        raise ValueError(f"{race_id}: invalid synthesized card: {reason}")
    return card


def _top3_groups(result: pd.DataFrame) -> dict[int, set[int]]:
    finish = pd.to_numeric(result.get("finish_position"), errors="coerce")
    horses = pd.to_numeric(result.get("horse_number"), errors="coerce")
    groups: dict[int, set[int]] = {}
    for f, h in zip(finish, horses):
        if pd.isna(f) or pd.isna(h):
            continue
        place = int(f)
        if 1 <= place <= 3:
            groups.setdefault(place, set()).add(int(h))
    return groups


def validate_result_payout(result: pd.DataFrame, payout: pd.DataFrame, race_id: str) -> None:
    """Fail closed when final-result and trifecta-payout archives disagree."""
    for name, frame in (("result", result), ("payout", payout)):
        if frame.empty:
            raise ValueError(f"{race_id}: empty {name}")
        if "race_id" not in frame.columns:
            raise ValueError(f"{race_id}: {name} missing race_id")
        ids = {clean_str(x) for x in frame["race_id"].tolist() if clean_str(x)}
        if ids != {race_id}:
            raise ValueError(f"{race_id}: {name} race_id mismatch {sorted(ids)}")

    # Every actual starter must have a final popularity label. Cancelled/excluded runners
    # can remain in the card but are allowed to have no numeric finish/popularity.
    finish = pd.to_numeric(result.get("finish_position"), errors="coerce")
    pop = pd.to_numeric(result.get("popularity"), errors="coerce")
    if (finish.notna() & pop.isna()).any():
        bad = pd.to_numeric(
            result.loc[finish.notna() & pop.isna(), "horse_number"], errors="coerce"
        ).dropna().astype(int).tolist()
        raise ValueError(f"{race_id}: starters missing final popularity: {bad}")

    groups = _top3_groups(result)
    top_horses = set().union(*groups.values()) if groups else set()
    if len(top_horses) < 3 or 1 not in groups:
        raise ValueError(f"{race_id}: result does not contain at least three top finishers")
    finish_by_horse: dict[int, int] = {}
    for place, horses_at_place in groups.items():
        for horse in horses_at_place:
            finish_by_horse[horse] = place

    if "bet_type" not in payout.columns or "combination" not in payout.columns or "amount" not in payout.columns:
        raise ValueError(f"{race_id}: payout schema missing bet_type/combination/amount")
    tri = payout[payout["bet_type"].astype(str).str.replace("3", "三").str.contains("三連単", na=False)]
    if tri.empty:
        # Some archives use the ASCII leading 3 without conversion-friendly spelling.
        tri = payout[payout["bet_type"].astype(str).str.contains(r"(?:3|三)連単", regex=True, na=False)]
    if tri.empty:
        raise ValueError(f"{race_id}: trifecta payout missing")

    valid_count = 0
    for _, row in tri.iterrows():
        nums = [int(x) for x in re.findall(r"\d{1,2}", clean_str(row.get("combination")))]
        amount = pd.to_numeric(pd.Series([row.get("amount")]), errors="coerce").iloc[0]
        if len(nums) != 3 or pd.isna(amount) or float(amount) <= 0:
            raise ValueError(f"{race_id}: malformed trifecta payout row")
        positions = [finish_by_horse.get(horse) for horse in nums]
        # Dead heats can produce finish sequences such as 1-1-3.  A payout is valid when
        # all three horses are within the official top-three finish positions and the
        # ordered ticket does not contradict their recorded finish order.
        if any(pos is None or pos > 3 for pos in positions) or positions != sorted(positions):
            raise ValueError(
                f"{race_id}: trifecta payout {nums} conflicts with result top finishers "
                f"positions={positions}"
            )
        valid_count += 1
    if valid_count == 0:
        raise ValueError(f"{race_id}: no valid trifecta payout rows")


NETKEIBA_BASE = "https://race.netkeiba.com"
WEB_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
)
WEB_SESSION = requests.Session()
WEB_SESSION.headers.update({
    "User-Agent": WEB_UA,
    "Accept-Language": "ja,en-US;q=0.8,en;q=0.6",
})


def _request_web(url: str, *, pause: float = 0.12) -> str:
    last = None
    for attempt in range(3):
        try:
            response = WEB_SESSION.get(
                url,
                timeout=30,
                headers={"Referer": "https://race.netkeiba.com/"},
            )
            response.raise_for_status()
            if not response.encoding or response.encoding.lower() == "iso-8859-1":
                response.encoding = response.apparent_encoding
            html = response.text
            if len(html) < 1000:
                raise RuntimeError(f"unexpectedly short response ({len(html)} bytes)")
            time.sleep(pause)
            return html
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"GET failed after retries: {url}: {last}")


def discover_netkeiba_calendar_dates(start: date, end: date) -> list[str]:
    """Discover actual central-JRA racing dates from netkeiba's monthly calendar.

    This prevents an entirely missing source-archive day from being silently overlooked.
    """
    dates: set[str] = set()
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        url = f"{NETKEIBA_BASE}/top/calendar.html?year={cursor.year}&month={cursor.month}"
        html = _request_web(url, pause=0.08)
        found = set(re.findall(r"kaisai_date=(20\d{6})", html))
        # A normal month with central racing must expose at least one dated race-list link.
        monthly = []
        for compact in found:
            try:
                d = datetime.strptime(compact, "%Y%m%d").date()
            except ValueError:
                continue
            if d.year == cursor.year and d.month == cursor.month and start <= d <= end:
                monthly.append(d.isoformat())
        if not monthly and cursor <= end:
            raise RuntimeError(
                f"netkeiba calendar returned no JRA dates for {cursor.year}-{cursor.month:02d}; "
                "refusing to assume a blank month"
            )
        dates.update(monthly)
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)
    return sorted(dates)


def discover_netkeiba_race_ids(target: date) -> list[str]:
    url = f"{NETKEIBA_BASE}/top/race_list.html?kaisai_date={target:%Y%m%d}"
    html = _request_web(url, pause=0.08)
    title = BeautifulSoup(html, "lxml").title
    title_text = title.get_text(" ", strip=True) if title else ""
    expected_date_label = f"{target.year}年{target.month}月{target.day}日"
    if expected_date_label not in title_text:
        raise RuntimeError(
            f"race-list date mismatch for {target}: page title={title_text[:120]!r}"
        )
    ids = []
    seen = set()
    for rid in re.findall(r"race_id=(\d{12})", html):
        if rid in seen or rid[:4] != str(target.year):
            continue
        if rid[4:6] not in {f"{i:02d}" for i in range(1, 11)}:
            continue
        race_no = int(rid[-2:])
        if not 1 <= race_no <= 12:
            continue
        seen.add(rid)
        ids.append(rid)
    ids.sort(key=lambda x: (int(x[4:6]), int(x[6:8]), int(x[8:10]), int(x[10:12])))
    if not ids:
        raise RuntimeError(f"netkeiba race list is empty for calendar race date {target}")

    # Each active central-JRA meeting publishes a 1R..12R slot set. Cancelled races still
    # retain a race id on the day list, so a gap here signals a scrape problem.
    by_slot: dict[tuple[str, str, str], set[int]] = {}
    for rid in ids:
        by_slot.setdefault((rid[4:6], rid[6:8], rid[8:10]), set()).add(int(rid[10:12]))
    expected = set(range(1, 13))
    bad = []
    for slot, numbers in sorted(by_slot.items()):
        if numbers != expected:
            bad.append({
                "slot": slot,
                "missing": sorted(expected - numbers),
                "unexpected": sorted(numbers - expected),
            })
    if bad:
        raise RuntimeError(f"netkeiba race-list structural gap on {target}: {bad[:10]}")
    return ids


def _header_index(headers: list[str], *needles: str) -> int | None:
    for idx, header in enumerate(headers):
        h = re.sub(r"\s+", "", header)
        if all(needle in h for needle in needles):
            return idx
    return None


def _entity_id(cell, kind: str) -> str:
    if cell is None:
        return ""
    for anchor in cell.find_all("a", href=True):
        href = anchor.get("href") or ""
        # Handles /horse/2022101234/, /jockey/00666/, /trainer/result/recent/01105/.
        m = re.search(rf"/{kind}/(?:[^?#]*/)?(\d{{4,12}})/?(?:[?#]|$)", href)
        if m:
            return m.group(1)
    return ""


def _cell_text(cells, idx: int | None) -> str:
    if idx is None or idx >= len(cells):
        return ""
    return cells[idx].get_text(" ", strip=True)


def parse_netkeiba_full_result(html: str, race_id: str, target: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse a complete final-result table plus trifecta payout from netkeiba.

    Only this repair function sees post-race fields. `synthesize_card_from_result()` later
    projects the strict pre-race subset before Predictjra is run.
    """
    soup = BeautifulSoup(html, "lxml")
    race_name_node = soup.select_one(".RaceName") or soup.find("h1")
    race_name = clean_str(race_name_node.get_text(" ", strip=True) if race_name_node else "")
    if not race_name:
        raise ValueError(f"{race_id}: netkeiba result missing race name")

    race_data_node = soup.select_one(".RaceData01")
    race_data = race_data_node.get_text(" ", strip=True) if race_data_node else soup.get_text(" ", strip=True)[:5000]
    sm = re.search(r"(?:^|[/\s])(芝|ダート|ダ|障)(\d{3,4})m", race_data)
    if not sm:
        raise ValueError(f"{race_id}: cannot parse surface/distance from result page")
    surface = {"ダ": "ダート"}.get(sm.group(1), sm.group(1))
    distance_m = int(sm.group(2))
    direction = ""
    dm = re.search(r"\([^)]*(右|左)[^)]*\)", race_data)
    if dm:
        direction = dm.group(1)
    wm = re.search(r"天候[:：]\s*([^/\s]+)", race_data)
    cm = re.search(r"馬場[:：]\s*([^/\s]+)", race_data)
    weather = wm.group(1) if wm else ""
    track_condition = cm.group(1) if cm else ""

    result_table = soup.select_one("table.RaceTable01") or soup.select_one(".ResultTableWrap table")
    if result_table is None:
        for table in soup.find_all("table"):
            heads = [c.get_text(" ", strip=True) for c in table.find_all("th")]
            joined = "|".join(heads)
            if "着順" in joined and "馬番" in joined and "馬名" in joined:
                result_table = table
                break
    if result_table is None:
        raise ValueError(f"{race_id}: netkeiba final-result table not found")

    header_row = None
    for row in result_table.find_all("tr"):
        cells = row.find_all(["th", "td"], recursive=False)
        texts = [c.get_text(" ", strip=True) for c in cells]
        joined = "|".join(texts)
        if "着順" in joined and "馬番" in joined and "馬名" in joined:
            header_row = row
            break
    if header_row is None:
        raise ValueError(f"{race_id}: result header row not found")
    headers = [c.get_text(" ", strip=True) for c in header_row.find_all(["th", "td"], recursive=False)]
    idx = {
        "finish_position": _header_index(headers, "着順"),
        "waku": _header_index(headers, "枠"),
        "horse_number": _header_index(headers, "馬番"),
        "horse_name": _header_index(headers, "馬名"),
        "sex_age": _header_index(headers, "性齢"),
        "weight_carried": _header_index(headers, "斤量"),
        "jockey": _header_index(headers, "騎手"),
        "time": _header_index(headers, "タイム"),
        "margin": _header_index(headers, "着差"),
        "popularity": _header_index(headers, "人気"),
        "win_odds": _header_index(headers, "単勝", "オッズ"),
        "last_3f": _header_index(headers, "後3F"),
        "passing_order": _header_index(headers, "コーナー", "通過"),
        "trainer": _header_index(headers, "厩舎"),
        "horse_weight": _header_index(headers, "馬体重"),
    }
    for required in ("finish_position", "horse_number", "horse_name", "sex_age", "weight_carried", "jockey", "popularity", "win_odds", "trainer"):
        if idx[required] is None:
            raise ValueError(f"{race_id}: result table missing column {required}; headers={headers}")

    rows: list[dict] = []
    for row in result_table.find_all("tr"):
        if row is header_row:
            continue
        cells = row.find_all(["th", "td"], recursive=False)
        if not cells or idx["horse_number"] is None or len(cells) <= idx["horse_number"]:
            continue
        horse_text = _cell_text(cells, idx["horse_number"])
        hm = re.search(r"\d{1,2}", horse_text)
        if not hm:
            continue
        horse_number = int(hm.group())
        if not 1 <= horse_number <= 18:
            continue
        finish_text = _cell_text(cells, idx["finish_position"])
        fm = re.match(r"\s*(\d+)", finish_text)
        finish_position = int(fm.group(1)) if fm else ""
        waku_text = _cell_text(cells, idx["waku"])
        waku_m = re.search(r"[1-8]", waku_text)
        waku = int(waku_m.group()) if waku_m else ""
        pop_text = _cell_text(cells, idx["popularity"])
        pop_m = re.search(r"\d+", pop_text)
        popularity = int(pop_m.group()) if pop_m else ""
        odds_text = _cell_text(cells, idx["win_odds"])
        odds_m = re.search(r"\d+(?:\.\d+)?", odds_text.replace(",", ""))
        win_odds = float(odds_m.group()) if odds_m else ""
        weight_text = _cell_text(cells, idx["weight_carried"])
        weight_m = re.search(r"\d+(?:\.\d+)?", weight_text)
        weight_carried = float(weight_m.group()) if weight_m else ""

        horse_cell = cells[idx["horse_name"]] if idx["horse_name"] is not None else None
        jockey_cell = cells[idx["jockey"]] if idx["jockey"] is not None else None
        trainer_cell = cells[idx["trainer"]] if idx["trainer"] is not None else None
        rows.append({
            "race_id": race_id,
            "race_name": race_name,
            "date": target.isoformat(),
            "surface": surface,
            "distance_m": distance_m,
            "direction": direction,
            "weather": weather,
            "track_condition": track_condition,
            "finish_position": finish_position,
            "waku": waku,
            "horse_number": horse_number,
            "horse_name": _cell_text(cells, idx["horse_name"]),
            "horse_id": _entity_id(horse_cell, "horse"),
            "sex_age": _cell_text(cells, idx["sex_age"]),
            "weight_carried": weight_carried,
            "jockey": _cell_text(cells, idx["jockey"]),
            "jockey_id": _entity_id(jockey_cell, "jockey"),
            "time": _cell_text(cells, idx["time"]),
            "margin": _cell_text(cells, idx["margin"]),
            "popularity": popularity,
            "win_odds": win_odds,
            "last_3f": _cell_text(cells, idx["last_3f"]),
            "passing_order": _cell_text(cells, idx["passing_order"]),
            "trainer": _cell_text(cells, idx["trainer"]),
            "trainer_id": _entity_id(trainer_cell, "trainer"),
            "horse_weight": _cell_text(cells, idx["horse_weight"]),
        })

    result = pd.DataFrame(rows)
    if len(result) < 5:
        raise ValueError(f"{race_id}: only {len(result)} result runners parsed")
    if result["horse_number"].astype(int).duplicated().any():
        raise ValueError(f"{race_id}: duplicate horse numbers parsed from netkeiba")
    if (result["horse_id"].astype(str).str.len() == 0).any():
        missing = result.loc[result["horse_id"].astype(str).str.len() == 0, "horse_number"].tolist()
        raise ValueError(f"{race_id}: horse IDs missing from result page: {missing}")

    payout_rows: list[dict] = []
    for tr in soup.find_all("tr"):
        cells = tr.find_all(["th", "td"], recursive=False)
        if len(cells) < 3:
            continue
        bet_text = re.sub(r"\s+", "", cells[0].get_text(" ", strip=True))
        if "3連単" not in bet_text and "三連単" not in bet_text:
            continue
        combo_nums = [int(x) for x in re.findall(r"(?<!\d)(\d{1,2})(?!\d)", cells[1].get_text(" ", strip=True))]
        amounts = [int(x.replace(",", "")) for x in re.findall(r"([\d,]+)\s*円", cells[2].get_text(" ", strip=True))]
        pops = []
        if len(cells) >= 4:
            pops = [int(x) for x in re.findall(r"(\d+)\s*人気", cells[3].get_text(" ", strip=True))]
        if not combo_nums or len(combo_nums) % 3 != 0:
            raise ValueError(f"{race_id}: cannot parse trifecta combinations from netkeiba")
        combos = [combo_nums[i:i+3] for i in range(0, len(combo_nums), 3)]
        if len(amounts) != len(combos):
            raise ValueError(
                f"{race_id}: trifecta combination/payout count mismatch {len(combos)} != {len(amounts)}"
            )
        if pops and len(pops) != len(combos):
            pops = []
        for i, (combo, amount) in enumerate(zip(combos, amounts)):
            payout_rows.append({
                "race_id": race_id,
                "bet_type": "三連単",
                "combination": "-".join(str(x) for x in combo),
                "amount": amount,
                "popularity": pops[i] if pops else "",
            })
    payout = pd.DataFrame(payout_rows)
    if payout.empty:
        raise ValueError(f"{race_id}: netkeiba trifecta payout not parsed")
    validate_result_payout(result, payout, race_id)
    return result, payout


def repair_result_archive_from_web(
    source_root: Path,
    *,
    start: date,
    end: date,
) -> dict:
    """Repair missing 2026 result/payout files against netkeiba's calendar and race lists.

    Existing valid source files are never replaced. Invalid/mismatched files abort the
    refresh rather than being silently overwritten. Missing files are generated only when
    a complete netkeiba final-result page passes the same result/payout consistency checks.
    """
    calendar_dates = discover_netkeiba_calendar_dates(start, end)
    expected_by_date: dict[str, list[str]] = {}
    repaired: list[dict] = []
    result_root = source_root / "data" / "race_results" / str(start.year)
    payout_root = source_root / "data" / "race_payouts"
    result_root.mkdir(parents=True, exist_ok=True)
    payout_root.mkdir(parents=True, exist_ok=True)

    for date_s in calendar_dates:
        target = datetime.strptime(date_s, "%Y-%m-%d").date()
        ids = discover_netkeiba_race_ids(target)
        expected_by_date[date_s] = ids
        for rid in ids:
            result_path = result_root / f"{rid}.csv"
            payout_path = payout_root / f"{rid}.csv"
            if result_path.is_file() and payout_path.is_file():
                # Do not trust presence alone: validate every existing target race pair.
                result = read_csv(result_path)
                payout = read_csv(payout_path)
                if result_date(result, rid) != date_s:
                    raise RuntimeError(
                        f"{rid}: source result date mismatch {result_date(result, rid)} != {date_s}"
                    )
                validate_result_payout(result, payout, rid)
                continue

            url = f"{NETKEIBA_BASE}/race/result.html?race_id={rid}"
            html = _request_web(url, pause=0.18)
            result, payout = parse_netkeiba_full_result(html, rid, target)
            if result_date(result, rid) != date_s:
                raise RuntimeError(f"{rid}: repaired result date mismatch")

            # If only one side was missing, the existing side must agree with the web repair.
            if result_path.is_file():
                existing_result = read_csv(result_path)
                validate_result_payout(existing_result, payout, rid)
                result = existing_result
            if payout_path.is_file():
                existing_payout = read_csv(payout_path)
                validate_result_payout(result, existing_payout, rid)
                payout = existing_payout

            result_was_missing = not result_path.is_file()
            payout_was_missing = not payout_path.is_file()
            result.to_csv(result_path, index=False, encoding="utf-8-sig")
            payout.to_csv(payout_path, index=False, encoding="utf-8-sig")
            repaired.append({
                "date": date_s,
                "raceId": rid,
                "resultCreated": result_was_missing,
                "payoutCreated": payout_was_missing,
                "source": url,
            })

    return {
        "calendarDates": calendar_dates,
        "expectedByDate": expected_by_date,
        "repaired": repaired,
    }


def validate_archive_structure(
    by_date: dict[str, dict],
    expected_by_date: dict[str, list[str]] | None = None,
) -> None:
    """Reject silent holes in the central-JRA result archive.

    When an independently discovered netkeiba race list is available, it is the exact
    expected race-id set for each actual calendar date. The legacy 1R..12R/meeting-day
    structural check remains only for offline unit tests or old cache sources where no
    independent date/race list was supplied.
    """
    if expected_by_date is not None:
        errors: list[str] = []
        actual_dates = {d for d in by_date if not d.startswith("__")}
        expected_dates = set(expected_by_date)
        missing_dates = sorted(expected_dates - actual_dates)
        unexpected_dates = sorted(actual_dates - expected_dates)
        if missing_dates:
            errors.append(f"missing race dates={missing_dates}")
        if unexpected_dates:
            errors.append(f"unexpected race dates={unexpected_dates}")
        for date_s in sorted(expected_dates & actual_dates):
            expected = set(expected_by_date.get(date_s) or [])
            actual = {Path(x).stem for x in by_date[date_s].get("raceFiles", [])}
            if actual != expected:
                errors.append(
                    f"{date_s}: missing race ids={sorted(expected - actual)} "
                    f"unexpected race ids={sorted(actual - expected)}"
                )
        if errors:
            raise RuntimeError(
                "Historical result archive does not match independently discovered "
                "JRA race lists; refresh aborted: " + " | ".join(errors[:50])
            )
        return

    slots: dict[tuple[int, int, int], set[int]] = {}
    slot_dates: dict[tuple[int, int, int], set[str]] = {}
    meeting_days: dict[tuple[int, int], set[int]] = {}
    errors: list[str] = []

    for date_s, info in by_date.items():
        if date_s.startswith("__"):
            continue
        for filename in info.get("raceFiles", []):
            rid = Path(filename).stem
            if not re.fullmatch(r"\d{12}", rid):
                errors.append(f"{date_s}: invalid race id {rid}")
                continue
            venue = int(rid[4:6])
            meeting = int(rid[6:8])
            meeting_day = int(rid[8:10])
            race_no = int(rid[10:12])
            key = (venue, meeting, meeting_day)
            slots.setdefault(key, set()).add(race_no)
            slot_dates.setdefault(key, set()).add(date_s)
            meeting_days.setdefault((venue, meeting), set()).add(meeting_day)

    expected_races = set(range(1, 13))
    for key, race_nos in sorted(slots.items()):
        if race_nos != expected_races:
            missing = sorted(expected_races - race_nos)
            extra = sorted(race_nos - expected_races)
            errors.append(f"meeting-day {key}: missing races={missing}, unexpected={extra}")
        dates = slot_dates.get(key, set())
        if len(dates) != 1:
            errors.append(f"meeting-day {key}: mapped to multiple dates {sorted(dates)}")

    for key, days in sorted(meeting_days.items()):
        if not days:
            continue
        expected_days = set(range(1, max(days) + 1))
        if days != expected_days:
            errors.append(
                f"meeting {key}: missing meeting-day numbers {sorted(expected_days - days)}"
            )

    if errors:
        raise RuntimeError(
            "Historical result archive has structural holes; refresh aborted: "
            + " | ".join(errors[:50])
        )

def inspect_result_backfill(
    source_root: Path,
    *,
    start: date = BACKFILL_START,
    expected_by_date: dict[str, list[str]] | None = None,
) -> dict[str, dict]:
    """Index complete 2026 result/payout races, grouped by actual race date."""
    result_root = source_root / "data" / "race_results" / "2026"
    if not result_root.is_dir():
        raise FileNotFoundError(result_root)

    by_date: dict[str, dict] = {}
    for path in sorted(result_root.glob("*.csv")):
        if not CANONICAL_RACE_FILE.fullmatch(path.name):
            continue
        rid = path.stem
        try:
            result = read_csv(path)
            d = result_date(result, rid)
            if pd.Timestamp(d).date() < start:
                continue
            card = synthesize_card_from_result(result, rid)
            payout_path = source_root / "data" / "race_payouts" / f"{rid}.csv"
            if not payout_path.is_file():
                raise FileNotFoundError(f"payout missing: {payout_path}")
            payout = read_csv(payout_path)
            validate_result_payout(result, payout, rid)
            # Keep the synthesized card in memory only for validation; cache staging rebuilds
            # it from the immutable result file so there is a single source of truth.
            _ = card
            entry = by_date.setdefault(d, {"raceFiles": [], "errors": []})
            entry["raceFiles"].append(path.name)
        except Exception as exc:  # noqa: BLE001
            # If even the date field is broken, keep the error under a special bucket so the
            # entire cache refresh fails rather than silently dropping the file.
            key = "__invalid__"
            try:
                raw = read_csv(path)
                key = result_date(raw, rid)
            except Exception:
                pass
            entry = by_date.setdefault(key, {"raceFiles": [], "errors": []})
            entry["errors"].append({"file": path.name, "reason": f"{type(exc).__name__}: {exc}"})

    if "__invalid__" in by_date:
        errors = by_date["__invalid__"]["errors"]
        raise RuntimeError(f"Invalid 2026 historical result files: {errors[:10]}")

    for d, info in by_date.items():
        info["raceFiles"] = sorted(set(info["raceFiles"]))
        info["safe"] = bool(info["raceFiles"]) and not info["errors"]
        info["reason"] = "" if info["safe"] else f"{len(info['errors'])} result/payout validation errors"

    validate_archive_structure(by_date, expected_by_date=expected_by_date)
    actual_dates = sorted(d for d in by_date if not d.startswith("__"))
    if not actual_dates or actual_dates[0] != EXPECTED_FIRST_JRA_DATE.isoformat():
        raise RuntimeError(
            "Historical result archive does not start at the expected first 2026 JRA day: "
            f"expected={EXPECTED_FIRST_JRA_DATE.isoformat()} actual={actual_dates[0] if actual_dates else 'none'}"
        )
    return by_date


def choose_runner_snapshot(
    source_root: Path,
    compact: str,
    race_id: str,
    sanitized_card: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Use a sanitized old prediction only when its runner set is sound; else synthesize."""
    pred_path = source_root / "data" / "predictions" / compact / f"{race_id}.csv"
    card_nums = runner_numbers(sanitized_card)
    if pred_path.exists():
        try:
            pred = read_csv(pred_path)
            sanitized, removed = sanitize_prediction_snapshot(pred)
            ok, reason = valid_race_frame(sanitized, race_id)
            if ok and runner_numbers(sanitized) == card_nums:
                return sanitized, {
                    "source": "sanitized-legacy-prediction",
                    "removedColumns": removed,
                }
            fallback_reason = reason or "runner set differs from archived race card"
        except Exception as exc:  # noqa: BLE001
            fallback_reason = f"{type(exc).__name__}: {exc}"
    else:
        fallback_reason = "legacy prediction snapshot absent"

    return synthesize_runner_snapshot(sanitized_card, race_id), {
        "source": "synthesized-from-sanitized-card",
        "reason": fallback_reason,
        "removedColumns": [],
    }


def copy_file(src: Path, root: Path, dst_root: Path) -> None:
    rel = src.relative_to(root)
    dst = dst_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def latest_2026_source_date(source_root: Path) -> date:
    dates: list[date] = []
    card_root = source_root / "data" / "race_cards"
    if card_root.is_dir():
        for child in card_root.iterdir():
            if child.is_dir() and re.fullmatch(r"2026\d{4}", child.name):
                try:
                    dates.append(datetime.strptime(child.name, "%Y%m%d").date())
                except ValueError:
                    pass
    result_root = source_root / "data" / "race_results" / "2026"
    if result_root.is_dir():
        for path in result_root.glob("*.csv"):
            if not CANONICAL_RACE_FILE.fullmatch(path.name):
                continue
            try:
                dates.append(datetime.strptime(result_date(read_csv(path), path.stem), "%Y-%m-%d").date())
            except Exception:
                continue
    if not dates:
        raise RuntimeError("Cannot determine latest 2026 historical source date")
    latest = max(dates)
    if latest < EXPECTED_FIRST_JRA_DATE:
        raise RuntimeError(f"Latest source date {latest} predates 2026 JRA season")
    return latest


def build_cache(
    source_root: Path,
    cache_dir: Path,
    *,
    web_discovery: bool = True,
) -> dict:
    card_root = source_root / "data" / "race_cards"
    if not card_root.exists():
        raise FileNotFoundError(card_root)

    # Results/payouts are the authoritative immutable archive for dates that predate the
    # saved race-card snapshots. Before trusting the source repository, independently
    # enumerate actual 2026 JRA dates/race IDs and repair only genuinely missing archives.
    web_repair = {"calendarDates": [], "expectedByDate": None, "repaired": []}
    if web_discovery:
        latest = latest_2026_source_date(source_root)
        web_repair = repair_result_archive_from_web(
            source_root, start=EXPECTED_FIRST_JRA_DATE, end=latest
        )
    result_backfill = inspect_result_backfill(
        source_root, expected_by_date=web_repair.get("expectedByDate")
    )

    card_inspections: dict[str, dict] = {}
    ignored_files: list[dict] = []
    for card_dir in sorted(card_root.iterdir()):
        if not card_dir.is_dir() or not re.fullmatch(r"20\d{6}", card_dir.name):
            continue
        if not any(card_dir.glob("*.csv")):
            continue
        try:
            date_s = datetime.strptime(card_dir.name, "%Y%m%d").date().isoformat()
        except ValueError:
            continue
        inspection = inspect_card_date(source_root, card_dir)
        card_inspections[date_s] = inspection
        if inspection["ignoredFiles"]:
            ignored_files.append({"date": date_s, "files": inspection["ignoredFiles"]})

    available_dates = sorted(set(result_backfill) | set(card_inspections))
    safe_dates: list[str] = []
    skipped_dates: list[dict] = []
    date_sources: dict[str, str] = {}
    inspections: dict[str, dict] = {}

    for date_s in available_dates:
        if date_s.startswith("__"):
            continue
        card_info = card_inspections.get(date_s)
        result_info = result_backfill.get(date_s)

        # Prefer a real archived pre-race card when it is complete. Otherwise use the
        # leakage-safe card synthesized from final result rows. If both exist, their race
        # sets must match exactly; this catches silent partial-day archives.
        if card_info and not card_info.get("safe"):
            skipped_dates.append({
                "date": date_s,
                "reason": f"archived race card is present but unsafe: {card_info.get('reason')}",
                "schemaErrors": card_info.get("schemaErrors", []),
                "missingArchives": card_info.get("missingArchives", []),
            })
            continue

        if card_info and card_info.get("safe"):
            card_ids = {Path(x).stem for x in card_info.get("raceFiles", [])}
            result_ids = {Path(x).stem for x in (result_info or {}).get("raceFiles", [])}
            if result_ids and card_ids != result_ids:
                missing_card = sorted(result_ids - card_ids)
                missing_result = sorted(card_ids - result_ids)
                skipped_dates.append({
                    "date": date_s,
                    "reason": "archived card/result race-set mismatch",
                    "missingCardRaceIds": missing_card,
                    "missingResultRaceIds": missing_result,
                })
                continue
            safe_dates.append(date_s)
            date_sources[date_s] = "archived-race-card"
            inspections[date_s] = card_info
            continue

        if result_info and result_info.get("safe"):
            safe_dates.append(date_s)
            date_sources[date_s] = "result-derived-pre-race-card"
            inspections[date_s] = result_info
            continue

        reasons = []
        if card_info and not card_info.get("safe"):
            reasons.append(f"card: {card_info.get('reason')}")
        if result_info and not result_info.get("safe"):
            reasons.append(f"result: {result_info.get('reason')}")
        skipped_dates.append({
            "date": date_s,
            "reason": "; ".join(reasons) or "no complete card/result archive",
            "schemaErrors": (card_info or {}).get("schemaErrors", []),
            "missingArchives": (card_info or {}).get("missingArchives", []),
            "resultErrors": (result_info or {}).get("errors", []),
        })

    if skipped_dates:
        # Accuracy takes priority over partial coverage. One broken date means the refresh
        # stops and the previous durable cache remains untouched.
        raise RuntimeError(
            "Historical cache refresh refused because one or more dates are incomplete: "
            + json.dumps(skipped_dates[:20], ensure_ascii=False)
        )
    if not safe_dates:
        raise RuntimeError("No complete historical dates are available for caching")

    source_sha = source_commit(source_root)
    cache_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="predictjra-history-cache-") as tmp:
        staging = Path(tmp) / "source"
        target_horse_ids: set[str] = set()
        date_race_counts: dict[str, int] = {}
        copied_paths: set[str] = set()
        runner_snapshot_summary: list[dict] = []

        for date_s in safe_dates:
            compact = date_s.replace("-", "")
            source_kind = date_sources[date_s]
            race_names = inspections[date_s]["raceFiles"]
            date_race_counts[date_s] = len(race_names)
            legacy_count = 0
            synthesized_count = 0
            result_card_count = 0
            removed_columns: set[str] = set()
            synthesized_reasons: list[dict] = []

            for name in race_names:
                rid = Path(name).stem
                result_path = source_root / "data" / "race_results" / "2026" / f"{rid}.csv"
                payout_path = source_root / "data" / "race_payouts" / f"{rid}.csv"
                for path in (result_path, payout_path):
                    if not path.exists():
                        raise FileNotFoundError(f"cache source missing: {path}")
                result = read_csv(result_path)
                payout = read_csv(payout_path)
                validate_result_payout(result, payout, rid)

                if source_kind == "archived-race-card":
                    card_path = source_root / "data" / "race_cards" / compact / f"{rid}.csv"
                    if not card_path.is_file():
                        raise FileNotFoundError(f"cache source missing: {card_path}")
                    card = sanitize_card(read_csv(card_path))
                    card_dst = staging / card_path.relative_to(source_root)
                    snapshot, snapshot_meta = choose_runner_snapshot(
                        source_root, compact, rid, card
                    )
                else:
                    card = synthesize_card_from_result(result, rid)
                    card_dst = staging / "data" / "race_cards" / compact / f"{rid}.csv"
                    snapshot = synthesize_runner_snapshot(card, rid)
                    snapshot_meta = {
                        "source": "synthesized-from-result-derived-card",
                        "reason": "archived pre-race card absent; immutable program facts projected from final result archive",
                        "removedColumns": [],
                    }
                    result_card_count += 1

                ok, reason = valid_race_frame(card, rid)
                if not ok:
                    raise ValueError(f"{rid}: invalid sanitized card: {reason}")
                # Prohibited current-race fields must never survive into the cache card.
                leaked = sorted(PROHIBITED_CURRENT_COLUMNS & set(card.columns))
                if leaked:
                    raise ValueError(f"{rid}: synthesized/sanitized card leaked columns: {leaked}")

                card_dst.parent.mkdir(parents=True, exist_ok=True)
                card.to_csv(card_dst, index=False, encoding="utf-8-sig")
                copied_paths.add(str(card_dst.relative_to(staging)))

                snapshot_dst = staging / "data" / "predictions" / compact / f"{rid}.csv"
                snapshot_dst.parent.mkdir(parents=True, exist_ok=True)
                snapshot.to_csv(snapshot_dst, index=False, encoding="utf-8-sig")
                copied_paths.add(str(snapshot_dst.relative_to(staging)))
                if snapshot_meta["source"] == "sanitized-legacy-prediction":
                    legacy_count += 1
                    removed_columns.update(snapshot_meta.get("removedColumns", []))
                else:
                    synthesized_count += 1
                    synthesized_reasons.append({"raceId": rid, "reason": snapshot_meta.get("reason", "")})

                copy_file(result_path, source_root, staging)
                copy_file(payout_path, source_root, staging)
                copied_paths.add(str(result_path.relative_to(source_root)))
                copied_paths.add(str(payout_path.relative_to(source_root)))

                if "horse_id" in card.columns:
                    target_horse_ids.update(
                        x for x in card["horse_id"].apply(clean_str).tolist() if x
                    )

            runner_snapshot_summary.append({
                "date": date_s,
                "races": len(race_names),
                "cardSource": source_kind,
                "resultDerivedCards": result_card_count,
                "sanitizedLegacySnapshots": legacy_count,
                "synthesizedFromCards": synthesized_count,
                "removedColumns": sorted(removed_columns),
                "synthesizedReasons": synthesized_reasons,
            })

        ids = sorted(target_horse_ids)
        if not ids:
            raise RuntimeError("No horse IDs were found while building historical cache")
        pattern = re.compile(r",(?:" + "|".join(re.escape(x) for x in ids) + r"),")

        history_files = 0
        result_root = source_root / "data" / "race_results"
        for path in sorted(result_root.glob("*/*.csv")):
            try:
                text = path.read_text(encoding="utf-8-sig", errors="ignore")
            except Exception:
                continue
            if not pattern.search(text):
                continue
            rel = str(path.relative_to(source_root))
            if rel not in copied_paths:
                copy_file(path, source_root, staging)
                copied_paths.add(rel)
            history_files += 1

        tmp_archive = Path(tmp) / "history-source.tar.gz"
        with tarfile.open(tmp_archive, "w:gz", compresslevel=6) as tar:
            tar.add(staging / "data", arcname="data", recursive=True)

        archive_path = cache_dir / "history-source.tar.gz"
        for stale in cache_dir.glob("history-source.tar.gz.part*"):
            stale.unlink()
        if archive_path.exists():
            archive_path.unlink()

        archive_size = tmp_archive.stat().st_size
        archive_sha = sha256_file(tmp_archive)
        archive_parts: list[str] = []
        if archive_size <= 90 * 1024 * 1024:
            shutil.copy2(tmp_archive, archive_path)
            archive_name: str | None = archive_path.name
        else:
            archive_name = None
            chunk_size = 80 * 1024 * 1024
            with tmp_archive.open("rb") as src:
                index = 1
                while True:
                    chunk = src.read(chunk_size)
                    if not chunk:
                        break
                    part = cache_dir / f"history-source.tar.gz.part{index:03d}"
                    part.write_bytes(chunk)
                    archive_parts.append(part.name)
                    index += 1

        result_derived_dates = [d for d in safe_dates if date_sources[d] == "result-derived-pre-race-card"]
        manifest = {
            "cacheVersion": CACHE_VERSION,
            "generatedAt": datetime.now(JST).isoformat(timespec="seconds"),
            "sourceRepository": SOURCE_REPO,
            "sourceRef": SOURCE_REF,
            "sourceCommit": source_sha,
            "availableDates": available_dates,
            "safeDates": safe_dates,
            "skippedDates": [],
            "ignoredFiles": ignored_files,
            "dateRaceCounts": date_race_counts,
            "dateSources": date_sources,
            "resultDerivedDates": result_derived_dates,
            "webRepair": {
                "enabled": bool(web_discovery),
                "calendarDates": web_repair.get("calendarDates") or [],
                "repairedRaceCount": len(web_repair.get("repaired") or []),
                "repairedRaces": web_repair.get("repaired") or [],
                "source": "netkeiba monthly calendar + daily race list + final result page",
            },
            "runnerSnapshotSummary": runner_snapshot_summary,
            "targetHorseCount": len(target_horse_ids),
            "historicalResultFiles": history_files,
            "cachedFileCount": len(copied_paths),
            "archive": {
                "path": archive_name,
                "parts": archive_parts,
                "sizeBytes": archive_size,
                "sha256": archive_sha,
            },
            "policy": {
                "raceEnumeration": "independently discover actual JRA dates from netkeiba monthly calendars and exact race IDs from each daily race list; then require the repaired result archive to match that set exactly",
                "missingArchiveRepair": "create only missing result/payout files from the matching netkeiba final-result page; existing valid files are retained and any disagreement aborts refresh",
                "resultDerivedCard": "when an archived pre-race card is absent, project only race_id/name/surface/distance/frame/horse/name/sex-age/assigned-weight/jockey/trainer/entity IDs from the final result; never project finish/time/margin/popularity/odds/bodyweight",
                "resultPayoutCrossCheck": "every starter requires final popularity; top3 result groups must exist; every trifecta payout combination must agree with the result top3",
                "stored": "sanitized program/runner facts, runner-set snapshots, final results, trifecta payouts, and required historical race result files",
                "notStored": "target-race current odds/actual popularity/bodyweight, archived model outputs, Predictjra derived indices, selections, hit judgement, or recovery-rate calculations",
                "runnerSnapshot": "use sanitized legacy prediction only when valid and runner-identical; otherwise synthesize race_id/horse_number from the sanitized/pre-race-only card",
                "completion": "cache refresh is fail-closed: independently discovered race-date/race-id sets must exactly equal the repaired result archive; any schema, card/result race-set, result/payout, web-discovery, archive-structure, or leakage inconsistency aborts before the durable cache is replaced",
            },
        }
        (cache_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--cache-dir", default="data/history_cache", type=Path)
    args = parser.parse_args()
    manifest = build_cache(args.source_root.resolve(), args.cache_dir.resolve())
    print(json.dumps({
        "cacheVersion": manifest["cacheVersion"],
        "sourceCommit": manifest["sourceCommit"],
        "safeDates": manifest["safeDates"],
        "skippedDates": [x["date"] for x in manifest["skippedDates"]],
        "dateRaceCounts": manifest["dateRaceCounts"],
        "runnerSnapshotSummary": [
            {
                "date": x["date"],
                "races": x["races"],
                "sanitizedLegacySnapshots": x["sanitizedLegacySnapshots"],
                "synthesizedFromCards": x["synthesizedFromCards"],
            }
            for x in manifest["runnerSnapshotSummary"]
        ],
        "targetHorseCount": manifest["targetHorseCount"],
        "historicalResultFiles": manifest["historicalResultFiles"],
        "cachedFileCount": manifest["cachedFileCount"],
        "archiveSizeBytes": manifest["archive"]["sizeBytes"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
