#!/usr/bin/env python3
"""Build a durable leakage-safe historical facts cache for Predictjra.

Target races are resolved from exact historical race IDs. SportsNavi server-rendered
meeting lists are the primary independent enumeration source and netkeiba static date lists
are the cross-check/fallback. The code never assumes that every meeting must contain 1R-12R.

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
CACHE_VERSION = "predictjra-historical-facts-v7-multisource-complete"
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



RACE_NETKEIBA_BASE = "https://race.netkeiba.com"
DB_NETKEIBA_BASE = "https://db.netkeiba.com"
SPORTSNAVI_BASE = "https://sports.yahoo.co.jp/keiba"
WEB_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
)
WEB_SESSION = requests.Session()
WEB_SESSION.headers.update({
    "User-Agent": WEB_UA,
    "Accept-Language": "ja,en-US;q=0.8,en;q=0.6",
})


def _request_web(url: str, *, pause: float = 0.10, attempts: int = 5) -> str:
    """GET server-rendered public racing pages with bounded retry/backoff."""
    last = None
    for attempt in range(attempts):
        try:
            headers = {}
            if "sports.yahoo.co.jp" in url:
                headers["Referer"] = "https://sports.yahoo.co.jp/keiba/"
            elif "netkeiba.com" in url:
                headers["Referer"] = "https://db.netkeiba.com/"
            response = WEB_SESSION.get(url, timeout=35, headers=headers)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                try:
                    delay = max(1.0, min(20.0, float(retry_after)))
                except (TypeError, ValueError):
                    delay = min(12.0, 1.5 * (2 ** attempt))
                time.sleep(delay)
                raise RuntimeError("HTTP 429 rate limited")
            response.raise_for_status()
            if not response.encoding or response.encoding.lower() == "iso-8859-1":
                response.encoding = response.apparent_encoding
            html = response.text
            if len(html) < 1000:
                raise RuntimeError(f"unexpectedly short response ({len(html)} bytes)")
            if pause:
                time.sleep(pause)
            return html
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt + 1 < attempts:
                time.sleep(min(10.0, 0.8 * (2 ** attempt)))
    raise RuntimeError(f"GET failed after {attempts} attempts: {url}: {last}")


def is_central_jra_race_id(race_id: str) -> bool:
    if not re.fullmatch(r"20\d{10}", str(race_id or "")):
        return False
    venue = int(race_id[4:6])
    race_no = int(race_id[-2:])
    return 1 <= venue <= 10 and 1 <= race_no <= 12


def _observed_meeting_slots(
    source_root: Path,
    *,
    start: date,
    end: date,
) -> tuple[dict[str, dict], list[str]]:
    """Return observed meeting-day prefixes without inventing missing race numbers."""
    result_root = source_root / "data" / "race_results" / str(start.year)
    if not result_root.is_dir():
        raise FileNotFoundError(result_root)

    slots: dict[str, dict] = {}
    warnings: list[str] = []
    for path in sorted(result_root.glob("*.csv")):
        rid = path.stem
        if not is_central_jra_race_id(rid):
            continue
        try:
            result = read_csv(path)
            date_s = result_date(result, rid)
            d = pd.Timestamp(date_s).date()
        except Exception as exc:  # noqa: BLE001
            warnings.append(
                f"{rid}: cannot read date while enumerating source: "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        if not (start <= d <= end):
            continue
        prefix = rid[:10]
        slot = slots.setdefault(prefix, {"date": date_s, "observed": set()})
        if slot["date"] != date_s:
            warnings.append(
                f"{prefix}: meeting slot maps to multiple dates "
                f"{slot['date']} and {date_s}"
            )
            continue
        slot["observed"].add(rid)
    if not slots:
        raise RuntimeError("No central-JRA meeting slots were discovered from the result archive")
    return slots, warnings


def discover_expected_from_source(
    source_root: Path,
    *,
    start: date,
    end: date,
) -> tuple[dict[str, list[str]], list[str]]:
    """Offline fallback: return only race IDs actually observed in the source archive."""
    slots, warnings = _observed_meeting_slots(source_root, start=start, end=end)
    expected_by_date: dict[str, set[str]] = {}
    for slot in slots.values():
        expected_by_date.setdefault(slot["date"], set()).update(slot["observed"])
    return {d: sorted(ids) for d, ids in sorted(expected_by_date.items())}, warnings


def discover_db_race_ids(target: date) -> list[str]:
    """Exact completed race IDs exposed by the static netkeiba DB date-list page."""
    url = f"{DB_NETKEIBA_BASE}/race/list/{target:%Y%m%d}/"
    html = _request_web(url, pause=0.05, attempts=5)
    ids: set[str] = set()
    for rid in re.findall(r"/race/(20\d{10})/?", html):
        if is_central_jra_race_id(rid):
            ids.add(rid)
    return sorted(ids)


def extract_sportsnavi_meeting_race_ids(html: str, prefix: str) -> list[str]:
    """Extract exact result race IDs from SportsNavi SSR meeting-list HTML."""
    if not re.fullmatch(r"20\d{8}", prefix):
        raise ValueError(f"invalid meeting prefix: {prefix}")
    key = prefix[2:]
    found = set(re.findall(r"/keiba/race/result/(\d{10})/?", html))
    if not found:
        found = set(re.findall(r"/race/result/(\d{10})/?", html))
    if not found:
        found = set(re.findall(r"(?<!\d)(\d{10})(?!\d)", html))
    ids = {"20" + token for token in found if token.startswith(key)}
    return sorted(rid for rid in ids if is_central_jra_race_id(rid))


def discover_sportsnavi_meeting_race_ids(prefix: str) -> list[str]:
    """Exact completed race IDs for one JRA meeting-day from SportsNavi SSR HTML."""
    key = prefix[2:]
    url = f"{SPORTSNAVI_BASE}/race/list/{key}"
    html = _request_web(url, pause=0.04, attempts=5)
    return extract_sportsnavi_meeting_race_ids(html, prefix)


def _authoritative_list_covers_observed(
    authoritative: list[str],
    observed: set[str],
    prefix: str,
) -> bool:
    actual = {rid for rid in authoritative if rid.startswith(prefix)}
    return bool(actual) and observed.issubset(actual)


def discover_expected_race_ids_resilient(
    source_root: Path,
    *,
    start: date,
    end: date,
    verify_static_lists: bool = True,
) -> tuple[dict[str, list[str]], list[str]]:
    """Resolve exact historical race IDs without assuming every meeting has 12 races.

    Order of authority:
      1. SportsNavi server-rendered meeting list
      2. netkeiba static date list
      3. source-observed IDs only (offline/test fallback)

    An authoritative list is accepted only when it contains every race already observed in
    the source archive. This avoids replacing real source facts with a partially rendered page.
    """
    slots, warnings = _observed_meeting_slots(source_root, start=start, end=end)
    if not verify_static_lists:
        expected: dict[str, set[str]] = {}
        for slot in slots.values():
            expected.setdefault(slot["date"], set()).update(slot["observed"])
        return {d: sorted(ids) for d, ids in sorted(expected.items())}, warnings

    db_by_date: dict[str, list[str] | None] = {}
    expected_by_date: dict[str, set[str]] = {}
    unverified_prefixes: list[str] = []

    for prefix, slot in sorted(slots.items(), key=lambda kv: (kv[1]["date"], kv[0])):
        date_s = slot["date"]
        observed = set(slot["observed"])
        chosen: list[str] = []
        source_name = ""

        try:
            sports = discover_sportsnavi_meeting_race_ids(prefix)
            if _authoritative_list_covers_observed(sports, observed, prefix):
                chosen = [rid for rid in sports if rid.startswith(prefix)]
                source_name = "sportsnavi"
            elif sports:
                warnings.append(
                    f"{prefix}: SportsNavi list did not cover all source-observed IDs; "
                    "trying netkeiba"
                )
        except Exception as exc:  # noqa: BLE001
            warnings.append(
                f"{prefix}: SportsNavi meeting-list unavailable: "
                f"{type(exc).__name__}: {exc}"
            )

        if not chosen:
            if date_s not in db_by_date:
                try:
                    db_by_date[date_s] = discover_db_race_ids(pd.Timestamp(date_s).date())
                except Exception as exc:  # noqa: BLE001
                    db_by_date[date_s] = None
                    warnings.append(
                        f"{date_s}: netkeiba date-list unavailable: "
                        f"{type(exc).__name__}: {exc}"
                    )
            db_ids = db_by_date.get(date_s) or []
            candidates = [rid for rid in db_ids if rid.startswith(prefix)]
            if _authoritative_list_covers_observed(candidates, observed, prefix):
                chosen = candidates
                source_name = "netkeiba"

        if not chosen:
            # Never fabricate 1R..12R. Reusing observed IDs is safer than inventing races.
            chosen = sorted(observed)
            source_name = "source-observed-fallback"
            unverified_prefixes.append(prefix)
            warnings.append(
                f"{prefix}: no authoritative meeting list available; "
                f"using {len(chosen)} source-observed IDs only"
            )

        expected_by_date.setdefault(date_s, set()).update(chosen)
        warnings.append(f"{prefix}: race enumeration source={source_name}, races={len(chosen)}")

    # Date-level netkeiba lists can reveal an entire meeting prefix that is absent from
    # the source archive, so merge every independently listed central-JRA race for each
    # observed calendar date (not only prefixes already present in source files).
    observed_by_date: dict[str, set[str]] = {}
    for slot in slots.values():
        observed_by_date.setdefault(slot["date"], set()).update(slot["observed"])
    for date_s, observed_ids in sorted(observed_by_date.items()):
        if date_s not in db_by_date:
            try:
                db_by_date[date_s] = discover_db_race_ids(pd.Timestamp(date_s).date())
            except Exception as exc:  # noqa: BLE001
                db_by_date[date_s] = None
                warnings.append(
                    f"{date_s}: netkeiba date-list unavailable for whole-date cross-check: "
                    f"{type(exc).__name__}: {exc}"
                )
        db_ids = set(db_by_date.get(date_s) or [])
        if db_ids and observed_ids.issubset(db_ids):
            before = len(expected_by_date.setdefault(date_s, set()))
            expected_by_date[date_s].update(db_ids)
            added = len(expected_by_date[date_s]) - before
            if added:
                warnings.append(
                    f"{date_s}: netkeiba date-list added {added} race IDs from otherwise "
                    "missing/partial meeting prefixes"
                )

    if unverified_prefixes:
        # If the whole-date static list covered all observed races, those prefixes are no
        # longer unverified even if their SportsNavi meeting page failed.
        still_unverified = []
        for prefix in unverified_prefixes:
            date_s = slots[prefix]["date"]
            db_ids = set(db_by_date.get(date_s) or [])
            if not _authoritative_list_covers_observed(list(db_ids), set(slots[prefix]["observed"]), prefix):
                still_unverified.append(prefix)
        if still_unverified:
            warnings.append(
                "unverified meeting prefixes: " + ", ".join(still_unverified)
            )

    expected = {d: sorted(ids) for d, ids in sorted(expected_by_date.items())}
    if EXPECTED_FIRST_JRA_DATE.isoformat() not in expected:
        warnings.append(
            f"resolved dates do not include expected first JRA day "
            f"{EXPECTED_FIRST_JRA_DATE.isoformat()}"
        )
    return expected, warnings

def restore_existing_cache_facts(cache_dir: Path, source_root: Path) -> dict:
    """Reuse already-validated result/payout facts before attempting network repair."""
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        return {"restoredResults": 0, "restoredPayouts": 0, "sourceVersion": ""}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        archive = manifest.get("archive") or {}
        with tempfile.TemporaryDirectory(prefix="predictjra-old-cache-") as tmp:
            combined = Path(tmp) / "history-source.tar.gz"
            if archive.get("path"):
                src = cache_dir / archive["path"]
                if not src.is_file():
                    return {"restoredResults": 0, "restoredPayouts": 0, "sourceVersion": manifest.get("cacheVersion", "")}
                combined.write_bytes(src.read_bytes())
            else:
                parts = archive.get("parts") or []
                if not parts or any(not (cache_dir / name).is_file() for name in parts):
                    return {"restoredResults": 0, "restoredPayouts": 0, "sourceVersion": manifest.get("cacheVersion", "")}
                with combined.open("wb") as dst:
                    for name in parts:
                        dst.write((cache_dir / name).read_bytes())

            expected_sha = archive.get("sha256")
            if expected_sha and sha256_file(combined) != expected_sha:
                return {"restoredResults": 0, "restoredPayouts": 0, "sourceVersion": manifest.get("cacheVersion", "")}

            extract = Path(tmp) / "extract"
            extract.mkdir()
            with tarfile.open(combined, "r:gz") as tf:
                tf.extractall(extract)

            restored_results = 0
            restored_payouts = 0
            old_results = extract / "data" / "race_results" / "2026"
            if old_results.is_dir():
                dst_dir = source_root / "data" / "race_results" / "2026"
                dst_dir.mkdir(parents=True, exist_ok=True)
                for src in old_results.glob("*.csv"):
                    if not is_central_jra_race_id(src.stem):
                        continue
                    dst = dst_dir / src.name
                    if not dst.exists():
                        shutil.copy2(src, dst)
                        restored_results += 1
            old_payouts = extract / "data" / "race_payouts"
            if old_payouts.is_dir():
                dst_dir = source_root / "data" / "race_payouts"
                dst_dir.mkdir(parents=True, exist_ok=True)
                for src in old_payouts.glob("*.csv"):
                    if not is_central_jra_race_id(src.stem):
                        continue
                    dst = dst_dir / src.name
                    if not dst.exists():
                        shutil.copy2(src, dst)
                        restored_payouts += 1
            return {
                "restoredResults": restored_results,
                "restoredPayouts": restored_payouts,
                "sourceVersion": manifest.get("cacheVersion", ""),
            }
    except Exception as exc:  # noqa: BLE001
        return {
            "restoredResults": 0,
            "restoredPayouts": 0,
            "sourceVersion": "",
            "warning": f"{type(exc).__name__}: {exc}",
        }

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
        m = re.search(rf"/{kind}/(?:[^/?#]+/)*(\d{{4,12}})/?(?:[?#]|$)", href)
        if m:
            return m.group(1)
    return ""


def _cell_text(cells, idx: int | None) -> str:
    if idx is None or idx >= len(cells):
        return ""
    return cells[idx].get_text(" ", strip=True)


def _header_index_any(headers: list[str], *needles: str) -> int | None:
    for needle in needles:
        idx = _header_index(headers, needle)
        if idx is not None:
            return idx
    return None


def parse_netkeiba_full_result(html: str, race_id: str, target: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse final result + trifecta payout from either netkeiba DB or race pages."""
    soup = BeautifulSoup(html, "lxml")
    race_name_node = soup.select_one(".RaceName") or soup.find("h1")
    race_name = clean_str(race_name_node.get_text(" ", strip=True) if race_name_node else "")
    if not race_name:
        raise ValueError(f"{race_id}: netkeiba result missing race name")

    race_data_node = soup.select_one(".RaceData01")
    race_data = (
        race_data_node.get_text(" ", strip=True)
        if race_data_node
        else soup.get_text(" ", strip=True)[:7000]
    )
    sm = re.search(r"(芝|ダート|ダ|障)[^0-9]{0,16}(\d{3,4})m", race_data)
    if not sm:
        raise ValueError(f"{race_id}: cannot parse surface/distance from result page")
    surface = {"ダ": "ダート"}.get(sm.group(1), sm.group(1))
    distance_m = int(sm.group(2))

    direction = ""
    dm = re.search(r"(?:芝|ダート|ダ|障)[^0-9]{0,10}(右|左)", race_data)
    if not dm:
        dm = re.search(r"\([^)]*(右|左)[^)]*\)", race_data)
    if dm:
        direction = dm.group(1)

    wm = re.search(r"天候\s*[:：]\s*([^/\s]+)", race_data)
    cm = re.search(r"馬場\s*[:：]\s*([^/\s]+)", race_data)
    if not cm:
        cm = re.search(r"(?:芝|ダート|ダ|障)\s*[:：]\s*([^/\s]+)", race_data)
    weather = wm.group(1) if wm else ""
    track_condition = cm.group(1) if cm else ""

    result_table = (
        soup.select_one("table.RaceTable01")
        or soup.select_one(".ResultTableWrap table")
    )
    if result_table is None:
        for table in soup.find_all("table"):
            heads = [c.get_text(" ", strip=True) for c in table.find_all("th")]
            joined = re.sub(r"\s+", "", "|".join(heads))
            if "着順" in joined and "馬番" in joined and "馬名" in joined:
                result_table = table
                break
    if result_table is None:
        raise ValueError(f"{race_id}: netkeiba final-result table not found")

    header_row = None
    for row in result_table.find_all("tr"):
        cells = row.find_all(["th", "td"], recursive=False)
        texts = [c.get_text(" ", strip=True) for c in cells]
        joined = re.sub(r"\s+", "", "|".join(texts))
        if "着順" in joined and "馬番" in joined and "馬名" in joined:
            header_row = row
            break
    if header_row is None:
        raise ValueError(f"{race_id}: result header row not found")
    headers = [
        c.get_text(" ", strip=True)
        for c in header_row.find_all(["th", "td"], recursive=False)
    ]
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
        "win_odds": _header_index_any(headers, "単勝オッズ", "単勝"),
        "last_3f": _header_index_any(headers, "後3F", "上り"),
        "passing_order": _header_index(headers, "通過"),
        "trainer": _header_index_any(headers, "調教師", "厩舎"),
        "horse_weight": _header_index(headers, "馬体重"),
    }
    for required in (
        "finish_position", "horse_number", "horse_name", "sex_age",
        "weight_carried", "jockey", "popularity", "win_odds", "trainer",
    ):
        if idx[required] is None:
            raise ValueError(
                f"{race_id}: result table missing column {required}; headers={headers}"
            )

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
        missing = result.loc[
            result["horse_id"].astype(str).str.len() == 0, "horse_number"
        ].tolist()
        raise ValueError(f"{race_id}: horse IDs missing from result page: {missing}")

    payout_rows: list[dict] = []
    for tr in soup.find_all("tr"):
        cells = tr.find_all(["th", "td"], recursive=False)
        if len(cells) < 3:
            continue
        bet_text = re.sub(r"\s+", "", cells[0].get_text(" ", strip=True))
        if "3連単" not in bet_text and "三連単" not in bet_text:
            continue
        combo_nums = [
            int(x)
            for x in re.findall(
                r"(?<!\d)(\d{1,2})(?!\d)",
                cells[1].get_text(" ", strip=True),
            )
        ]
        amount_text = cells[2].get_text(" ", strip=True)
        amounts = [
            int(x.replace(",", ""))
            for x in re.findall(r"(?<!\d)(\d[\d,]*)(?!\d)", amount_text)
        ]
        pops: list[int] = []
        if len(cells) >= 4:
            pops = [
                int(x.replace(",", ""))
                for x in re.findall(
                    r"(?<!\d)(\d[\d,]*)(?!\d)",
                    cells[3].get_text(" ", strip=True),
                )
            ]
        if not combo_nums or len(combo_nums) % 3 != 0:
            raise ValueError(
                f"{race_id}: cannot parse trifecta combinations from netkeiba"
            )
        combos = [combo_nums[i:i + 3] for i in range(0, len(combo_nums), 3)]
        if len(amounts) != len(combos):
            raise ValueError(
                f"{race_id}: trifecta combination/payout count mismatch "
                f"{len(combos)} != {len(amounts)}"
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



def parse_netkeiba_trifecta_payout(html: str, race_id: str) -> pd.DataFrame:
    """Parse only trifecta payout from a netkeiba result page."""
    soup = BeautifulSoup(html, "lxml")
    payout_rows: list[dict] = []
    for tr in soup.find_all("tr"):
        cells = tr.find_all(["th", "td"], recursive=False)
        if len(cells) < 3:
            continue
        bet_text = re.sub(r"\s+", "", cells[0].get_text(" ", strip=True))
        if "3連単" not in bet_text and "三連単" not in bet_text:
            continue
        combo_nums = [
            int(x) for x in re.findall(
                r"(?<!\d)(\d{1,2})(?!\d)", cells[1].get_text(" ", strip=True)
            )
        ]
        amounts = [
            int(x.replace(",", "")) for x in re.findall(
                r"(?<!\d)(\d[\d,]*)(?!\d)", cells[2].get_text(" ", strip=True)
            )
        ]
        pops: list[int] = []
        if len(cells) >= 4:
            pops = [
                int(x.replace(",", "")) for x in re.findall(
                    r"(?<!\d)(\d[\d,]*)(?!\d)", cells[3].get_text(" ", strip=True)
                )
            ]
        if not combo_nums or len(combo_nums) % 3 != 0:
            continue
        combos = [combo_nums[i:i + 3] for i in range(0, len(combo_nums), 3)]
        if len(amounts) != len(combos):
            continue
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
    return payout


def _sports_entity_id(cell, kind: str) -> str:
    if cell is None:
        return ""
    for anchor in cell.find_all("a", href=True):
        href = anchor.get("href") or ""
        m = re.search(rf"/keiba/directory/{kind}/(\d{{4,12}})/?", href)
        if m:
            return m.group(1)
    return ""


def parse_sportsnavi_trifecta_payout(html: str, race_id: str) -> pd.DataFrame:
    """Parse trifecta payout from SportsNavi's server-rendered result page."""
    soup = BeautifulSoup(html, "lxml")
    rows: list[dict] = []
    for tr in soup.find_all("tr"):
        cells = tr.find_all(["th", "td"], recursive=False)
        if len(cells) < 3:
            continue
        first = re.sub(r"\s+", "", cells[0].get_text(" ", strip=True))
        if "3連単" not in first and "三連単" not in first:
            continue
        combo_text = cells[1].get_text(" ", strip=True)
        combo_nums = [int(x) for x in re.findall(r"(?<!\d)(\d{1,2})(?!\d)", combo_text)]
        amount_text = cells[2].get_text(" ", strip=True)
        amounts = [int(x.replace(",", "")) for x in re.findall(r"(\d[\d,]*)\s*円?", amount_text)]
        pop_text = cells[3].get_text(" ", strip=True) if len(cells) >= 4 else ""
        pops = [int(x.replace(",", "")) for x in re.findall(r"(?<!\d)(\d[\d,]*)(?!\d)", pop_text)]
        if not combo_nums or len(combo_nums) % 3 != 0:
            continue
        combos = [combo_nums[i:i + 3] for i in range(0, len(combo_nums), 3)]
        if len(amounts) != len(combos):
            # Some markup places multiple payout tokens in one cell separated by <br>.
            amounts = [int(x.replace(",", "")) for x in re.findall(r"\d[\d,]*", amount_text)]
        if len(amounts) != len(combos):
            continue
        if pops and len(pops) != len(combos):
            pops = []
        for i, (combo, amount) in enumerate(zip(combos, amounts)):
            rows.append({
                "race_id": race_id,
                "bet_type": "三連単",
                "combination": "-".join(str(x) for x in combo),
                "amount": amount,
                "popularity": pops[i] if pops else "",
            })
    payout = pd.DataFrame(rows)
    if payout.empty:
        # Text fallback covers minor table-markup changes without depending on CSS classes.
        text = soup.get_text(" ", strip=True)
        m = re.search(
            r"(?:3連単|三連単)\s+(\d{1,2})\s*[-→>]\s*(\d{1,2})\s*[-→>]\s*(\d{1,2})\s+([\d,]+)円(?:\s+(\d+))?",
            text,
        )
        if m:
            payout = pd.DataFrame([{
                "race_id": race_id,
                "bet_type": "三連単",
                "combination": f"{m.group(1)}-{m.group(2)}-{m.group(3)}",
                "amount": int(m.group(4).replace(",", "")),
                "popularity": int(m.group(5)) if m.group(5) else "",
            }])
    if payout.empty:
        raise ValueError(f"{race_id}: SportsNavi trifecta payout not parsed")
    return payout


def parse_sportsnavi_full_result(
    html: str,
    race_id: str,
    target: date,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse SportsNavi SSR result table without relying on fragile CSS class names."""
    soup = BeautifulSoup(html, "lxml")
    page_text = soup.get_text(" ", strip=True)

    race_name = ""
    for node in soup.find_all(["h1", "h2", "h3"]):
        txt = clean_str(node.get_text(" ", strip=True))
        if not txt or txt in {"払戻金", "競走成績", "コーナー通過順位", "通過タイム（ラップタイム）"}:
            continue
        if re.search(r"(?:サラ系|未勝利|新馬|クラス|ステークス|カップ|賞|障害)", txt):
            race_name = txt
            break
    if not race_name:
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        race_name = re.sub(r"^.*?サラ系", "サラ系", title).split(" 結果")[0].strip()
    if not race_name:
        raise ValueError(f"{race_id}: SportsNavi race name not parsed")

    sm = re.search(r"(芝|ダート|障害|障)\s*[・･]?\s*(?:外\s*)?(右|左)?\s*(\d{3,4})m", page_text)
    if not sm:
        raise ValueError(f"{race_id}: SportsNavi surface/distance not parsed")
    surface = "障" if sm.group(1) == "障害" else sm.group(1)
    direction = sm.group(2) or ""
    distance_m = int(sm.group(3))
    wm = re.search(r"天気\s*[:：]?\s*(晴|曇|小雨|雨|小雪|雪)", page_text)
    cm = re.search(r"馬場\s*[:：]?\s*(良|稍重|重|不良)", page_text)
    weather = wm.group(1) if wm else ""
    track_condition = cm.group(1) if cm else ""

    result_table = None
    for table in soup.find_all("table"):
        txt = re.sub(r"\s+", "", table.get_text(" ", strip=True))
        if "着順" in txt and "馬番" in txt and "人気" in txt and "調教師" in txt:
            result_table = table
            break
    if result_table is None:
        raise ValueError(f"{race_id}: SportsNavi result table not found")

    parsed: list[dict] = []
    for tr in result_table.find_all("tr"):
        cells = tr.find_all(["th", "td"], recursive=False)
        if len(cells) < 8:
            continue
        texts = [c.get_text(" ", strip=True) for c in cells]
        fm = re.match(r"\s*(\d+)", texts[0])
        hm = re.search(r"\d{1,2}", texts[2] if len(texts) > 2 else "")
        if not hm:
            continue
        horse_number = int(hm.group())
        if not 1 <= horse_number <= 18:
            continue
        finish_position = int(fm.group(1)) if fm else ""
        waku_m = re.search(r"[1-8]", texts[1] if len(texts) > 1 else "")
        waku = int(waku_m.group()) if waku_m else ""

        horse_cell = cells[3]
        horse_anchor = next((a for a in horse_cell.find_all("a", href=True) if "/keiba/directory/horse/" in (a.get("href") or "")), None)
        horse_name = clean_str(horse_anchor.get_text(" ", strip=True) if horse_anchor else texts[3].split()[0])
        horse_id = _sports_entity_id(horse_cell, "horse")
        hm_meta = re.search(r"(牡|牝|せん|セ|騸)(\d+)\s*/\s*([0-9]{3}(?:\([+\-]?\d+\))?)", texts[3])
        sex_age = ""
        horse_weight = ""
        if hm_meta:
            sex = {"せん": "セ", "騸": "セ"}.get(hm_meta.group(1), hm_meta.group(1))
            sex_age = f"{sex}{hm_meta.group(2)}"
            horse_weight = hm_meta.group(3)

        time_margin = texts[4] if len(texts) > 4 else ""
        tm = re.search(r"\d+:\d{2}\.\d", time_margin)
        time_value = tm.group() if tm else ""
        margin = time_margin[tm.end():].strip(" -") if tm else ""

        pass_last = texts[5] if len(texts) > 5 else ""
        l3 = re.search(r"(\d{2}\.\d)\s*$", pass_last)
        last_3f = l3.group(1) if l3 else ""
        passing = pass_last[:l3.start()].strip() if l3 else pass_last

        jockey_cell = cells[6]
        jockey_anchor = next((a for a in jockey_cell.find_all("a", href=True) if "/keiba/directory/jockey/" in (a.get("href") or "")), None)
        jockey = clean_str(jockey_anchor.get_text(" ", strip=True) if jockey_anchor else texts[6])
        jockey_id = _sports_entity_id(jockey_cell, "jockey")
        weights = re.findall(r"(?<!\d)(\d{2}(?:\.\d)?)(?!\d)", texts[6])
        weight_carried = float(weights[-1]) if weights else ""

        pop_odds = texts[7] if len(texts) > 7 else ""
        pom = re.search(r"(\d+)\s*\(\s*([\d.]+)\s*\)", pop_odds)
        popularity = int(pom.group(1)) if pom else ""
        win_odds = float(pom.group(2)) if pom else ""

        trainer_cell = cells[8] if len(cells) > 8 else cells[-1]
        trainer_anchor = next((a for a in trainer_cell.find_all("a", href=True) if "/keiba/directory/trainer/" in (a.get("href") or "")), None)
        trainer = clean_str(trainer_anchor.get_text(" ", strip=True) if trainer_anchor else trainer_cell.get_text(" ", strip=True))
        trainer_id = _sports_entity_id(trainer_cell, "trainer")

        parsed.append({
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
            "horse_name": horse_name,
            "horse_id": horse_id,
            "sex_age": sex_age,
            "weight_carried": weight_carried,
            "jockey": jockey,
            "jockey_id": jockey_id,
            "time": time_value,
            "margin": margin,
            "popularity": popularity,
            "win_odds": win_odds,
            "last_3f": last_3f,
            "passing_order": passing,
            "trainer": trainer,
            "trainer_id": trainer_id,
            "horse_weight": horse_weight,
        })

    result = pd.DataFrame(parsed)
    if len(result) < 5:
        raise ValueError(f"{race_id}: only {len(result)} SportsNavi runners parsed")
    if result["horse_number"].astype(int).duplicated().any():
        raise ValueError(f"{race_id}: duplicate SportsNavi horse numbers")
    if (result["horse_id"].astype(str).str.len() == 0).any():
        missing = result.loc[result["horse_id"].astype(str).str.len() == 0, "horse_number"].tolist()
        raise ValueError(f"{race_id}: SportsNavi horse IDs missing: {missing}")
    payout = parse_sportsnavi_trifecta_payout(html, race_id)
    validate_result_payout(result, payout, race_id)
    return result, payout


def fetch_sportsnavi_result(
    race_id: str,
    target: date,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    url = f"{SPORTSNAVI_BASE}/race/result/{race_id[2:]}"
    html = _request_web(url, pause=0.08, attempts=5)
    result, payout = parse_sportsnavi_full_result(html, race_id, target)
    return result, payout, url


def fetch_sportsnavi_payout(race_id: str) -> tuple[pd.DataFrame, str]:
    url = f"{SPORTSNAVI_BASE}/race/result/{race_id[2:]}"
    html = _request_web(url, pause=0.08, attempts=5)
    return parse_sportsnavi_trifecta_payout(html, race_id), url


def fetch_netkeiba_payout_with_fallback(race_id: str) -> tuple[pd.DataFrame, str]:
    errors: list[str] = []
    for url in [
        f"{DB_NETKEIBA_BASE}/race/{race_id}/",
        f"{RACE_NETKEIBA_BASE}/race/result.html?race_id={race_id}",
    ]:
        try:
            html = _request_web(url, pause=0.10, attempts=5)
            return parse_netkeiba_trifecta_payout(html, race_id), url
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
    raise RuntimeError(" | ".join(errors))


def fetch_multisource_full_result(
    race_id: str,
    target: date,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    errors: list[str] = []
    try:
        return fetch_sportsnavi_result(race_id, target)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"SportsNavi: {type(exc).__name__}: {exc}")
    try:
        return fetch_netkeiba_result_with_fallback(race_id, target)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"netkeiba: {type(exc).__name__}: {exc}")
    raise RuntimeError(" | ".join(errors))


def fetch_multisource_payout(race_id: str) -> tuple[pd.DataFrame, str]:
    errors: list[str] = []
    try:
        return fetch_sportsnavi_payout(race_id)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"SportsNavi: {type(exc).__name__}: {exc}")
    try:
        return fetch_netkeiba_payout_with_fallback(race_id)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"netkeiba: {type(exc).__name__}: {exc}")
    raise RuntimeError(" | ".join(errors))


def fetch_netkeiba_result_with_fallback(
    race_id: str,
    target: date,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Fetch one final result from static DB first, modern race page second."""
    urls = [
        f"{DB_NETKEIBA_BASE}/race/{race_id}/",
        f"{RACE_NETKEIBA_BASE}/race/result.html?race_id={race_id}",
    ]
    errors: list[str] = []
    for url in urls:
        try:
            html = _request_web(url, pause=0.12, attempts=5)
            result, payout = parse_netkeiba_full_result(html, race_id, target)
            return result, payout, url
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
    raise RuntimeError(" | ".join(errors))


def repair_result_archive_from_web(
    source_root: Path,
    *,
    start: date,
    end: date,
    verify_static_lists: bool = True,
) -> dict:
    """Repair exact historical result/payout gaps using independent web sources.

    Important invariants:
    - never invent race numbers;
    - never disable later repairs because an earlier race failed;
    - when the result is already valid, fetch/replace only the payout side;
    - a successful cache build must have zero unresolved races.
    """
    expected_by_date, discovery_warnings = discover_expected_race_ids_resilient(
        source_root,
        start=start,
        end=end,
        verify_static_lists=verify_static_lists,
    )
    repaired: list[dict] = []
    unresolved: list[dict] = []
    result_root = source_root / "data" / "race_results" / str(start.year)
    payout_root = source_root / "data" / "race_payouts"
    result_root.mkdir(parents=True, exist_ok=True)
    payout_root.mkdir(parents=True, exist_ok=True)

    for date_s, ids in expected_by_date.items():
        target = pd.Timestamp(date_s).date()
        for rid in ids:
            result_path = result_root / f"{rid}.csv"
            payout_path = payout_root / f"{rid}.csv"

            existing_result: pd.DataFrame | None = None
            existing_result_valid = False
            result_error = ""
            if result_path.is_file():
                try:
                    existing_result = read_csv(result_path)
                    actual_date = result_date(existing_result, rid)
                    if actual_date != date_s:
                        raise ValueError(f"source result date mismatch {actual_date} != {date_s}")
                    synthesize_card_from_result(existing_result, rid)
                    existing_result_valid = True
                except Exception as exc:  # noqa: BLE001
                    result_error = f"{type(exc).__name__}: {exc}"

            existing_payout_valid = False
            payout_error = ""
            if existing_result_valid and payout_path.is_file():
                try:
                    existing_payout = read_csv(payout_path)
                    validate_result_payout(existing_result, existing_payout, rid)
                    existing_payout_valid = True
                except Exception as exc:  # noqa: BLE001
                    payout_error = f"{type(exc).__name__}: {exc}"

            if existing_result_valid and existing_payout_valid:
                continue

            # Most archive holes are payout-only. Do not expose a valid result to an
            # unnecessary full-page parser when only the missing/invalid payout is needed.
            if existing_result_valid:
                try:
                    payout, source_url = fetch_multisource_payout(rid)
                    validate_result_payout(existing_result, payout, rid)
                    payout_was_missing = not payout_path.is_file()
                    payout.to_csv(payout_path, index=False, encoding="utf-8-sig")
                    repaired.append({
                        "date": date_s,
                        "raceId": rid,
                        "repairKind": "payout-only",
                        "resultCreated": False,
                        "payoutCreated": payout_was_missing,
                        "replacedInvalidPayout": bool(payout_error),
                        "source": source_url,
                    })
                    continue
                except Exception as exc:  # noqa: BLE001
                    unresolved.append({
                        "date": date_s,
                        "raceId": rid,
                        "side": "payout",
                        "reason": f"{type(exc).__name__}: {exc}",
                        "existingResult": "valid",
                        "existingPayout": payout_error or "missing",
                    })
                    continue

            # Missing/invalid result requires a fully validated pair. The source clone is
            # temporary; replacing a malformed archived pair never modifies the upstream repo.
            try:
                result, payout, source_url = fetch_multisource_full_result(rid, target)
                actual_date = result_date(result, rid)
                if actual_date != date_s:
                    raise ValueError(f"repaired result date mismatch {actual_date} != {date_s}")
                synthesize_card_from_result(result, rid)
                validate_result_payout(result, payout, rid)

                # If an existing payout independently agrees with the fetched result, retain it.
                retained_existing_payout = False
                if payout_path.is_file():
                    try:
                        candidate = read_csv(payout_path)
                        validate_result_payout(result, candidate, rid)
                        payout = candidate
                        retained_existing_payout = True
                    except Exception:
                        pass

                result_was_missing = not result_path.is_file()
                payout_was_missing = not payout_path.is_file()
                result.to_csv(result_path, index=False, encoding="utf-8-sig")
                if not retained_existing_payout:
                    payout.to_csv(payout_path, index=False, encoding="utf-8-sig")
                repaired.append({
                    "date": date_s,
                    "raceId": rid,
                    "repairKind": "full-pair",
                    "resultCreated": result_was_missing,
                    "payoutCreated": payout_was_missing,
                    "replacedInvalidResult": bool(result_error),
                    "retainedExistingPayout": retained_existing_payout,
                    "source": source_url,
                })
            except Exception as exc:  # noqa: BLE001
                unresolved.append({
                    "date": date_s,
                    "raceId": rid,
                    "side": "result+payout",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "existingResult": result_error or "missing",
                    "existingPayout": payout_error or ("present" if payout_path.is_file() else "missing"),
                })

    return {
        "calendarDates": sorted(expected_by_date),
        "expectedByDate": expected_by_date,
        "repaired": repaired,
        "unresolved": unresolved,
        "discoveryWarnings": discovery_warnings,
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

    # Without an independently discovered expected set, do not infer completeness from
    # race-number sequences. Historical JRA meetings can be shortened/cancelled, so the old
    # fixed 1R..12R assumption is deliberately forbidden. Only basic ID/date consistency is
    # checked in this offline fallback.
    seen: dict[str, str] = {}
    errors: list[str] = []
    for date_s, info in by_date.items():
        if date_s.startswith("__"):
            continue
        for filename in info.get("raceFiles", []):
            rid = Path(filename).stem
            if not is_central_jra_race_id(rid):
                errors.append(f"{date_s}: invalid central-JRA race id {rid}")
                continue
            previous = seen.get(rid)
            if previous and previous != date_s:
                errors.append(f"{rid}: mapped to multiple dates {previous}, {date_s}")
            seen[rid] = date_s
    if errors:
        raise RuntimeError(
            "Historical result archive has invalid IDs/date mappings: "
            + " | ".join(errors[:50])
        )

def inspect_result_backfill(
    source_root: Path,
    *,
    start: date = BACKFILL_START,
    expected_by_date: dict[str, list[str]] | None = None,
) -> dict[str, dict]:
    """Index valid central-JRA result/payout pairs and quarantine broken dates."""
    result_root = source_root / "data" / "race_results" / "2026"
    if not result_root.is_dir():
        raise FileNotFoundError(result_root)

    expected_by_date = expected_by_date or {}
    rid_to_expected_date = {
        rid: date_s
        for date_s, ids in expected_by_date.items()
        for rid in ids
    }
    by_date: dict[str, dict] = {
        date_s: {"raceFiles": [], "errors": [], "warnings": []}
        for date_s in expected_by_date
    }

    for path in sorted(result_root.glob("*.csv")):
        rid = path.stem
        if not is_central_jra_race_id(rid):
            continue
        try:
            result = read_csv(path)
            d = result_date(result, rid)
            if pd.Timestamp(d).date() < start:
                continue
        except Exception as exc:  # noqa: BLE001
            expected_date = rid_to_expected_date.get(rid)
            if expected_date:
                by_date.setdefault(
                    expected_date, {"raceFiles": [], "errors": [], "warnings": []}
                )["errors"].append({
                    "file": path.name,
                    "reason": f"cannot read/validate result date: {type(exc).__name__}: {exc}",
                })
            continue

        entry = by_date.setdefault(
            d, {"raceFiles": [], "errors": [], "warnings": []}
        )
        try:
            card = synthesize_card_from_result(result, rid)
            payout_path = source_root / "data" / "race_payouts" / f"{rid}.csv"
            if not payout_path.is_file():
                raise FileNotFoundError(f"payout missing: {payout_path}")
            payout = read_csv(payout_path)
            validate_result_payout(result, payout, rid)
            _ = card
            entry["raceFiles"].append(path.name)
        except Exception as exc:  # noqa: BLE001
            entry["errors"].append({
                "file": path.name,
                "reason": f"{type(exc).__name__}: {exc}",
            })

    for date_s, info in by_date.items():
        info["raceFiles"] = sorted(set(info["raceFiles"]))
        if expected_by_date:
            expected = set(expected_by_date.get(date_s) or [])
            actual = {Path(x).stem for x in info["raceFiles"]}
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            if missing:
                info["errors"].append({
                    "reason": f"missing expected race ids: {missing}",
                    "missingRaceIds": missing,
                })
            if unexpected:
                info["errors"].append({
                    "reason": f"unexpected race ids: {unexpected}",
                    "unexpectedRaceIds": unexpected,
                })
        info["safe"] = bool(info["raceFiles"]) and not info["errors"]
        info["reason"] = (
            "" if info["safe"]
            else f"{len(info['errors'])} result/payout/completeness errors"
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

    # Reuse already-validated cached facts first. Then enumerate expected central-JRA
    # race IDs from observed meeting slots and repair only scattered missing files.
    # Static netkeiba DB lists are a cross-check, never a hard dependency.
    cache_restore = restore_existing_cache_facts(cache_dir, source_root)
    latest = latest_2026_source_date(source_root)
    if web_discovery:
        web_repair = repair_result_archive_from_web(
            source_root,
            start=EXPECTED_FIRST_JRA_DATE,
            end=latest,
            verify_static_lists=True,
        )
    else:
        expected, warnings = discover_expected_race_ids_resilient(
            source_root,
            start=EXPECTED_FIRST_JRA_DATE,
            end=latest,
            verify_static_lists=False,
        )
        web_repair = {
            "calendarDates": sorted(expected),
            "expectedByDate": expected,
            "repaired": [],
            "unresolved": [],
            "discoveryWarnings": warnings,
        }

    unresolved = web_repair.get("unresolved") or []
    unverified = [
        w for w in (web_repair.get("discoveryWarnings") or [])
        if str(w).startswith("unverified meeting prefixes:")
    ]
    if unresolved or unverified:
        details = []
        for row in unresolved[:40]:
            details.append(
                f"{row.get('date')} {row.get('raceId')} [{row.get('side', 'unknown')}]: "
                f"{row.get('reason')}"
            )
        details.extend(unverified[:5])
        raise RuntimeError(
            "Historical backfill remains incomplete after SportsNavi/netkeiba repair; "
            "no date will be silently quarantined. Exact unresolved items: "
            + " | ".join(details)
        )

    result_backfill = inspect_result_backfill(
        source_root,
        start=EXPECTED_FIRST_JRA_DATE,
        expected_by_date=web_repair.get("expectedByDate") or {},
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

    available_dates = sorted(set(web_repair.get("expectedByDate") or {}) | set(result_backfill) | set(card_inspections))
    safe_dates: list[str] = []
    skipped_dates: list[dict] = []
    date_sources: dict[str, str] = {}
    inspections: dict[str, dict] = {}

    for date_s in available_dates:
        card_info = card_inspections.get(date_s)
        result_info = result_backfill.get(date_s)

        # Prefer the archived pre-race card when its runner set exactly matches the verified
        # result set; otherwise reconstruct a leakage-safe card from immutable result fields.
        if result_info and result_info.get("safe"):
            if card_info and card_info.get("safe"):
                card_ids = {Path(x).stem for x in card_info.get("raceFiles", [])}
                result_ids = {Path(x).stem for x in result_info.get("raceFiles", [])}
                if card_ids == result_ids:
                    safe_dates.append(date_s)
                    date_sources[date_s] = "archived-race-card"
                    inspections[date_s] = card_info
                    continue

            safe_dates.append(date_s)
            date_sources[date_s] = "result-derived-pre-race-card"
            inspections[date_s] = result_info
            continue

        reasons = []
        if result_info and not result_info.get("safe"):
            reasons.append(f"result: {result_info.get('reason')}")
        if card_info and not card_info.get("safe"):
            reasons.append(f"card: {card_info.get('reason')}")
        elif card_info and card_info.get("safe") and not result_info:
            reasons.append("result: expected result/payout set unavailable")

        skipped_dates.append({
            "date": date_s,
            "reason": "; ".join(reasons) or "no complete result/payout archive",
            "schemaErrors": (card_info or {}).get("schemaErrors", []),
            "missingArchives": (card_info or {}).get("missingArchives", []),
            "resultErrors": (result_info or {}).get("errors", []),
        })

    # v7 does not silently quarantine incomplete dates. A successful refresh means the
    # complete authoritative race set is validated, so skippedDates must be empty.
    if skipped_dates:
        details = []
        for row in skipped_dates[:30]:
            exact = row.get("resultErrors") or []
            if exact:
                for err in exact[:5]:
                    details.append(
                        f"{row['date']} {err.get('file', '')}: {err.get('reason', row['reason'])}"
                    )
            else:
                details.append(f"{row['date']}: {row['reason']}")
        raise RuntimeError(
            "Historical cache completeness check failed after all repair sources; "
            "refresh aborted instead of quarantining dates: " + " | ".join(details)
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
            "skippedDates": skipped_dates,
            "ignoredFiles": ignored_files,
            "dateRaceCounts": date_race_counts,
            "dateSources": date_sources,
            "resultDerivedDates": result_derived_dates,
            "cacheReuse": cache_restore,
            "webRepair": {
                "enabled": bool(web_discovery),
                "calendarDates": web_repair.get("calendarDates") or [],
                "repairedRaceCount": len(web_repair.get("repaired") or []),
                "repairedRaces": web_repair.get("repaired") or [],
                "unresolvedRaceCount": len(web_repair.get("unresolved") or []),
                "unresolvedRaces": web_repair.get("unresolved") or [],
                "discoveryWarnings": web_repair.get("discoveryWarnings") or [],
                "source": "source facts + SportsNavi SSR meeting/results + netkeiba static/result fallback",
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
                "raceEnumeration": "use exact race IDs from SportsNavi SSR meeting lists, cross-check with netkeiba static date lists, and never synthesize nonexistent 1R..12R slots",
                "missingArchiveRepair": "reuse prior validated cache facts first; repair only the missing side using SportsNavi first and netkeiba fallback; a valid existing result is never re-fetched merely because payout is missing",
                "resultDerivedCard": "when an archived pre-race card is absent, project only race_id/name/surface/distance/frame/horse/name/sex-age/assigned-weight/jockey/trainer/entity IDs from the final result; never project finish/time/margin/popularity/odds/bodyweight",
                "resultPayoutCrossCheck": "every starter requires final popularity; top3 result groups must exist; every trifecta payout combination must agree with the result top3",
                "stored": "sanitized program/runner facts, runner-set snapshots, final results, trifecta payouts, and required historical race result files",
                "notStored": "target-race current odds/actual popularity/bodyweight, archived model outputs, Predictjra derived indices, selections, hit judgement, or recovery-rate calculations",
                "runnerSnapshot": "use sanitized legacy prediction only when valid and runner-identical; otherwise synthesize race_id/horse_number from the sanitized/pre-race-only card",
                "completion": "successful refresh requires zero unresolved races and zero quarantined dates; if both independent web sources cannot repair an exact race, fail with that race ID instead of silently omitting the date.",
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
