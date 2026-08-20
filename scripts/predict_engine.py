#!/usr/bin/env python3
"""Predictjra live pre-race index engine.

Rules fixed for future races:
- No current odds / actual popularity / horse bodyweight/change in prediction inputs.\n- Previous-race popularity is allowed only in the separate market-popularity estimator.
- Per-run indices: 展開 / タイム / 成績.
- 近走: recent-weighted base + ceiling + repeatability.
- 今回: 展開 50% + コース 50%.
- 総合: 近走 60% + 今回 40%.
- 危険: lowest 総合 among estimated-popularity ranks 1-3.
- Selection: exclude danger, then top min(ceil(field/2), 7) by 総合.
- 本命: selected top 総合.
- 対抗: selected horse with the lowest estimated popularity (largest rank number).
- 相手: remaining selected horses.

The rich card source is netkeiba's 5-run racecard. Direct HTTP is tried first,
then Selenium. If the rich table is mostly available, missing individual horses are
kept with neutral history rather than silently dropped.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable

from bs4 import BeautifulSoup

TRACKS = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
}

NETKEIBA_BASE = "https://race.netkeiba.com"

POPULARITY_PROFILES = {
    "open": {"recent": .32, "class": .22, "consistency": .06, "upward": .08, "age": .06},
    "class": {"recent": .42, "class": .13, "consistency": .11, "upward": .10, "age": .04},
    "maiden": {"recent": .52, "class": .04, "consistency": .13, "upward": .09, "age": .02},
    "jump": {"recent": .50, "class": .17, "consistency": .10, "upward": .03},
    "debut": {"class": .35, "age": .10},
}

MODEL_VERSION = "predictjra-live-index-v2-market-memory"


def clean(value) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def clamp(value: float, lo: float = 45, hi: float = 98) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return (lo + hi) / 2
    if not math.isfinite(x):
        return (lo + hi) / 2
    return max(lo, min(hi, x))


def parse_float(value, default=math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_surface(raw: str) -> str:
    s = clean(raw)
    if "障" in s:
        return "障害"
    if "芝" in s:
        return "芝"
    if "ダ" in s:
        return "ダート"
    return ""


def parse_distance(raw: str) -> float:
    s = clean(raw).replace(",", "")
    m = re.search(r"(\d{3,4})m?", s)
    if not m:
        return math.nan
    return float(m.group(1))


def parse_date_text(raw: str) -> str:
    s = str(raw or "")
    m = re.search(r"(20\d{2})[./](\d{1,2})[./](\d{1,2})", s)
    if not m:
        return ""
    return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"


def recency_weighted(values: list[float]) -> float:
    vals = [float(x) for x in values[:5]]
    if not vals:
        return 72.0
    base = [0.34, 0.25, 0.18, 0.13, 0.10][:len(vals)]
    total = sum(base)
    return sum(v * w for v, w in zip(vals, base)) / total


def race_profile(race_name: str, surface: str) -> str:
    name = clean(race_name)
    if surface == "障害" or "障害" in name or "JS" in name:
        return "jump"
    if "新馬" in name:
        return "debut"
    if "未勝利" in name:
        return "maiden"
    if any(token in name for token in ("G1", "G2", "G3", "Ｇ１", "Ｇ２", "Ｇ３", "OP", "オープン", "リステッド")):
        return "open"
    return "class"


def parse_age(text: str) -> int | None:
    m = re.search(r"[牡牝セ騙](\d{1,2})", str(text or ""))
    return int(m.group(1)) if m else None


def normalize_entity_id(value) -> str:
    digits = re.sub(r"\D", "", str(value or ""))
    return digits.zfill(5) if digits else ""


def clamp01(value: float, default: float = 0.5) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(x):
        return default
    return max(0.0, min(1.0, x))


def market_strength(popularity: int | None, field: int | None) -> float:
    if not isinstance(popularity, int) or not isinstance(field, int) or field <= 1:
        return 0.5
    return clamp01(1 - (popularity - 1) / (field - 1))


def market_recency(values: list[float], limit: int = 5) -> float:
    vals = [clamp01(v) for v in values[:limit]]
    if not vals:
        return 0.5
    weights = [0.40, 0.25, 0.16, 0.11, 0.08][:len(vals)]
    sw = sum(weights)
    return clamp01(sum(v * w for v, w in zip(vals, weights)) / sw)


def parse_class_level(text: str) -> int:
    s = clean(text).upper()
    # Longer roman-numeral grade labels must be checked first.
    if any(x in s for x in ("GIII", "G3", "ＧⅢ", "Ｇ３")):
        return 5
    if any(x in s for x in ("GII", "G2", "ＧⅡ", "Ｇ２")):
        return 6
    if any(x in s for x in ("GI", "G1", "ＧⅠ", "Ｇ１")):
        return 7
    if "リステッド" in s or "(L)" in s or "（L）" in s or "OP" in s or "オープン" in s:
        return 4
    if "3勝" in s or "３勝" in s:
        return 3
    if "2勝" in s or "２勝" in s:
        return 2
    if "1勝" in s or "１勝" in s:
        return 1
    if "未勝利" in s or "新馬" in s:
        return 0
    return 3


def load_popularity_model() -> dict:
    data_dir = Path(__file__).resolve().parents[1] / "data"
    for filename in ("popularity_model.json", "popularity_model_20260815_16.json"):
        path = data_dir / filename
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "coefficients" in payload and "features" in payload:
            return payload
    return {}


def entity_prior(model: dict, kind: str, entity_id: str, entity_name: str) -> float:
    priors = model.get("entityPriors") or {}
    by_id = priors.get(kind) or {}
    by_name = priors.get(f"{kind}Name") or {}
    if entity_id and entity_id in by_id:
        return clamp01(by_id[entity_id])
    key = clean(entity_name)
    if key and key in by_name:
        return clamp01(by_name[key])
    return 0.5


def parse_history_cell(text: str) -> dict | None:
    raw = re.sub(r"\s+", " ", str(text or "")).strip()
    date_s = parse_date_text(raw)
    if not date_s:
        return None

    date_match = re.search(r"20\d{2}[./]\d{1,2}[./]\d{1,2}", raw)
    tail = raw[date_match.end():].strip() if date_match else raw
    venue_match = re.match(r"([^\s\d]{1,8})", tail)
    venue = venue_match.group(1) if venue_match else ""

    fd = re.search(r"(?<!\d)(\d{1,2})\s*(?:着)?\s+(\d{1,2})頭", raw)
    if not fd:
        fd = re.search(r"(?<!\d)(\d{1,2})\s*(?:着)?\s*(\d{1,2})頭", raw)
    finish = int(fd.group(1)) if fd else None
    field = int(fd.group(2)) if fd else None

    popularity_match = re.search(r"(?<!\d)(\d{1,2})\s*(?:人|人気)", raw)
    popularity = int(popularity_match.group(1)) if popularity_match else None
    class_level = parse_class_level(raw)

    sd = re.search(r"(芝|ダ|障(?:害)?)\s*(\d{3,4})", raw)
    surface = normalize_surface(sd.group(1)) if sd else ""
    distance = float(sd.group(2)) if sd else math.nan

    tm = re.search(r"(?<!\d)(\d+:\d{2}\.\d)(?!\d)", raw)
    time_text = tm.group(1) if tm else ""

    # Last 3F is normally the only 30.x-40.x standalone decimal in the cell.
    decimals = [float(x) for x in re.findall(r"(?<![\d:])(\d{2}\.\d)(?!\d)", raw)]
    last3f = next((x for x in reversed(decimals) if 28.0 <= x <= 50.0), math.nan)

    positions = []
    pos_candidates = re.findall(r"(?<!\d)(\d{1,2}(?:-\d{1,2}){1,4})(?!\d)", raw)
    if pos_candidates:
        positions = [int(x) for x in pos_candidates[-1].split("-")]

    # netkeiba's final "(0.8)" / "(-0.2)" is the winner/runner-up time difference.
    parens = re.findall(r"\((-?\d+\.\d+)\)", raw)
    margin = math.nan
    if parens:
        candidates = [float(x) for x in parens if abs(float(x)) <= 10]
        if candidates:
            margin = candidates[-1]

    carried_match = re.search(r"(?<!\d)([4-6]\d(?:\.\d)?)\s+\d{3}kg", raw)
    carried_weight = float(carried_match.group(1)) if carried_match else math.nan

    return {
        "date": date_s,
        "venue": venue,
        "finish": finish,
        "field": field,
        "popularity": popularity,
        "marketStrength": market_strength(popularity, field),
        "classLevel": class_level,
        "carriedWeight": carried_weight,
        "surface": surface,
        "distance": distance,
        "time": time_text,
        "last3f": last3f,
        "positions": positions,
        "margin": margin,
        "raw": raw,
    }


def _horse_number(row, cells) -> int | None:
    node = (
        row.select_one("td.Umaban")
        or row.select_one(".Umaban")
        or row.select_one("td[class*='Umaban']")
    )
    if node:
        m = re.search(r"(?<!\d)(\d{1,2})(?!\d)", clean(node.get_text(" ", strip=True)))
        if m and 1 <= int(m.group(1)) <= 18:
            return int(m.group(1))

    # Shutuba_Past5_Table has historically used column 1 for horse number.
    if len(cells) >= 2:
        text = clean(cells[1].get_text(" ", strip=True))
        if re.fullmatch(r"\d{1,2}", text) and 1 <= int(text) <= 18:
            return int(text)

    for cell in cells[:4]:
        text = clean(cell.get_text(" ", strip=True))
        if re.fullmatch(r"\d{1,2}", text) and 1 <= int(text) <= 18:
            return int(text)
    return None


def _frame_number(row, cells) -> int | None:
    node = row.select_one("td.Waku") or row.select_one(".Waku")
    if node:
        m = re.search(r"[1-8]", clean(node.get_text(" ", strip=True)))
        if m:
            return int(m.group(0))
    if cells:
        t = clean(cells[0].get_text(" ", strip=True))
        if re.fullmatch(r"[1-8]", t):
            return int(t)
    return None


def _horse_name_and_id(row) -> tuple[str, str]:
    node = row.select_one(".HorseName a") or row.select_one("a[href*='/horse/']")
    if not node:
        return "", ""
    name = clean(node.get_text(" ", strip=True))
    href = node.get("href", "")
    m = re.search(r"/horse/(?:result/)?(\d+)", href)
    return name, (m.group(1) if m else "")


def _linked_entity(row, kind: str) -> tuple[str, str]:
    node = row.select_one(f"a[href*='/{kind}/']")
    if not node:
        return "", ""
    name = clean(node.get_text(" ", strip=True))
    href = node.get("href", "")
    ids = re.findall(r"\d{4,5}", href)
    return name, normalize_entity_id(ids[-1] if ids else "")


def _current_carried_weight(row) -> float:
    jockey_node = row.select_one("a[href*='/jockey/']")
    if not jockey_node:
        return math.nan
    cell = jockey_node.find_parent(["td", "th"])
    text = cell.get_text(" ", strip=True) if cell else row.get_text(" ", strip=True)
    values = [
        float(x) for x in re.findall(r"(?<!\d)([4-6]\d(?:\.\d)?)(?!\d)", text)
    ]
    return values[-1] if values else math.nan


def parse_rich_card(html: str, race_id: str, base_entries: list[dict] | None = None) -> dict:
    soup = BeautifulSoup(html, "lxml")
    page_text = soup.get_text(" ", strip=True)

    race_name_node = soup.select_one("h1")
    race_name = clean(race_name_node.get_text(" ", strip=True)) if race_name_node else ""

    condition_match = re.search(r"(芝|ダ|障(?:害)?)\s*(\d{3,4})m", page_text)
    surface = normalize_surface(condition_match.group(1)) if condition_match else ""
    distance = float(condition_match.group(2)) if condition_match else math.nan

    table = soup.select_one("table.Shutuba_Past5_Table")
    if not table:
        for candidate in soup.find_all("table"):
            txt = clean(candidate.get_text(" ", strip=True))
            if "馬番" in txt and ("前走" in txt or "5走" in txt):
                table = candidate
                break
    if not table:
        raise ValueError(f"{race_id}: 5-run racecard table not found")

    parsed: dict[int, dict] = {}
    rows = table.select("tbody tr") or table.find_all("tr")
    for row in rows:
        cells = row.find_all(["td", "th"], recursive=False)
        if len(cells) < 2:
            continue
        no = _horse_number(row, cells)
        if no is None:
            continue

        frame = _frame_number(row, cells)
        name, horse_id = _horse_name_and_id(row)
        jockey_name, jockey_id = _linked_entity(row, "jockey")
        trainer_name, trainer_id = _linked_entity(row, "trainer")
        current_carried_weight = _current_carried_weight(row)
        row_text = row.get_text(" ", strip=True)
        age = parse_age(row_text)

        # Do not parse current odds, current popularity, current horse weight/change.
        histories = []
        for cell in cells:
            cell_text = cell.get_text(" ", strip=True)
            if not re.search(r"20\d{2}[./]\d{1,2}[./]\d{1,2}", cell_text):
                continue
            history = parse_history_cell(cell_text)
            if history:
                histories.append(history)
        histories = histories[:5]

        parsed[no] = {
            "no": no,
            "frame": frame,
            "name": name,
            "horseId": horse_id,
            "age": age,
            "jockeyName": jockey_name,
            "jockeyId": jockey_id,
            "trainerName": trainer_name,
            "trainerId": trainer_id,
            "currentCarriedWeight": current_carried_weight,
            "histories": histories,
        }

    base = {int(e["horse"]): e for e in (base_entries or [])}
    warnings: list[str] = []
    if base:
        coverage = len(set(parsed) & set(base)) / max(1, len(base))
        if coverage < 0.80:
            raise ValueError(
                f"{race_id}: rich-card horse coverage too low "
                f"({len(set(parsed)&set(base))}/{len(base)})"
            )
        for no, entry in base.items():
            if no not in parsed:
                warnings.append(f"horse {no}: rich history missing; neutral history used")
                parsed[no] = {
                    "no": no,
                    "frame": entry.get("frame"),
                    "name": entry.get("name", ""),
                    "horseId": "",
                    "age": None,
                    "jockeyName": "",
                    "jockeyId": "",
                    "trainerName": "",
                    "trainerId": "",
                    "currentCarriedWeight": math.nan,
                    "histories": [],
                }
            else:
                if not parsed[no].get("frame") and entry.get("frame"):
                    parsed[no]["frame"] = entry["frame"]
                if not parsed[no].get("name") and entry.get("name"):
                    parsed[no]["name"] = entry["name"]

    if len(parsed) < 5:
        raise ValueError(f"{race_id}: only {len(parsed)} horses parsed from rich card")

    class_hint = ""
    compact_page = clean(page_text[:5000])
    grade_match = re.search(
        r"(G[123]|Ｇ[１２３]|GⅠ|GⅡ|GⅢ|オープン|リステッド|\(L\)|（L）)",
        compact_page,
        re.IGNORECASE,
    )
    if grade_match:
        class_hint = grade_match.group(1)

    current_class_level = parse_class_level(
        f"{race_name} {class_hint} {page_text[:3000]}"
    )

    venue = TRACKS.get(race_id[4:6], race_id[4:6])
    return {
        "raceId": race_id,
        "raceName": race_name,
        "classHint": class_hint,
        "classLevel": current_class_level,
        "venue": venue,
        "raceNo": int(race_id[-2:]),
        "surface": surface,
        "distanceM": distance if math.isfinite(distance) else None,
        "entries": [parsed[k] for k in sorted(parsed)],
        "qualityWarnings": warnings,
    }


def fetch_rich_card(
    race_id: str,
    base_entries: list[dict],
    request_html: Callable,
    selenium_html: Callable,
) -> tuple[dict, str]:
    url = f"{NETKEIBA_BASE}/race/shutuba_past.html?race_id={race_id}&rf=shutuba_submenu"
    errors: list[str] = []

    for attempt in range(2):
        try:
            html = request_html(url, pause=0.5, referer="https://race.netkeiba.com/")
            card = parse_rich_card(html, race_id, base_entries)
            return card, "netkeiba-past5"
        except Exception as exc:  # noqa: BLE001
            errors.append(f"http{attempt+1}: {exc}")

    try:
        html = selenium_html(url, wait_seconds=3.0)
        card = parse_rich_card(html, race_id, base_entries)
        return card, "netkeiba-past5-selenium"
    except Exception as exc:  # noqa: BLE001
        errors.append(f"selenium: {exc}")

    raise RuntimeError(f"{race_id}: rich-card fetch failed; " + " | ".join(errors))


def run_indices(history: dict) -> dict:
    finish = history.get("finish")
    field = history.get("field")
    if not isinstance(finish, int) or not isinstance(field, int) or field <= 1:
        finish_strength = 0.5
    else:
        finish_strength = max(0.0, min(1.0, 1 - (finish - 1) / (field - 1)))

    result_index = clamp(48 + 50 * finish_strength)

    margin = parse_float(history.get("margin"))
    if math.isfinite(margin):
        gap_score = clamp(96 - 6.0 * max(0.0, margin))
    else:
        gap_score = clamp(55 + 40 * finish_strength)
    last_score = clamp(55 + 40 * finish_strength)
    time_index = clamp(0.75 * gap_score + 0.25 * last_score)

    positions = history.get("positions") or []
    if isinstance(field, int) and field > 1 and positions:
        first = positions[0]
        last = positions[-1]
        front = max(0.0, min(1.0, 1 - (first - 1) / max(field - 1, 1)))
        improvement = (last - finish) / max(field, 1) if isinstance(finish, int) else 0.0
        pace_index = clamp(64 + 18 * finish_strength + 10 * max(-1, min(1, improvement)) + 4)
    else:
        front = math.nan
        pace_index = clamp(62 + 26 * finish_strength + 3.5)

    composite = 0.25 * pace_index + 0.35 * time_index + 0.40 * result_index
    return {
        **history,
        "paceIndex": pace_index,
        "timeIndex": time_index,
        "resultIndex": result_index,
        "frontStrength": front,
        "composite": composite,
    }


def _course_score(runs: list[dict], cap: float) -> float | None:
    vals = [r["composite"] for r in runs[:5] if math.isfinite(float(r.get("composite", math.nan)))]
    if not vals:
        return None
    value = 0.72 * recency_weighted(vals) + 0.28 * max(vals)
    return min(cap, value)


def course_index(
    runs: list[dict],
    venue: str,
    surface: str,
    distance: float | None,
    proxy: float,
) -> float:
    valid_distance = distance is not None and math.isfinite(float(distance)) and float(distance) > 0
    exact = [
        r for r in runs
        if r.get("venue") == venue
        and (not surface or r.get("surface") == surface)
        and (not valid_distance or (
            math.isfinite(float(r.get("distance", math.nan)))
            and abs(float(r["distance"]) - float(distance)) <= 100
        ))
    ]
    exact_score = _course_score(exact, 98)
    if exact_score is not None:
        return clamp(0.84 * exact_score + 0.16 * proxy)

    venue_runs = [
        r for r in runs
        if r.get("venue") == venue and (not surface or r.get("surface") == surface)
    ]
    venue_score = _course_score(venue_runs, 86)
    if venue_score is not None:
        return clamp(0.80 * venue_score + 0.20 * proxy)

    if valid_distance:
        dist_runs = [
            r for r in runs
            if (not surface or r.get("surface") == surface)
            and math.isfinite(float(r.get("distance", math.nan)))
            and abs(float(r["distance"]) - float(distance)) <= 200
        ]
        dist_score = _course_score(dist_runs, 82)
        if dist_score is not None:
            return clamp(0.76 * dist_score + 0.24 * proxy)

    return clamp(min(78, 0.88 * proxy + 0.12 * 72))


def build_prediction(card: dict) -> dict:
    entries = card["entries"]
    horse_count = len(entries)
    if horse_count < 5:
        raise ValueError(f"{card['raceId']}: need at least 5 horses, got {horse_count}")

    contexts = {}
    front_type_count = 0

    for entry in entries:
        runs = [run_indices(h) for h in entry.get("histories", [])[:5]]
        composites = [r["composite"] for r in runs]
        fronts = [r["frontStrength"] for r in runs if math.isfinite(float(r.get("frontStrength", math.nan)))]
        front_ratio = mean(fronts) if fronts else math.nan
        if math.isfinite(front_ratio) and front_ratio >= 0.58:
            front_type_count += 1

        if composites:
            base = recency_weighted(composites)
            ceiling = max(composites)
            consistency = clamp(96 - 1.8 * pstdev(composites), 55, 96)
            hist_recent = 0.70 * base + 0.20 * ceiling + 0.10 * consistency
            history_weight = min(len(composites) / 5.0, 1.0) * 0.84
            recent = clamp(history_weight * hist_recent + (1 - history_weight) * 72)
        else:
            ceiling = 72.0
            consistency = 72.0
            recent = 72.0

        if len(composites) >= 3:
            recent_part = mean(composites[:2])
            older_part = mean(composites[-2:])
            upward = clamp(72 + 1.2 * (recent_part - older_part))
        else:
            upward = 72.0

        contexts[entry["no"]] = {
            "entry": entry,
            "runs": runs,
            "frontRatio": front_ratio,
            "recent": recent,
            "ceiling": ceiling,
            "consistency": consistency,
            "upward": upward,
        }

    pace_regime = "fast" if front_type_count >= 3 else "slow" if front_type_count <= 1 else "medium"

    detail_horses = []
    totals: dict[int, float] = {}

    for no in sorted(contexts):
        c = contexts[no]
        proxy = c["recent"]
        fr = c["frontRatio"]

        if math.isfinite(fr):
            if pace_regime == "fast":
                style = 70 + 20 * (1 - fr)
            elif pace_regime == "slow":
                style = 70 + 20 * fr
            else:
                style = 76 + 10 * (1 - abs(fr - 0.5) * 2)
            pace = clamp(0.78 * style + 0.22 * proxy)
        else:
            pace = clamp(0.78 * proxy + 0.22 * 75)

        course = course_index(
            c["runs"], card["venue"], card.get("surface", ""),
            card.get("distanceM"), proxy,
        )
        today = clamp(0.50 * pace + 0.50 * course)
        total = clamp(0.60 * c["recent"] + 0.40 * today)
        totals[no] = total

        recent_strings = [
            f"{int(round(r['paceIndex']))}/{int(round(r['timeIndex']))}/{int(round(r['resultIndex']))}"
            for r in c["runs"][:5]
        ]
        while len(recent_strings) < 5:
            recent_strings.append("評価外")

        detail_horses.append({
            "no": no,
            "name": c["entry"].get("name", ""),
            "recent": recent_strings,
            "recentIndex": int(round(c["recent"])),
            "pace": int(round(pace)),
            "course": int(round(course)),
            "today": int(round(today)),
            "total": int(round(total)),
            "_ceiling": c["ceiling"],
            "_consistency": c["consistency"],
            "_upward": c["upward"],
            "_age": c["entry"].get("age"),
        })

    ordered = sorted(detail_horses, key=lambda h: (-totals[h["no"]], -h["recentIndex"], h["no"]))
    rank_map = {h["no"]: i + 1 for i, h in enumerate(ordered)}
    for h in detail_horses:
        h["rank"] = rank_map[h["no"]]

    # Market-popularity model v2.
    # Unlike the ability index, this deliberately remembers how the public priced
    # the horse in previous races, plus jockey/trainer market tendency.
    popularity_model = load_popularity_model()
    coeff = popularity_model.get("coefficients") or {}
    model_features = set(popularity_model.get("features") or [])
    current_class_level = int(card.get("classLevel", 3))

    total_order_for_market = sorted(
        detail_horses,
        key=lambda h: (-totals[h["no"]], -h["recentIndex"], h["no"]),
    )
    total_rank_strength = {
        h["no"]: 1.0 - i / max(horse_count - 1, 1)
        for i, h in enumerate(total_order_for_market)
    }
    recent_order_for_market = sorted(
        detail_horses,
        key=lambda h: (-h["recentIndex"], -totals[h["no"]], h["no"]),
    )
    recent_rank_strength = {
        h["no"]: 1.0 - i / max(horse_count - 1, 1)
        for i, h in enumerate(recent_order_for_market)
    }

    for h in detail_horses:
        entry = contexts[h["no"]]["entry"]
        runs = contexts[h["no"]]["runs"]
        market_values = [
            market_strength(r.get("popularity"), r.get("field"))
            for r in runs
            if isinstance(r.get("popularity"), int) and isinstance(r.get("field"), int)
        ]
        last_market = market_values[0] if market_values else 0.5
        recent3_market = market_recency(market_values, 3)
        recent5_market = market_recency(market_values, 5)

        last = runs[0] if runs else {}
        last_finish = (
            1.0 - (last["finish"] - 1) / max(last["field"] - 1, 1)
            if isinstance(last.get("finish"), int)
            and isinstance(last.get("field"), int)
            and last["field"] > 1
            else 0.5
        )
        surprise_strength = clamp01(0.5 + (last_finish - last_market) / 2.0)

        last_pop = last.get("popularity")
        last_field = last.get("field")
        last_lowpop_win = float(
            isinstance(last.get("finish"), int)
            and last["finish"] == 1
            and isinstance(last_pop, int)
            and isinstance(last_field, int)
            and last_pop >= max(6, math.ceil(last_field / 2))
        )

        current_weight = parse_float(entry.get("currentCarriedWeight"))
        last_weight = parse_float(last.get("carriedWeight"))
        if math.isfinite(current_weight) and math.isfinite(last_weight):
            weight_delta = current_weight - last_weight
            carried_change_strength = clamp01(0.5 - weight_delta / 12.0)
            handicap_rebound_risk = clamp01(
                last_lowpop_win * max(weight_delta - 1.0, 0.0) / 5.0,
                default=0.0,
            )
        else:
            weight_delta = 0.0
            carried_change_strength = 0.5
            handicap_rebound_risk = 0.0

        jockey_market = entity_prior(
            popularity_model, "jockey",
            entry.get("jockeyId", ""), entry.get("jockeyName", "")
        )
        trainer_market = entity_prior(
            popularity_model, "trainer",
            entry.get("trainerId", ""), entry.get("trainerName", "")
        )
        age = entry.get("age")
        age_strength = clamp01((10.0 - float(age or 5)) / 8.0)

        factors = {
            "total_rank_strength": clamp01(total_rank_strength[h["no"]]),
            "recent_rank_strength": clamp01(recent_rank_strength[h["no"]]),
            "last_market_strength": clamp01(last_market),
            "recent3_market_strength": clamp01(recent3_market),
            "recent5_market_strength": clamp01(recent5_market),
            "last_finish_strength": clamp01(last_finish),
            "surprise_strength": clamp01(surprise_strength),
            "jockey_market_strength": clamp01(jockey_market),
            "trainer_market_strength": clamp01(trainer_market),
            "age_strength": clamp01(age_strength),
            "carried_change_strength": clamp01(carried_change_strength),
            "last_lowpop_win": last_lowpop_win,
            "handicap_rebound_risk": clamp01(handicap_rebound_risk, default=0.0),
        }

        if (
            popularity_model.get("version", "").endswith("market-memory")
            and model_features.issubset(factors)
            and model_features
        ):
            market_score = float(coeff.get("intercept", 0.0))
            for key in popularity_model["features"]:
                market_score += float(coeff.get(key, 0.0)) * float(factors[key])
        else:
            # Conservative fallback if the calibrated v2 file has not been generated yet.
            market_score = (
                0.36 * factors["recent5_market_strength"]
                + 0.14 * factors["last_market_strength"]
                + 0.16 * factors["total_rank_strength"]
                + 0.08 * factors["recent_rank_strength"]
                + 0.08 * factors["jockey_market_strength"]
                + 0.06 * factors["trainer_market_strength"]
                + 0.06 * factors["last_finish_strength"]
                + 0.06 * factors["age_strength"]
            )

        # Class/handicap context is directly visible in the pre-race five-run card.
        last_class_level = int(last.get("classLevel", current_class_level)) if last else current_class_level
        max_recent_class = max(
            [int(r.get("classLevel", 0)) for r in runs] or [current_class_level]
        )
        class_gap = max(current_class_level - max_recent_class, 0)
        class_readiness = clamp01(1.0 - class_gap / 4.0)

        # A low-popularity handicap win followed by a large assigned-weight rise is
        # a classic market-overreaction trap for an ability-only model.
        context_adjustment = (
            0.035 * (class_readiness - 0.5)
            - 0.090 * factors["handicap_rebound_risk"]
        )
        h["_popScore"] = market_score + context_adjustment
        h["popularityFactors"] = {
            key: round(float(value) * 100, 1)
            for key, value in factors.items()
        }
        h["popularityContext"] = {
            "classLevel": current_class_level,
            "lastClassLevel": last_class_level,
            "maxRecentClassLevel": max_recent_class,
            "assignedWeightDelta": round(weight_delta, 1),
            "model": (
                popularity_model.get("version")
                or "fallback-market-memory"
            ),
        }

    pop_order = sorted(
        detail_horses,
        key=lambda h: (-h["_popScore"], -h["recentIndex"], -h["total"], h["no"])
    )
    expected_popularity = {h["no"]: i + 1 for i, h in enumerate(pop_order)}
    for h in detail_horses:
        h["expectedPopularity"] = expected_popularity[h["no"]]

    danger = sorted(
        [h for h in detail_horses if h["expectedPopularity"] <= 3],
        key=lambda h: (h["total"], h["recentIndex"], -h["expectedPopularity"], -h["no"]),
    )[0]

    target_count = min((horse_count + 1) // 2, 7)
    selected = sorted(
        [h for h in detail_horses if h["no"] != danger["no"]],
        key=lambda h: (-totals[h["no"]], -h["recentIndex"], h["no"]),
    )[:target_count]

    main = selected[0]
    second = sorted(
        [h for h in selected if h["no"] != main["no"]],
        key=lambda h: (-h["expectedPopularity"], -totals[h["no"]], h["no"]),
    )[0]
    opponents = [
        h["no"] for h in sorted(
            [h for h in selected if h["no"] not in (main["no"], second["no"])],
            key=lambda h: (-totals[h["no"]], -h["recentIndex"], h["no"]),
        )
    ]

    for h in detail_horses:
        h["excluded"] = h["no"] == danger["no"]
        for private_key in ("_ceiling", "_consistency", "_upward", "_age", "_popScore"):
            h.pop(private_key, None)

    prediction = {
        "axes": [main["no"], second["no"]],
        "opponents": opponents,
        "excluded": [danger["no"]],
    }

    index_detail = {
        "title": f"{card['venue']}{card['raceNo']}R"
                 + (f" {card['raceName']}" if card.get("raceName") else ""),
        "horseCount": horse_count,
        "paceRegime": pace_regime,
        "raceConditions": {
            "surface": card.get("surface") or None,
            "distanceM": int(card["distanceM"]) if card.get("distanceM") else None,
        },
        "qualityWarnings": list(card.get("qualityWarnings", [])),
        "horses": detail_horses,
        "prediction": prediction,
    }

    return {
        "prediction": prediction,
        "danger": [danger["no"]],
        "indexDetail": index_detail,
        "estimatedPopularity": {str(k): int(v) for k, v in expected_popularity.items()},
        "totalIndex": {str(h["no"]): int(h["total"]) for h in detail_horses},
        "modelMeta": {
            "version": MODEL_VERSION,
            "estimatedPopularity": {str(k): int(v) for k, v in expected_popularity.items()},
            "totalIndex": {str(h["no"]): int(h["total"]) for h in detail_horses},
            "indexDetail": index_detail,
            "performanceSource": (
                "netkeiba pre-race 5-run card; per-run 展開/タイム/成績; "
                "近走60% + 今回40%; 今回=展開50%+コース50%"
            ),
            "popularityMethod": (
                "market-memory v2: previous-race popularity, recent market memory, "
                "ability rank, jockey/trainer market priors, age and assigned-weight change; "
                "current odds/actual popularity/bodyweight are not used"
            ),
            "selectionRule": (
                "danger=lowest total among estimated-popularity top3; "
                "exclude danger; select top min(ceil(field/2),7) total; "
                "main=top total; second=lowest estimated popularity among selected"
            ),
            "prohibitedInputs": [
                "current odds", "current actual popularity",
                "horse bodyweight", "horse bodyweight change",
            ],
        },
    }
