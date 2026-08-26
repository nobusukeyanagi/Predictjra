#!/usr/bin/env python3
"""Predictjra live pre-race index engine.

Live predictions always import the explicitly applied production logic snapshot.

Rules fixed for future races:
- No current odds / actual popularity / horse bodyweight/change in prediction inputs.
- Previous-race popularity is allowed only in the separate market-popularity estimator.
- Per-run indices: 展開 / タイム / 成績.
- 近走: recent-weighted base + ceiling + repeatability.
- 今回: 展開 50% + コース 50%.
- 総合: 近走 60% + 今回 40%.
- 危険: lowest 総合 among estimated-popularity ranks 1-3.
- Selection: exclude danger, then top min(ceil(field/2), 7) by 総合.
- 本命: highest 単EV among ability-safe selected horses (all predicted races get one main).
- 対抗: selected horse with the lowest estimated popularity (largest rank number), excluding 本命.
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
from typing import Callable

import numpy as np
from joblib import load as joblib_load
from bs4 import BeautifulSoup

from prediction_logic_production import (
    FEATURE_COLS,
    MODEL_VERSION,
    SELECTION_RULE_TEXT,
    build_index_core,
    build_market_profile,
    build_popularity_feature_row,
    fallback_top3_score,
    clamp,
    clamp01,
    expected_popularity_from_scores,
    market_recency,
    market_score_from_model,
    market_strength,
    parse_class_level,
    rank_strengths,
    select_prediction,
)

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



def clean(value) -> str:
    return re.sub(r"\s+", "", str(value or ""))



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






def load_popularity_model() -> dict:
    data_dir = Path(__file__).resolve().parents[1] / "data"
    for filename in ("popularity_model.json", "popularity_model_20260815_16.json"):
        path = data_dir / filename
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "features" in payload or "entityPriors" in payload:
            return payload
    return {}


_POPULARITY_CLASSIFIER = None
_POPULARITY_CLASSIFIER_LOADED = False


def load_popularity_classifier():
    global _POPULARITY_CLASSIFIER, _POPULARITY_CLASSIFIER_LOADED
    if _POPULARITY_CLASSIFIER_LOADED:
        return _POPULARITY_CLASSIFIER
    _POPULARITY_CLASSIFIER_LOADED = True
    path = Path(__file__).resolve().parents[1] / "data" / "popularity_model_v54.joblib"
    try:
        model = joblib_load(path)
        if not hasattr(model, "predict_proba"):
            raise TypeError("classifier does not expose predict_proba")
        _POPULARITY_CLASSIFIER = model
    except Exception as exc:
        print(f"POPULARITY_MODEL_FALLBACK: {exc}")
        _POPULARITY_CLASSIFIER = None
    return _POPULARITY_CLASSIFIER


def load_time_baselines() -> dict:
    path = Path(__file__).resolve().parents[1] / "data" / "time_baselines.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


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


def market_surface_kind(surface: str) -> str:
    value = clean(surface)
    if "障" in value:
        return "jump"
    if "ダ" in value:
        return "dirt"
    return "turf"


def entity_context_prior(
    model: dict,
    kind: str,
    entity_id: str,
    entity_name: str,
    surface: str,
) -> tuple[float, float]:
    """Return (overall, target-surface) market priors without using today's market."""
    overall = entity_prior(model, kind, entity_id, entity_name)
    priors = model.get("entityPriors") or {}
    surface_key = market_surface_kind(surface)
    by_id = priors.get(f"{kind}Surface") or {}
    by_name = priors.get(f"{kind}NameSurface") or {}
    scoped = None
    if entity_id:
        scoped = by_id.get(f"{entity_id}|{surface_key}")
    if scoped is None:
        key = clean(entity_name)
        if key:
            scoped = by_name.get(f"{key}|{surface_key}")
    if scoped is None:
        return overall, overall
    return overall, clamp01(scoped)


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
    condition_match = re.search(
        r"(?:芝|ダ|障(?:害)?)\s*\d{3,4}\s*m?\s*(稍重|不良|良|重)", raw
    )
    track_condition = condition_match.group(1) if condition_match else ""

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
        "trackCondition": track_condition,
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





def build_prediction(card: dict) -> dict:
    """Build a live prediction using the shared deterministic prediction core."""
    entries = card["entries"]
    horse_count = len(entries)
    if horse_count < 5:
        raise ValueError(f"{card['raceId']}: need at least 5 horses, got {horse_count}")

    core_kwargs = {
        "venue": card["venue"],
        "surface": card.get("surface", ""),
        "distance_m": card.get("distanceM"),
    }
    # Production v2 does not accept the v3-only keyword. Until apply promotes v3,
    # keep the live adapter perfectly backward-compatible with the current production file.
    if "v3-run-flow-power" in str(MODEL_VERSION):
        core_kwargs["time_baselines"] = load_time_baselines()

    index_core = build_index_core(
        [
            {
                "no": int(entry["no"]),
                "name": entry.get("name", ""),
                "histories": list(entry.get("histories", []))[:5],
                "age": entry.get("age"),
            }
            for entry in entries
        ],
        **core_kwargs,
    )
    detail_horses = index_core["horses"]
    totals = index_core["totals"]
    runs_by_no = index_core["runsByNo"]
    pace_regime = index_core["paceRegime"]

    # Market-popularity model.  Feature construction and scoring are shared with
    # the historical rebuild; only the source of the calibrated coefficients differs.
    popularity_model = load_popularity_model()
    current_class_level = int(card.get("classLevel", 3))
    total_rank_strength, recent_rank_strength = rank_strengths(detail_horses, totals)
    entry_by_no = {int(e["no"]): e for e in entries}

    feature_rows: list[dict] = []
    feature_horses: list[dict] = []
    for h in detail_horses:
        no = int(h["no"])
        entry = entry_by_no[no]
        jockey_market, jockey_surface_market = entity_context_prior(
            popularity_model, "jockey",
            entry.get("jockeyId", ""), entry.get("jockeyName", ""),
            card.get("surface", ""),
        )
        trainer_market, trainer_surface_market = entity_context_prior(
            popularity_model, "trainer",
            entry.get("trainerId", ""), entry.get("trainerName", ""),
            card.get("surface", ""),
        )
        factors, context = build_market_profile(
            runs_by_no.get(no, []),
            total_rank_strength=total_rank_strength[no],
            recent_rank_strength=recent_rank_strength[no],
            current_carried_weight=parse_float(entry.get("currentCarriedWeight")),
            jockey_market_strength=jockey_market,
            trainer_market_strength=trainer_market,
            jockey_surface_market_strength=jockey_surface_market,
            trainer_surface_market_strength=trainer_surface_market,
            age=entry.get("age"),
            current_class_level=current_class_level,
            current_surface=card.get("surface", ""),
            current_distance=card.get("distanceM"),
            current_date=card.get("targetDate") or date.today().isoformat(),
        )
        feature_row = build_popularity_feature_row(h, factors)
        feature_rows.append(feature_row)
        feature_horses.append(h)
        h["popularityFactors"] = {
            key: round(float(value) * 100, 1) for key, value in factors.items()
        }
        h["popularityContext"] = {
            **context,
            "model": popularity_model.get("version") or "fallback-v54",
            "historicalOddsRuns": int(round(feature_row.get("history_count", 0.0) * 5)),
        }

    classifier = load_popularity_classifier()
    if classifier is not None and feature_rows:
        X = np.asarray(
            [[float(row.get(name, 0.5)) for name in FEATURE_COLS] for row in feature_rows],
            dtype=float,
        )
        probabilities = classifier.predict_proba(X)[:, 1]
        for h, score in zip(feature_horses, probabilities):
            h["_popScore"] = float(score)
    else:
        for h, row in zip(feature_horses, feature_rows):
            h["_popScore"] = fallback_top3_score(row)

    expected_popularity = expected_popularity_from_scores(detail_horses)
    for h in detail_horses:
        h["expectedPopularity"] = int(expected_popularity[h["no"]])

    prediction, danger_no, _target_count = select_prediction(
        detail_horses, totals, expected_popularity
    )

    for h in detail_horses:
        h["excluded"] = h["no"] == danger_no
        h.pop("_popScore", None)

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
        "danger": [danger_no],
        "indexDetail": index_detail,
        "estimatedPopularity": {str(k): int(v) for k, v in expected_popularity.items()},
        "totalIndex": {str(h["no"]): int(h["total"]) for h in detail_horses},
        "modelMeta": {
            "version": MODEL_VERSION,
            "estimatedPopularity": {str(k): int(v) for k, v in expected_popularity.items()},
            "totalIndex": {str(h["no"]): int(h["total"]) for h in detail_horses},
            "indexDetail": index_detail,
            "performanceSource": (
                "netkeiba pre-race 5-run card; per-run 走/展/力 0-100; "
                "走=median/MAD standard-clock normalization; near=36/25/18/12/9; "
                "current=時35%+展33%+実32%; total=近走40%+今回60%"
                if "v3-run-flow-power" in MODEL_VERSION
                else "netkeiba pre-race 5-run card; legacy per-run 展開/タイム/成績; "
                     "近走60% + 今回40%; 今回=展開50%+コース50%"
            ),
            "popularityMethod": (
                "v54 Top3 classifier: previous-race popularity + previous win odds, "
                "recent ability indices, jockey/trainer historical market priors and context; "
                "current-race odds/actual popularity/bodyweight are not used"
            ),
            "selectionRule": SELECTION_RULE_TEXT,
            "logicSource": "scripts/prediction_logic_production.py",
            "prohibitedInputs": [
                "current odds", "current actual popularity",
                "horse bodyweight", "horse bodyweight change",
            ],
        },
    }
