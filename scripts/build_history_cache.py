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
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

JST = ZoneInfo("Asia/Tokyo")
CACHE_VERSION = "predictjra-historical-facts-v3-card-complete"
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
            ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
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


def build_cache(source_root: Path, cache_dir: Path) -> dict:
    card_root = source_root / "data" / "race_cards"
    if not card_root.exists():
        raise FileNotFoundError(card_root)

    available_dates: list[str] = []
    safe_dates: list[str] = []
    skipped_dates: list[dict] = []
    ignored_files: list[dict] = []
    inspections: dict[str, dict] = {}

    for card_dir in sorted(card_root.iterdir()):
        if not card_dir.is_dir() or not re.fullmatch(r"20\d{6}", card_dir.name):
            continue
        if not any(card_dir.glob("*.csv")):
            continue
        try:
            date_s = datetime.strptime(card_dir.name, "%Y%m%d").date().isoformat()
        except ValueError:
            continue
        available_dates.append(date_s)
        inspection = inspect_card_date(source_root, card_dir)
        inspections[date_s] = inspection
        if inspection["ignoredFiles"]:
            ignored_files.append({"date": date_s, "files": inspection["ignoredFiles"]})
        if inspection["safe"]:
            safe_dates.append(date_s)
        else:
            skipped_dates.append({
                "date": date_s,
                "reason": inspection["reason"],
                "schemaErrors": inspection["schemaErrors"],
                "missingArchives": inspection["missingArchives"],
            })

    if not safe_dates:
        raise RuntimeError("No complete historical race-card dates are available for caching")

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
            card_dir = source_root / "data" / "race_cards" / compact
            race_files = [card_dir / name for name in inspections[date_s]["raceFiles"]]
            date_race_counts[date_s] = len(race_files)
            legacy_count = 0
            synthesized_count = 0
            removed_columns: set[str] = set()
            synthesized_reasons: list[dict] = []

            for card_path in race_files:
                rid = card_path.stem
                result_path = source_root / "data" / "race_results" / rid[:4] / f"{rid}.csv"
                payout_path = source_root / "data" / "race_payouts" / f"{rid}.csv"
                for path in (card_path, result_path, payout_path):
                    if not path.exists():
                        raise FileNotFoundError(f"cache source missing: {path}")

                card = sanitize_card(read_csv(card_path))
                ok, reason = valid_race_frame(card, rid)
                if not ok:
                    raise ValueError(f"{rid}: invalid sanitized card: {reason}")
                card_dst = staging / card_path.relative_to(source_root)
                card_dst.parent.mkdir(parents=True, exist_ok=True)
                card.to_csv(card_dst, index=False, encoding="utf-8-sig")
                copied_paths.add(str(card_path.relative_to(source_root)))

                snapshot, snapshot_meta = choose_runner_snapshot(
                    source_root, compact, rid, card
                )
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
                "races": len(race_files),
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
                "raceEnumeration": "all canonical 12-digit archived race cards on each completed date",
                "stored": "sanitized program/runner facts, runner-set snapshots, final results, trifecta payouts, and required historical race result files",
                "notStored": "target-race current odds/actual popularity/bodyweight, archived model outputs, Predictjra derived indices, selections, hit judgement, or recovery-rate calculations",
                "runnerSnapshot": "use sanitized legacy prediction only when valid and runner-identical; otherwise synthesize race_id/horse_number only from sanitized race card",
                "completion": "a date is included only when every canonical race card is valid and has final result + payout",
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
