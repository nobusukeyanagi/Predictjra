#!/usr/bin/env python3
"""Build a durable local cache of immutable historical source data for Predictjra.

The rebuild/backtest workflow should not repeatedly scan the remote archive. This script
copies only the safe pre-race dates plus the historical result files required by those
horses, then packs them into data/history_cache/history-source.tar.gz.

The cache intentionally stores source facts/snapshots, not calculated Predictjra indices.
Changing the prediction logic therefore reuses the same cached inputs while recalculating
all derived features and predictions.
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
CACHE_VERSION = "predictjra-historical-facts-v1"
SOURCE_REPO = "sugaimo15/keibayosoku"
SOURCE_REF = "claude/horse-racing-predictor-ak6crm"


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
        ).strip()
    except Exception:
        return ""


def prediction_snapshot_issues(pred: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    if "win_odds" in pred.columns:
        odds = pd.to_numeric(pred["win_odds"], errors="coerce")
        if (odds > 0).any():
            issues.append("current odds populated")
    if "horse_weight" in pred.columns:
        bodyweight = pred["horse_weight"].apply(clean_str)
        if bodyweight.ne("").any():
            issues.append("current horse bodyweight populated")
    if "popularity" in pred.columns:
        popularity = pd.to_numeric(pred["popularity"], errors="coerce")
        if popularity.notna().any():
            issues.append("current actual popularity populated")
    return issues


def inspect_prediction_date(pred_dir: Path) -> dict:
    valid_files: list[str] = []
    ignored_files: list[str] = []
    contaminated_files: list[dict] = []
    schema_errors: list[dict] = []

    for path in sorted(pred_dir.glob("*.csv")):
        if not re.fullmatch(r"\d{12}\.csv", path.name):
            ignored_files.append(path.name)
            continue
        try:
            pred = read_csv(path)
        except Exception as exc:  # noqa: BLE001
            schema_errors.append({"file": path.name, "reason": f"{type(exc).__name__}: {exc}"})
            continue

        missing = {"horse_number"} - set(pred.columns)
        if missing:
            schema_errors.append({
                "file": path.name,
                "reason": "missing core columns: " + ", ".join(sorted(missing)),
            })
            continue

        issues = prediction_snapshot_issues(pred)
        if issues:
            contaminated_files.append({"file": path.name, "issues": issues})
            continue
        valid_files.append(path.name)

    if contaminated_files:
        safe = False
        reason = (
            f"{len(contaminated_files)} prediction files contain current-race "
            "odds/bodyweight/popularity"
        )
    elif schema_errors:
        safe = False
        reason = f"{len(schema_errors)} prediction files have unsupported schema/read errors"
    elif not valid_files:
        safe = False
        reason = "no valid 12-digit race prediction files"
    else:
        safe = True
        reason = ""

    return {
        "safe": safe,
        "validRaceFiles": valid_files,
        "ignoredFiles": ignored_files,
        "contaminatedFiles": contaminated_files,
        "schemaErrors": schema_errors,
        "reason": reason,
    }


def copy_file(src: Path, root: Path, dst_root: Path) -> None:
    rel = src.relative_to(root)
    dst = dst_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def build_cache(source_root: Path, cache_dir: Path) -> dict:
    pred_root = source_root / "data" / "predictions"
    if not pred_root.exists():
        raise FileNotFoundError(pred_root)

    available_dates: list[str] = []
    safe_dates: list[str] = []
    skipped_dates: list[dict] = []
    ignored_files: list[dict] = []
    inspections: dict[str, dict] = {}

    for pred_dir in sorted(pred_root.iterdir()):
        if not pred_dir.is_dir() or not re.fullmatch(r"20\d{6}", pred_dir.name):
            continue
        if not any(pred_dir.glob("*.csv")):
            continue
        try:
            date_s = datetime.strptime(pred_dir.name, "%Y%m%d").date().isoformat()
        except ValueError:
            continue
        available_dates.append(date_s)
        inspection = inspect_prediction_date(pred_dir)
        inspections[date_s] = inspection
        if inspection["ignoredFiles"]:
            ignored_files.append({"date": date_s, "files": inspection["ignoredFiles"]})
        if inspection["safe"]:
            safe_dates.append(date_s)
        else:
            skipped_dates.append({
                "date": date_s,
                "reason": inspection["reason"],
                "contaminatedFiles": inspection["contaminatedFiles"],
                "schemaErrors": inspection["schemaErrors"],
            })

    # A clean prediction snapshot can exist before the corresponding result/payout has
    # been archived. Cache only completed dates so `auto` refresh is safe even while a
    # new race day is still in progress.
    completed_safe_dates: list[str] = []
    for date_s in safe_dates:
        compact = date_s.replace("-", "")
        missing: list[str] = []
        for filename in inspections[date_s]["validRaceFiles"]:
            rid = Path(filename).stem
            required = [
                source_root / "data" / "race_cards" / compact / f"{rid}.csv",
                source_root / "data" / "race_results" / rid[:4] / f"{rid}.csv",
                source_root / "data" / "race_payouts" / f"{rid}.csv",
            ]
            if any(not path.exists() for path in required):
                missing.append(rid)
        if missing:
            skipped_dates.append({
                "date": date_s,
                "reason": f"incomplete archived date: card/result/payout missing for {len(missing)} races",
                "missingRaceIds": missing,
                "contaminatedFiles": [],
                "schemaErrors": [],
            })
        else:
            completed_safe_dates.append(date_s)
    safe_dates = completed_safe_dates

    if not safe_dates:
        raise RuntimeError("No safe completed historical prediction dates are available for caching")

    source_sha = source_commit(source_root)
    cache_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="predictjra-history-cache-") as tmp:
        staging = Path(tmp) / "source"
        target_horse_ids: set[str] = set()
        date_race_counts: dict[str, int] = {}
        copied_paths: set[str] = set()

        # Copy target-day source snapshots/results. These are immutable inputs for rebuilds.
        for date_s in safe_dates:
            compact = date_s.replace("-", "")
            pred_dir = source_root / "data" / "predictions" / compact
            race_files = [pred_dir / name for name in inspections[date_s]["validRaceFiles"]]
            date_race_counts[date_s] = len(race_files)

            for pred_path in race_files:
                rid = pred_path.stem
                card_path = source_root / "data" / "race_cards" / compact / f"{rid}.csv"
                result_path = source_root / "data" / "race_results" / rid[:4] / f"{rid}.csv"
                payout_path = source_root / "data" / "race_payouts" / f"{rid}.csv"
                paths = [pred_path, card_path, result_path, payout_path]
                for path in paths:
                    if not path.exists():
                        raise FileNotFoundError(f"cache source missing: {path}")

                # Prediction CSV is already validated as a clean pre-race snapshot.
                copy_file(pred_path, source_root, staging)
                copied_paths.add(str(pred_path.relative_to(source_root)))

                # race_cards may have been refreshed after the race. Persist only stable
                # program fields by stripping current odds/bodyweight/actual popularity.
                card = read_csv(card_path).drop(
                    columns=["win_odds", "horse_weight", "popularity"], errors="ignore"
                )
                card_dst = staging / card_path.relative_to(source_root)
                card_dst.parent.mkdir(parents=True, exist_ok=True)
                card.to_csv(card_dst, index=False, encoding="utf-8-sig")
                copied_paths.add(str(card_path.relative_to(source_root)))

                copy_file(result_path, source_root, staging)
                copy_file(payout_path, source_root, staging)
                copied_paths.add(str(result_path.relative_to(source_root)))
                copied_paths.add(str(payout_path.relative_to(source_root)))

                if "horse_id" in card.columns:
                    target_horse_ids.update(
                        x for x in card["horse_id"].apply(clean_str).tolist() if x
                    )

        # One expensive scan only: retain complete historical race result files that contain
        # at least one target horse. Full race rows are kept because relative time/last3F
        # calculations need the other runners in that race too.
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

        archive_path = cache_dir / "history-source.tar.gz"
        tmp_archive = Path(tmp) / "history-source.tar.gz"
        with tarfile.open(tmp_archive, "w:gz", compresslevel=6) as tar:
            tar.add(staging / "data", arcname="data", recursive=True)

        # GitHub rejects individual files >=100 MiB. Keep a single archive when small,
        # otherwise split it into 80 MiB chunks that are reassembled by the workflow.
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
                "stored": "sanitized pre-race program cards, clean prediction snapshots, final results, trifecta payouts, and required historical race result files",
                "notStored": "Predictjra derived indices, axes/opponents/danger selections, hit judgement, or recovery-rate calculations",
                "unsafeDates": "prediction dates containing current odds/bodyweight/actual popularity are excluded",
            },
        }
        (cache_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
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
        "targetHorseCount": manifest["targetHorseCount"],
        "historicalResultFiles": manifest["historicalResultFiles"],
        "cachedFileCount": manifest["cachedFileCount"],
        "archiveSizeBytes": manifest["archive"]["sizeBytes"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
