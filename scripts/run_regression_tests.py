#!/usr/bin/env python3
"""Run Predictjra regression checks from one stable entry point.

This utility is for local/maintenance use.  It does not change production data or
prediction logic.  ``quick`` focuses on prediction/selection contracts; ``full`` runs
every ``scripts/test_*.py`` regression script.
"""
from __future__ import annotations

import argparse
import compileall
import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

QUICK_TESTS = [
    "test_project_layout.py",
    "test_prediction_logic.py",
    "test_prediction_logic_production.py",
    "test_predict_engine.py",
    "test_popularity_v54.py",
    "test_single_win_d2.py",
    "test_single_win_d3.py",
    "test_v67_single_roi_weights.py",
    "test_v68_d_single_ev.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=("quick", "full"),
        default="quick",
        help="quick = prediction-focused checks, full = every scripts/test_*.py",
    )
    parser.add_argument("--list", action="store_true", help="list tests without running them")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="run remaining tests after a failure and report all failures at the end",
    )
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="skip repository-wide Python compile check",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_status() -> str:
    candidate = SCRIPTS / "prediction_logic_candidate.py"
    production = SCRIPTS / "prediction_logic_production.py"
    if not candidate.is_file() or not production.is_file():
        return "candidate/production status unavailable"
    return "candidate=production" if sha256(candidate) == sha256(production) else "candidate differs from production"


def selected_tests(profile: str) -> list[Path]:
    if profile == "quick":
        return [SCRIPTS / name for name in QUICK_TESTS]
    return sorted(SCRIPTS.glob("test_*.py"))


def main() -> int:
    args = parse_args()
    tests = selected_tests(args.profile)

    missing = [str(path.relative_to(ROOT)) for path in tests if not path.is_file()]
    if missing:
        print("Missing regression test(s): " + ", ".join(missing), file=sys.stderr)
        return 2

    print(f"Predictjra regression profile: {args.profile}")
    print(f"Logic snapshots: {candidate_status()}")
    print(f"Tests selected: {len(tests)}")

    if args.list:
        for path in tests:
            print(path.relative_to(ROOT))
        return 0

    if not args.skip_compile:
        print("\n== compileall ==")
        if not compileall.compile_dir(ROOT, quiet=1):
            print("Python compile check failed.", file=sys.stderr)
            return 1

    env = os.environ.copy()
    current = env.get("PYTHONPATH", "")
    prefix = os.pathsep.join((str(SCRIPTS), str(ROOT)))
    env["PYTHONPATH"] = prefix if not current else prefix + os.pathsep + current

    failures: list[str] = []
    for path in tests:
        rel = str(path.relative_to(ROOT))
        print(f"\n== {rel} ==")
        result = subprocess.run([sys.executable, str(path)], cwd=ROOT, env=env, check=False)
        if result.returncode != 0:
            failures.append(rel)
            if not args.continue_on_error:
                break

    if failures:
        print("\nFAILED: " + ", ".join(failures), file=sys.stderr)
        return 1

    print(f"\nOK: {len(tests)} regression scripts passed ({args.profile})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
