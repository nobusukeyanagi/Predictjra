#!/usr/bin/env python3
"""Regression checks for the repository's maintenance/import boundaries."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_root_d2_is_compatibility_shim():
    root_text = (ROOT / "single_win_d2.py").read_text(encoding="utf-8")
    assert "from scripts.single_win_d2 import *" in root_text
    assert "HistGradientBoostingClassifier" not in root_text

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts import single_win_d2 as canonical
    compat = load_module("compat_d2", ROOT / "single_win_d2.py")
    assert compat.MODEL_VERSION == canonical.MODEL_VERSION
    assert compat.Policy is canonical.Policy


def test_root_d2_cli_and_test_are_shims():
    optimizer = (ROOT / "optimize_single_win_d2.py").read_text(encoding="utf-8")
    root_test = (ROOT / "test_single_win_d2.py").read_text(encoding="utf-8")
    assert "from scripts.optimize_single_win_d2 import main" in optimizer
    assert "from scripts.test_single_win_d2 import" in root_test
    assert optimizer.count("def ") == 0
    assert root_test.count("def ") == 0


def test_prediction_import_boundaries():
    compat = (ROOT / "scripts" / "prediction_logic.py").read_text(encoding="utf-8")
    live = (ROOT / "scripts" / "predict_engine.py").read_text(encoding="utf-8")
    rebuild = (ROOT / "scripts" / "rebuild_history.py").read_text(encoding="utf-8")
    assert "from prediction_logic_production import *" in compat
    assert "from prediction_logic_production import" in live
    assert "from prediction_logic_candidate import" in rebuild


def main():
    tests = [
        test_root_d2_is_compatibility_shim,
        test_root_d2_cli_and_test_are_shims,
        test_prediction_import_boundaries,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"OK: {len(tests)} project-layout regression tests passed")


if __name__ == "__main__":
    main()
