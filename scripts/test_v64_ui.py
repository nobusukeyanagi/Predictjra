#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
css = (ROOT / "styles.css").read_text(encoding="utf-8")

# All daily tables must use one invariant width.
for rule in (
    "width: 756px !important;",
    "min-width: 756px !important;",
    "max-width: 756px !important;",
    "table-layout: fixed !important;",
):
    assert rule in css

# Main / Second columns need enough content width for the 27px square even with 10px side padding.
for nth in (2, 3):
    rule = (
        f".race-table th:nth-child({nth}), .race-table td:nth-child({nth}) "
        "{ width: 54px !important; min-width: 54px !important; max-width: 54px !important; }"
    )
    assert rule in css

# Role horse boxes themselves stay exact squares.
role_block = css.split(".horse-box.result-role-main,", 1)[1].split("}", 1)[0]
for rule in ("width: 27px;", "height: 27px;", "aspect-ratio: 1 / 1;"):
    assert rule in role_block

print("v64 fixed daily table widths and widened main/second columns OK")
