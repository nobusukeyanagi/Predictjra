#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
css = (ROOT / "styles.css").read_text(encoding="utf-8")

role = ".horse-box.result-role-main,\n.horse-box.result-role-second,\n.horse-box.result-role-danger"
assert role in css
for rule in (
    "flex: 0 0 27px;", "width: 27px;", "min-width: 27px;", "max-width: 27px;",
    "height: 27px;", "min-height: 27px;", "max-height: 27px;", "aspect-ratio: 1 / 1;"
):
    assert rule in css

# Main inner ring must be a real circular border, not a radial-gradient approximation.
main_block = css.split(".horse-box.result-role-main::before {", 1)[1].split("}", 1)[0]
assert "width: 6px;" in main_block
assert "height: 6px;" in main_block
assert "border: 1.2px solid var(--hit);" in main_block
assert "border-radius: 50%;" in main_block
assert "radial-gradient" not in main_block

print("v63 square horse boxes and smooth main badge OK")
