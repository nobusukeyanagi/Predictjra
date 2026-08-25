#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
css = (ROOT / "styles.css").read_text(encoding="utf-8")

# All three role symbols use exactly the same outer-circle geometry.
shared = ".horse-box.result-role-main::after,\n.horse-box.result-role-second::after,\n.horse-box.result-role-danger::after"
assert shared in css
assert "width: 14px;" in css
assert "height: 14px;" in css
assert "border: 1.5px solid currentColor;" in css
assert "border-radius: 50%;" in css

# Main = ◎, second = ○, danger = circled ×. Implementation may evolve.
assert ".horse-box.result-role-main::before" in css
assert ".horse-box.result-role-second::before" in css
assert ".horse-box.result-role-danger::before" in css

# Requested role colors remain: main/second red, danger purple.
assert ".horse-box.result-role-main::after,\n.horse-box.result-role-second::after {\n  color: var(--hit);" in css
assert ".horse-box.result-role-danger::after {\n  color: #6f42c1;" in css

# Obsolete letter/triangle badges must not return.
for obsolete in ('content: "本";', 'content: "対";', 'content: "危";', 'clip-path: polygon(50% 0%, 100% 100%, 0% 100%);'):
    assert obsolete not in css

print("v62 result role symbols OK")
