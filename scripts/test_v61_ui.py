#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
app = (ROOT / "app.js").read_text(encoding="utf-8")
css = (ROOT / "styles.css").read_text(encoding="utf-8")

assert "function modalDateLabel(iso)" in app
assert "`${String(iso).slice(5)}(${weekdays[d.getDay()]})`" in app
assert "syncRaceDetailFromData(race, day.date)" in app
assert "detail.date = dayDate || detail.date || ''" in app
assert "disabledDetail.date = dayDate || disabledDetail.date || ''" in app
assert "modalDateLabel(detail.date)" in app
assert 'grid-template-areas:' in css
assert '"nav close"' in css
assert '"title title"' in css
assert '.index-modal-heading { display: contents; }' in css
assert '.index-modal-nav { grid-area: nav;' in css
assert '.index-modal-close { grid-area: close;' in css
assert 'grid-area: title;' in css
print("v61 modal date/mobile header UI OK")
