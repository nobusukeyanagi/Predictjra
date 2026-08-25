#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "app.js").read_text(encoding="utf-8")
CSS = (ROOT / "styles.css").read_text(encoding="utf-8")


def test_v3_labels_are_time_flow_record():
    assert "['時', '展', '実']" in APP
    assert '「時・展・実」' in APP
    assert '<h3>近走「時」</h3>' in APP
    assert '<h3>近走「実」</h3>' in APP
    assert '<h3>今回「時」</h3>' in APP
    assert '<h3>今回「実」</h3>' in APP


def test_multiple_trifecta_combinations_are_stacked():
    assert 'class="trifecta-result-stack"' in APP
    assert 'class="trifecta-result-line"' in APP
    assert "item.horses.map(no => box(Number(no)))" in APP
    assert "<span class=\"place-sep\">&gt;</span>" in APP


def test_multiple_payouts_align_by_row():
    assert 'function trifectaPayoutMarkup' in APP
    assert 'class="trifecta-payout-stack"' in APP
    assert 'class="trifecta-payout-row"' in APP
    assert "multi-trifecta-row" in APP
    assert ".trifecta-result-line" in CSS
    assert ".trifecta-payout-row" in CSS
    assert "min-height: 27px" in CSS


def test_fixed_day_columns_remain_invariant():
    assert "width: 780px !important" in CSS
    assert "nth-child(5)" in CSS and "156px !important" in CSS
    assert "nth-child(8)" in CSS and "126px !important" in CSS


if __name__ == "__main__":
    tests=[v for k,v in globals().items() if k.startswith('test_') and callable(v)]
    for t in tests: t()
    print(f"PASS: {len(tests)} v65 UI tests")
