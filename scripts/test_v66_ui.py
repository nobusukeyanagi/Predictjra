#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / 'app.js').read_text(encoding='utf-8')
CSS = (ROOT / 'styles.css').read_text(encoding='utf-8')


def test_win_dead_heat_payouts_stack_on_horse_row_rhythm():
    assert 'function winPayoutMarkup' in APP
    assert 'class="win-payout-stack"' in APP
    assert 'class="win-payout-row"' in APP
    assert 'const singleMarkup = winPayoutMarkup(winItems);' in APP
    assert 'multi-payout-row' in APP
    assert '.win-payout-row,' in CSS
    assert 'height: 27px;' in CSS


def test_all_horse_boxes_are_same_immutable_square():
    assert '.horse-box,' in CSS
    assert '.index-horse-box,' in CSS
    assert 'flex: 0 0 27px !important;' in CSS
    assert 'width: 27px !important;' in CSS
    assert 'height: 27px !important;' in CSS
    assert 'min-width: 27px !important;' in CSS
    assert 'max-width: 27px !important;' in CSS


def test_opponent_column_has_room_for_five_boxes_and_padding():
    assert 'width: 790px !important;' in CSS
    assert 'nth-child(4)' in CSS and '168px !important' in CSS
    assert '5*27 + 4*2 = 143px' in CSS


def test_all_day_columns_share_fixed_widths_and_padding():
    assert '.race-table th,' in CSS and '.race-table td,' in CSS
    assert 'padding-left: 10px !important;' in CSS
    assert 'padding-right: 10px !important;' in CSS
    for width in ('76px', '54px', '168px', '156px', '66px', '90px', '126px'):
        assert f'width: {width} !important;' in CSS


if __name__ == '__main__':
    tests = [v for k, v in globals().items() if k.startswith('test_') and callable(v)]
    for test in tests:
        test()
    print(f'PASS: {len(tests)} v66 UI tests')
