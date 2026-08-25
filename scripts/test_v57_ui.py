#!/usr/bin/env python3
"""Static regression contract for Predictjra v57 UI-only changes."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
app = (ROOT / "app.js").read_text(encoding="utf-8")
css = (ROOT / "styles.css").read_text(encoding="utf-8")


def test_summary_layout_and_overall_frame() -> None:
    assert "grid-template-columns: minmax(160px, 1fr) 420px 30px;" in css
    assert "grid-template-columns: repeat(3, minmax(0, 1fr));" in css
    assert "column-gap: 18px;" in css
    assert ".overall-card {" in css
    assert "border: 3px solid var(--hit);" in css
    assert "border-radius: 0;" in css


def test_debut_result_display() -> None:
    assert "if (debut) return 'payout-miss';" in app
    assert "予想対象外" in app
    assert "debut ? '-%'" not in app
    assert ".no-prediction-label" in css and "text-align: left;" in css


def test_result_roles_are_not_colored_borders() -> None:
    assert "border: 3px solid #b3261e" not in css
    assert "border: 3px solid #16824b" not in css
    assert "border: 3px solid #7446b8" not in css
    assert 'width: 14px;' in css
    assert 'height: 14px;' in css
    assert '.horse-box.result-role-main::before' in css
    assert '.horse-box.result-role-danger::before' in css


def test_modal_default_and_labels() -> None:
    assert "Number(b.total || 0) - Number(a.total || 0)" in app
    assert 'aria-sort="descending" data-sort-direction="desc" data-initial-sort="desc">総合</th>' in app
    assert '>今走</th>' in app
    assert '>今回</th>' in app
    assert '>今回評価</th>' not in app
    assert "index-table th:nth-child(12)" in css and "index-table th:nth-child(14)" in css


def test_two_digit_score_display_without_changing_sort_values() -> None:
    assert "Math.min(99, Math.round(n))" in app
    assert "padStart(2, '0')" in app
    assert '${score2(horse.total)}' in app
    assert '${score2(horse.recentIndex)}' in app
    assert '${score2(horse.today)}' in app
    assert '${score2(part)}' in app
    # Sorting keeps raw values rather than the presentation-capped value.
    assert 'data-sort-value="${horse.total}"' in app


if __name__ == "__main__":
    tests = [
        test_summary_layout_and_overall_frame,
        test_debut_result_display,
        test_result_roles_are_not_colored_borders,
        test_modal_default_and_labels,
        test_two_digit_score_display_without_changing_sort_values,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"OK: {len(tests)} v57 UI regression tests passed")
