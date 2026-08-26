#!/usr/bin/env python3
"""Regression contract for v68 D-plan all-race single-win main selection."""
from __future__ import annotations

from prediction_logic_candidate import MODEL_VERSION, SELECTION_RULE_TEXT, select_prediction


def main() -> int:
    horses = [
        {"no": 1, "total": 90, "recentIndex": 90},
        {"no": 2, "total": 82, "recentIndex": 82},
        {"no": 3, "total": 70, "recentIndex": 70},
        {"no": 4, "total": 88, "recentIndex": 88},
        {"no": 5, "total": 86, "recentIndex": 86},
        {"no": 6, "total": 84, "recentIndex": 84},
        {"no": 7, "total": 78, "recentIndex": 78},
        {"no": 8, "total": 76, "recentIndex": 76},
    ]
    totals = {h["no"]: float(h["total"]) for h in horses}
    expected = {no: no for no in range(1, 9)}
    prediction, danger, target = select_prediction(horses, totals, expected)

    assert MODEL_VERSION == "predictjra-live-index-v3-run-flow-power-v68-d-single-ev"
    assert "singleEV" in SELECTION_RULE_TEXT
    assert danger == 3 and target == 4
    assert prediction["axes"][0] == 4  # value main can differ from top total (#1)
    assert all(0 <= h["singleEV"] <= 99 for h in horses)
    assert horses[3]["singleEV"] > horses[0]["singleEV"]
    # All-race purchase requirement: a main is always emitted whenever prediction is valid.
    assert len(prediction["axes"]) == 2 and prediction["axes"][0] in totals
    print("v68 D-plan singleEV selection contract: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
