#!/usr/bin/env python3
"""Regression contract for the v67 ROI-oriented index weights retained by v68 D selection."""
from __future__ import annotations

from prediction_logic_candidate import (
    CURRENT_TOTAL_WEIGHT,
    FEATURE_COLS,
    MODEL_VERSION,
    PER_RUN_WEIGHTS,
    RECENCY_WEIGHTS,
    RECENT_TOTAL_WEIGHT,
)


def main() -> int:
    assert MODEL_VERSION == "predictjra-live-index-v3-run-flow-power-v68-d-single-ev"
    assert PER_RUN_WEIGHTS == {"run": 0.35, "flow": 0.33, "power": 0.32}
    assert RECENCY_WEIGHTS == [0.36, 0.25, 0.18, 0.12, 0.09]
    assert RECENT_TOTAL_WEIGHT == 0.40
    assert CURRENT_TOTAL_WEIGHT == 0.60
    assert abs(sum(PER_RUN_WEIGHTS.values()) - 1.0) < 1e-12
    assert abs(sum(RECENCY_WEIGHTS) - 1.0) < 1e-12
    assert abs(RECENT_TOTAL_WEIGHT + CURRENT_TOTAL_WEIGHT - 1.0) < 1e-12
    assert CURRENT_TOTAL_WEIGHT > RECENT_TOTAL_WEIGHT

    # v68 keeps the v67 weighting and adds D-plan main selection. Keep no-leakage intact.
    forbidden = {
        "current_odds", "odds", "win_odds", "actual_popularity",
        "popularity", "bodyweight", "body_weight", "weight_change",
    }
    assert forbidden.isdisjoint(FEATURE_COLS)
    print("v67 weights retained under v68 D singleEV selection: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
