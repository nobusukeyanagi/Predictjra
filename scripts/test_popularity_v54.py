#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from joblib import load

from prediction_logic_production import (
    FEATURE_COLS,
    MODEL_VERSION,
    POPULARITY_MODEL_VERSION,
    odds_strength,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    assert MODEL_VERSION == "predictjra-live-index-v3-run-flow-power-v54-prior-odds-top3"
    assert POPULARITY_MODEL_VERSION == "predictjra-popularity-v54-hgb-prior-odds"
    forbidden = {"current_odds", "actual_popularity", "horse_weight", "bodyweight"}
    assert not forbidden.intersection(FEATURE_COLS)
    assert "last_odds_strength" in FEATURE_COLS
    assert "recent3_odds_strength" in FEATURE_COLS
    assert odds_strength(2.0) > odds_strength(10.0) > odds_strength(50.0)

    meta = json.loads((ROOT / "data/popularity_model.json").read_text(encoding="utf-8"))
    assert meta["version"] == POPULARITY_MODEL_VERSION
    assert meta["features"] == FEATURE_COLS
    assert float(meta["validation"]["meanTop3OverlapRate"]) >= 0.72
    assert "新馬" in meta.get("debutRacePolicy", "")

    model = load(ROOT / "data/popularity_model_v54.joblib")
    assert hasattr(model, "predict_proba")
    # 0.5-neutral row must be accepted with the exact live feature count.
    proba = model.predict_proba([[0.5] * len(FEATURE_COLS)])
    assert proba.shape == (1, 2)
    print(
        f"v54 popularity model OK: features={len(FEATURE_COLS)} "
        f"top3={meta['validation']['meanTop3OverlapRate']*100:.2f}%"
    )


if __name__ == "__main__":
    main()
