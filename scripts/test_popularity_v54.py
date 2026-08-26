#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from joblib import load

from prediction_logic_production import (
    DEBUT_RACE_POLICY,
    FEATURE_COLS,
    MODEL_VERSION,
    POPULARITY_MODEL_VERSION,
    odds_strength,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-path", type=Path, default=ROOT / "data/popularity_model.json")
    parser.add_argument("--model-path", type=Path, default=ROOT / "data/popularity_model_v54.joblib")
    parser.add_argument(
        "--require-policy",
        action="store_true",
        help="Require the structured debutRacePolicy emitted by v56+ rebuilds.",
    )
    return parser.parse_args()


def validate_debut_policy(meta: dict, *, require_policy: bool) -> str:
    """Validate semantics, not a Japanese/English description string.

    v55 rebuilds accidentally omitted debutRacePolicy from regenerated metadata.
    During the one-time migration, accept that legacy file only when its validation
    method independently states that debut races were excluded.  Every v56 rebuild
    must emit the structured policy and is checked again with --require-policy.
    """
    policy = meta.get("debutRacePolicy")
    if isinstance(policy, dict):
        assert policy == DEBUT_RACE_POLICY, (
            "debutRacePolicy semantic flags do not match live prediction policy"
        )
        return "structured"

    if require_policy:
        raise AssertionError("structured debutRacePolicy is missing from regenerated metadata")

    # Migration compatibility only: v55-generated popularity_model.json has no
    # policy object, but its time-series validation metadata records the exclusion.
    method = str((meta.get("validation") or {}).get("method") or "").lower()
    assert "debut" in method and "excluded" in method, (
        "legacy metadata does not prove that debut races were excluded"
    )
    return "legacy-v55-migration"


def main() -> None:
    args = parse_args()

    assert MODEL_VERSION.startswith("predictjra-live-index-v3-run-flow-power-")
    assert POPULARITY_MODEL_VERSION == "predictjra-popularity-v54-hgb-prior-odds"
    forbidden = {"current_odds", "actual_popularity", "horse_weight", "bodyweight"}
    assert not forbidden.intersection(FEATURE_COLS)
    assert "last_odds_strength" in FEATURE_COLS
    assert "recent3_odds_strength" in FEATURE_COLS
    assert odds_strength(2.0) > odds_strength(10.0) > odds_strength(50.0)

    meta = json.loads(args.metadata_path.read_text(encoding="utf-8"))
    assert meta["version"] == POPULARITY_MODEL_VERSION
    assert meta["features"] == FEATURE_COLS
    assert float(meta["validation"]["meanTop3OverlapRate"]) >= 0.72
    policy_mode = validate_debut_policy(meta, require_policy=args.require_policy)

    model = load(args.model_path)
    assert hasattr(model, "predict_proba")
    proba = model.predict_proba([[0.5] * len(FEATURE_COLS)])
    assert proba.shape == (1, 2)
    print(
        f"v54 popularity model OK: features={len(FEATURE_COLS)} "
        f"top3={meta['validation']['meanTop3OverlapRate']*100:.2f}% "
        f"debutPolicy={policy_mode}"
    )


if __name__ == "__main__":
    main()
