#!/usr/bin/env python3
"""Static regression contract for Predictjra v60 trifecta recovery display."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
app = (ROOT / "app.js").read_text(encoding="utf-8")


def test_trifecta_rate_is_hit_only() -> None:
    assert "triHit && triRate != null ? percent(triRate) : ''" in app
    assert "debut ? '-%'" not in app
    assert "triRate == null ? '' : percent(triRate)" not in app


if __name__ == "__main__":
    test_trifecta_rate_is_hit_only()
    print("PASS test_trifecta_rate_is_hit_only")
    print("OK: v60 trifecta recovery-rate display regression test passed")
