#!/usr/bin/env python3
"""Regression contract for Actions that write Predictjra main.

Rebuild historical predictions can run for a long time. It must not race the scheduled
Update race data workflow for main, and both writers must safely retry non-conflicting
manual pushes without ever force-pushing.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPDATE = ROOT / ".github" / "workflows" / "update-races.yml"
REBUILD = ROOT / ".github" / "workflows" / "rebuild-history.yml"
GROUP = "group: predictjra-data-writer"


def check(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert GROUP in text, f"{path.name}: shared writer concurrency group is missing"
    assert "cancel-in-progress: false" in text, f"{path.name}: writer must queue, not cancel"
    assert "git push origin HEAD:main" in text, f"{path.name}: explicit safe push is missing"
    assert "git fetch origin main" in text, f"{path.name}: push retry fetch is missing"
    assert "git rebase origin/main" in text, f"{path.name}: push retry rebase is missing"
    assert "force" not in "\n".join(
        line.lower() for line in text.splitlines()
        if line.strip().startswith("git push")
    ), f"{path.name}: force-push must never be used"


def main() -> int:
    check(UPDATE)
    check(REBUILD)
    print("OK: Actions main-writer serialization and safe push retry contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
