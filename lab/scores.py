"""Read scorecards and gates without importing torch (the scoring itself is `sandwm.scorecard`)."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path


def card(run: Path) -> dict | None:
    """The scorecard of a fetched run, or None if it has none."""
    try:
        return json.loads((Path(run) / "eval" / "scorecard.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def gates(board: Path) -> dict[str, float]:
    """The [gates] table of the board's gates.toml: check name -> threshold."""
    try:
        return {k: float(v) for k, v in tomllib.loads((Path(board) / "gates.toml").read_text(encoding="utf-8")).get("gates", {}).items()}
    except (OSError, ValueError, TypeError):
        return {}


def failing(scorecard: dict, thresholds: dict[str, float]) -> list[str]:
    """The gates this scorecard does not pass.  A check's own direction says which side of the threshold passes."""
    out = []
    for name, threshold in thresholds.items():
        entry = scorecard["checks"].get(name)
        value = entry["value"] if entry else None
        if value is None or not (value >= threshold if entry["better"] == "higher" else value <= threshold):
            out.append(name)
    return out


def gate_line(run: Path, board: Path) -> str:
    """'gates 7/11' for a fetched run; '' when there is nothing to say."""
    scorecard, thresholds = card(run), gates(board)
    if not scorecard or not thresholds:
        return ""
    return f"gates {len(thresholds) - len(failing(scorecard, thresholds))}/{len(thresholds)}"
