from __future__ import annotations

import json
from pathlib import Path

from veld.core.game import Game, PlacementAction
from veld.core.rules import Action


def save_game_log(path: str | Path, turns: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"turns": turns}, indent=2), encoding="utf-8")


def load_game_log(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def replay_ascii(path: str | Path) -> str:
    data = load_game_log(path)
    game = Game()
    frames = [game.render()]
    # Full state restoration is intentionally JSON-log driven in later training runs.
    for turn in data.get("turns", []):
        frames.append(f"turn={turn.get('turn')} action={turn.get('action')} state_hash={turn.get('state_hash')}")
    return "\n\n".join(frames)


def action_to_json(action: Action | PlacementAction) -> dict:
    return {
        "kind": action.__class__.__name__,
        "piece_id": getattr(action, "piece_id", None),
        "piece_type": getattr(getattr(action, "piece_type", None), "value", None),
        "target": {"q": action.target_hex.q, "r": action.target_hex.r},
    }
