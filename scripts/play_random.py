from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from veld.core.game import Game, RandomAgent


def main() -> None:
    for idx in range(10):
        game = Game(seed=idx)
        final = game.play_episode(RandomAgent(seed=idx * 2), RandomAgent(seed=idx * 2 + 1))
        winner = final.winner.label if final.winner else "draw"
        print(f"game={idx + 1} winner={winner} turns={final.turn_count}")


if __name__ == "__main__":
    main()
