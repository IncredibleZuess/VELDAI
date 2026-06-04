from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from veld.agents.mcts import MCTSAgent
from veld.core.game import Game, RandomAgent
from veld.core.pieces import Player


def main() -> None:
    wins = draws = losses = 0
    games = 50
    for idx in range(games):
        game = Game(seed=idx)
        final = game.play_episode(
            MCTSAgent(n_simulations=200, time_budget_seconds=0.0, seed=idx),
            RandomAgent(seed=idx + 5000),
        )
        if final.winner == Player.RANGER_A:
            wins += 1
        elif final.winner is None:
            draws += 1
        else:
            losses += 1
    print(f"MCTS vs Random over {games} games: wins={wins} draws={draws} losses={losses} win_rate={wins / games:.2%}")


if __name__ == "__main__":
    main()
