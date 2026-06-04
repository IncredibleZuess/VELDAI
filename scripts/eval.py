from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from veld.eval.tournament import TournamentEvaluator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", default="checkpoints")
    parser.add_argument("--n_games", type=int, default=100)
    parser.add_argument("--output", default="results/elo_leaderboard.csv")
    args = parser.parse_args()
    leaderboard = TournamentEvaluator(args.checkpoints, args.n_games).run(args.output)
    for name, rating in leaderboard:
        print(f"{name:24s} {rating:7.2f}")


if __name__ == "__main__":
    main()
