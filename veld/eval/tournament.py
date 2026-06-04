from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path

import torch

from veld.agents.mcts import MCTSAgent
from veld.core.game import Game, RandomAgent
from veld.core.pieces import Player
from veld.env.action_codec import ActionCodec
from veld.env.veld_env import encode_state
from veld.train.ppo_train import MaskedCategorical, VeldCNN


class PPOCheckpointAgent:
    def __init__(self, checkpoint: str | Path) -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = VeldCNN(payload["n_actions"])
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()
        self.codec = ActionCodec()

    def choose_action(self, state):
        legal = self.codec.legal_indices(state)
        if not legal:
            return None
        mask = torch.zeros((1, self.codec.action_space_size), dtype=torch.bool)
        mask[0, legal] = True
        obs = torch.tensor(encode_state(state))
        with torch.no_grad():
            logits, _ = self.model(obs)
            dist = MaskedCategorical(logits, mask)
            idx = int(dist.probs.argmax(dim=-1).item())
        if idx not in legal:
            idx = legal[0]
        return self.codec.decode(idx, state)


class TournamentEvaluator:
    def __init__(self, checkpoint_dir: str | Path = "checkpoints", n_games: int = 200) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.n_games = n_games
        self.players = self._load_players()
        self.elo = {name: 1000.0 for name in self.players}

    def _load_players(self):
        players = {
            "RandomAgent": lambda seed: RandomAgent(seed=seed),
            "MCTSAgent(200)": lambda seed: MCTSAgent(n_simulations=200, seed=seed),
            "MCTSAgent(500)": lambda seed: MCTSAgent(n_simulations=500, seed=seed),
        }
        for checkpoint in sorted(self.checkpoint_dir.glob("*.pt")):
            players[f"PPO:{checkpoint.stem}"] = lambda seed, cp=checkpoint: PPOCheckpointAgent(cp)
        return players

    def run(self, output_csv: str | Path = "results/elo_leaderboard.csv") -> list[tuple[str, float]]:
        for a_name, b_name in combinations(self.players, 2):
            score_a = 0.0
            for game_idx in range(self.n_games):
                game = Game(seed=game_idx)
                agent_a = self.players[a_name](game_idx)
                agent_b = self.players[b_name](game_idx + 999)
                final = game.play_episode(agent_a, agent_b)
                if final.winner == Player.RANGER_A:
                    score_a += 1.0
                elif final.winner is None:
                    score_a += 0.5
            actual_a = score_a / self.n_games
            self._update_elo(a_name, b_name, actual_a)
        leaderboard = sorted(self.elo.items(), key=lambda item: item[1], reverse=True)
        self._write_csv(output_csv, leaderboard)
        return leaderboard

    def _update_elo(self, a: str, b: str, actual_a: float, k: float = 32.0) -> None:
        expected_a = 1.0 / (1.0 + 10 ** ((self.elo[b] - self.elo[a]) / 400.0))
        self.elo[a] += k * (actual_a - expected_a)
        self.elo[b] += k * ((1.0 - actual_a) - (1.0 - expected_a))

    def _write_csv(self, path: str | Path, leaderboard: list[tuple[str, float]]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["agent", "elo"])
            for name, rating in leaderboard:
                writer.writerow([name, f"{rating:.2f}"])
