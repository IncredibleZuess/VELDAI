from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from veld.core.board import Hex, HexType
from veld.core.game import Game, RandomAgent
from veld.core.pieces import PieceType, Player
from veld.env.action_codec import ACTION_SPACE_SIZE, ActionCodec

PIECE_VALUE = {
    PieceType.IMPALA: 0.25,
    PieceType.EAGLE: 0.5,
    PieceType.LEOPARD: 0.75,
    PieceType.LION: 1.0,
}


def encode_state(state) -> np.ndarray:
    obs = np.zeros((7, 7, 8), dtype=np.float32)
    for hex_, terrain in state.board.hexes.items():
        q, r = hex_.q, hex_.r
        if terrain == HexType.VELD:
            obs[q, r, 0] = 1.0
        elif terrain == HexType.WATERING_HOLE:
            obs[q, r, 1] = 1.0
        elif terrain == HexType.THICKET:
            obs[q, r, 2] = 1.0
        owner = state.claims.get(hex_)
        if owner == Player.RANGER_A:
            obs[q, r, 3] = 1.0
        elif owner == Player.RANGER_B:
            obs[q, r, 4] = 1.0
    for piece in state.pieces:
        q, r = piece.position.q, piece.position.r
        if piece.owner == Player.RANGER_A:
            obs[q, r, 5] = PIECE_VALUE[piece.piece_type]
        else:
            obs[q, r, 6] = PIECE_VALUE[piece.piece_type]
    obs[:, :, 7] = 1.0 if state.current_player == Player.RANGER_A else -1.0
    return obs


class VeldEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, seed: int | None = None, opponent: RandomAgent | None = None, render_mode: str | None = None):
        super().__init__()
        self.codec = ActionCodec()
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(7, 7, 8), dtype=np.float32)
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        self.learning_player = Player.RANGER_A
        self.game = Game(seed=seed)
        self.opponent = opponent or RandomAgent(seed=None if seed is None else seed + 1000)
        self.render_mode = render_mode

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.game = Game(seed=seed)
            self.opponent = RandomAgent(seed=seed + 1000)
        else:
            self.game.reset()
        self._advance_to_learning_player()
        return encode_state(self.game.state), {"action_mask": self.action_masks()}

    def step(self, action: int):
        before = self.game.state
        before_claims = self._claim_score(before, self.learning_player)
        before_lion = self._lion_alive(before, self.learning_player)

        legal = set(self.codec.legal_indices(self.game.state))
        penalty = 0.0
        if int(action) not in legal:
            penalty = -0.05
            action = next(iter(legal), int(action))
        if legal:
            decoded = self.codec.decode(int(action), self.game.state)
            self.game.step(decoded)
        self._advance_to_learning_player()

        after = self.game.state
        reward = penalty + (self._claim_score(after, self.learning_player) - before_claims)
        if before_lion and not self._lion_alive(after, self.learning_player):
            reward -= 0.5
        if after.done:
            if after.winner == self.learning_player:
                reward += 1.0
            elif after.winner == self.learning_player.opponent():
                reward -= 1.0

        return encode_state(after), float(reward), bool(after.done), False, {"action_mask": self.action_masks()}

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=bool)
        for idx in self.codec.legal_indices(self.game.state):
            if 0 <= idx < self.action_space.n:
                mask[idx] = True
        return mask

    def render(self):
        return self.game.render()

    def _advance_to_learning_player(self) -> None:
        guard = 0
        while (
            not self.game.state.done
            and self.game.state.current_player != self.learning_player
            and guard < 64
        ):
            action = self.opponent.choose_action(self.game.state)
            if action is None:
                self.game.state = self.game.state.with_updates(done=True, winner=self.learning_player)
                break
            self.game.step(action)
            guard += 1

    def _claim_score(self, state, player: Player) -> float:
        score = 0.0
        for hex_, owner in state.claims.items():
            if owner != player:
                continue
            score += 0.2 if state.board.terrain_at(hex_) == HexType.WATERING_HOLE else 0.1
        return score

    def _lion_alive(self, state, player: Player) -> bool:
        return any(piece.owner == player and piece.piece_type == PieceType.LION for piece in state.pieces)
