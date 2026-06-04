from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from veld.agents import MCTSAgent, PPOCheckpointAgent
from veld.core.board import Hex
from veld.core.game import Agent, Game, PlacementAction, RandomAgent, get_legal_placements
from veld.core.pieces import Player
from veld.core.rules import Action, get_all_legal_actions


AI_MODES = {
    "human_vs_human": set(),
    "human_vs_random": {Player.RANGER_B},
    "random_vs_random": {Player.RANGER_A, Player.RANGER_B},
    "human_vs_mcts": {Player.RANGER_B},
    "human_vs_ppo": {Player.RANGER_B},
    "ppo_vs_mcts": {Player.RANGER_A, Player.RANGER_B},
    "ppo_watch": {Player.RANGER_A, Player.RANGER_B},
}
AUTO_STEP_MODES = {"random_vs_random", "ppo_vs_mcts", "ppo_watch"}


@dataclass
class GuiController:
    mode: str = "human_vs_human"
    seed: int | None = None
    watch_opponent_type: str = "random"
    checkpoint: str | Path | None = None
    game: Game = field(init=False)
    selected_piece_id: int | None = None
    legal_actions: list[Action | PlacementAction] = field(default_factory=list)
    autoplay: bool = False
    agent_a: Agent = field(init=False)
    agent_b: Agent = field(init=False)

    def __post_init__(self) -> None:
        self.game = Game(seed=self.seed)
        self.agent_a = RandomAgent(seed=self.seed)
        self.agent_b = RandomAgent(seed=None if self.seed is None else self.seed + 1)
        if self.mode == "human_vs_mcts":
            self.agent_b = MCTSAgent(seed=None if self.seed is None else self.seed + 1)
        elif self.mode == "human_vs_ppo":
            self.agent_b = self._build_ppo_agent()
        elif self.mode == "ppo_vs_mcts":
            self.agent_a = self._build_ppo_agent()
            self.agent_b = MCTSAgent(seed=None if self.seed is None else self.seed + 1)
        elif self.mode == "ppo_watch":
            self.agent_a = self._build_ppo_agent()
            self.agent_b = self._build_watch_opponent()
        self.refresh_legal_actions()

    @property
    def state(self):
        return self.game.state

    def reset(self) -> None:
        self.game.reset()
        self.selected_piece_id = None
        self.refresh_legal_actions()

    def undo(self) -> None:
        self.game.undo()
        self.selected_piece_id = None
        self.refresh_legal_actions()

    def current_player_is_ai(self) -> bool:
        return self.state.current_player in AI_MODES.get(self.mode, set())

    def refresh_legal_actions(self) -> None:
        if self.state.phase == "placement":
            self.legal_actions = get_legal_placements(self.state.current_player, self.state)
        elif self.selected_piece_id is None:
            self.legal_actions = get_all_legal_actions(self.state)
        else:
            self.legal_actions = [
                action for action in get_all_legal_actions(self.state) if action.piece_id == self.selected_piece_id
            ]

    def handle_hex_click(self, hex_: Hex | None) -> bool:
        before = self.state
        if hex_ is None or self.state.done or self.current_player_is_ai():
            return False
        if self.state.phase == "placement":
            for action in get_legal_placements(self.state.current_player, self.state):
                if action.target_hex == hex_:
                    self.game.step(action)
                    self.selected_piece_id = None
                    self.refresh_legal_actions()
                    return self.state is not before
            return False

        if self.selected_piece_id is not None:
            for action in self.legal_actions:
                if isinstance(action, Action) and action.target_hex == hex_:
                    self.game.step(action)
                    self.selected_piece_id = None
                    self.refresh_legal_actions()
                    return self.state is not before

        piece_id, piece = self.state.piece_at(hex_)
        if piece is not None and piece.owner == self.state.current_player:
            self.selected_piece_id = piece_id
            self.refresh_legal_actions()
            return False

        self.selected_piece_id = None
        self.refresh_legal_actions()
        return False

    def choose_ai_action(self):
        if self.state.current_player == Player.RANGER_A:
            return self.agent_a.choose_action(self.state)
        return self.agent_b.choose_action(self.state)

    def _build_ppo_agent(self) -> PPOCheckpointAgent:
        if self.checkpoint is None:
            raise ValueError(
                f"A PPO checkpoint is required for mode: {self.mode}. "
                "Provide one using the --checkpoint argument."
            )
        return PPOCheckpointAgent(self.checkpoint)

    def _build_watch_opponent(self) -> Agent:
        if self.watch_opponent_type == "mcts":
            return MCTSAgent(seed=None if self.seed is None else self.seed + 1)
        if self.watch_opponent_type == "random":
            return RandomAgent(seed=None if self.seed is None else self.seed + 1)
        raise ValueError(
            "Invalid watch opponent type. "
            f"Expected one of random|mcts, got: {self.watch_opponent_type}"
        )

    def step_ai(self) -> bool:
        if self.state.done or not self.current_player_is_ai():
            return False
        action = self.choose_ai_action()
        if action is None:
            self.game.state = self.game.state.with_updates(done=True, winner=self.state.current_player.opponent())
            return True
        self.game.step(action)
        self.selected_piece_id = None
        self.refresh_legal_actions()
        return True

    @property
    def auto_step_enabled(self) -> bool:
        return self.autoplay or self.mode in AUTO_STEP_MODES
