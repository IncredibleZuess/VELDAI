from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

from veld.core.game import PlacementAction, apply_placement, get_legal_placements
from veld.core.pieces import PieceType, Player
from veld.core.rules import Action, apply_action, get_all_legal_actions
from veld.core.state import GameState


def legal_actions_for(state: GameState) -> list[Action | PlacementAction]:
    if state.phase == "placement":
        return get_legal_placements(state.current_player, state)
    return get_all_legal_actions(state)


def transition(state: GameState, action: Action | PlacementAction) -> GameState:
    if state.phase == "placement":
        return apply_placement(state, action)  # type: ignore[arg-type]
    return apply_action(state, action)  # type: ignore[arg-type]


@dataclass
class MCTSNode:
    state: GameState
    parent: "MCTSNode | None" = None
    action: Action | PlacementAction | None = None
    children: list["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    untried_actions: list[Action | PlacementAction] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.untried_actions:
            self.untried_actions = legal_actions_for(self.state)

    def fully_expanded(self) -> bool:
        return not self.untried_actions


class MCTSAgent:
    def __init__(
        self,
        n_simulations: int = 200,
        exploration_c: float = math.sqrt(2.0),
        seed: int | None = None,
        time_budget_seconds: float | None = None,
    ) -> None:
        self.n_simulations = n_simulations
        self.exploration_c = exploration_c
        self.rng = random.Random(seed)
        self.time_budget_seconds = time_budget_seconds

    def choose_action(self, state: GameState, n_simulations: int | None = None):
        actions = legal_actions_for(state)
        if not actions:
            return None
        tactical = self._immediate_tactical_action(state, actions)
        if tactical is not None:
            return tactical

        root_player = state.current_player
        root = MCTSNode(state=state)
        simulations = n_simulations or self.n_simulations
        deadline = None if self.time_budget_seconds is None else time.monotonic() + self.time_budget_seconds

        for _ in range(simulations):
            if deadline is not None and time.monotonic() >= deadline:
                break
            node = self.select(root)
            if not node.state.done and node.untried_actions:
                node = self.expand(node)
            result = self.rollout(node.state, root_player)
            self.backprop(node, result)

        if not root.children:
            return self.rng.choice(actions)
        return max(root.children, key=lambda child: (child.visits, child.value / max(child.visits, 1))).action

    def select(self, node: MCTSNode) -> MCTSNode:
        while not node.state.done and node.fully_expanded() and node.children:
            node = max(node.children, key=self._ucb1)
        return node

    def expand(self, node: MCTSNode) -> MCTSNode:
        action = node.untried_actions.pop(self.rng.randrange(len(node.untried_actions)))
        child = MCTSNode(state=transition(node.state, action), parent=node, action=action)
        node.children.append(child)
        return child

    def rollout(self, state: GameState, root_player: Player, max_steps: int = 200) -> float:
        current = state
        steps = 0
        while not current.done and steps < max_steps:
            actions = legal_actions_for(current)
            if not actions:
                winner = current.current_player.opponent()
                current = current.with_updates(done=True, winner=winner)
                break
            tactical = self._immediate_tactical_action(current, actions)
            action = tactical or self.rng.choice(actions)
            current = transition(current, action)
            steps += 1
        if current.winner == root_player:
            return 1.0
        if current.winner is None:
            return 0.5
        return 0.0

    def backprop(self, node: MCTSNode, result: float) -> None:
        current: MCTSNode | None = node
        while current is not None:
            current.visits += 1
            current.value += result
            current = current.parent

    def _ucb1(self, node: MCTSNode) -> float:
        if node.visits == 0:
            return float("inf")
        parent_visits = max(1, node.parent.visits if node.parent else 1)
        return node.value / node.visits + self.exploration_c * math.sqrt(math.log(parent_visits) / node.visits)

    def _immediate_tactical_action(self, state: GameState, actions: list[Action | PlacementAction]):
        if state.phase == "placement":
            center = (3, 3)
            return min(
                actions,
                key=lambda action: (
                    abs(action.target_hex.q - center[0]) + abs(action.target_hex.r - center[1]),
                    action.target_hex.r,
                    action.target_hex.q,
                ),
            )
        for action in actions:
            if not isinstance(action, Action):
                continue
            _, target_piece = state.piece_at(action.target_hex)
            if target_piece is not None and target_piece.piece_type == PieceType.LION:
                return action
        capture_actions: list[tuple[int, Action]] = []
        claim_actions: list[tuple[int, Action]] = []
        enemy_lions = [
            piece.position
            for piece in state.pieces
            if piece.owner != state.current_player and piece.piece_type == PieceType.LION
        ]
        for action in actions:
            if not isinstance(action, Action):
                continue
            piece = state.pieces[action.piece_id]
            _, target_piece = state.piece_at(action.target_hex)
            if target_piece is not None:
                value = {
                    PieceType.LION: 100,
                    PieceType.LEOPARD: 8,
                    PieceType.EAGLE: 5,
                    PieceType.IMPALA: 3,
                }[target_piece.piece_type]
                capture_actions.append((value, action))
            if piece.piece_type == PieceType.IMPALA and action.target_hex not in state.claims:
                terrain_bonus = 3 if state.board.terrain_at(action.target_hex).value == "watering_hole" else 1
                claim_actions.append((terrain_bonus, action))
        if capture_actions:
            return max(capture_actions, key=lambda item: item[0])[1]
        if claim_actions:
            return max(claim_actions, key=lambda item: item[0])[1]
        if enemy_lions:
            enemy_lion = enemy_lions[0]
            return min(
                (action for action in actions if isinstance(action, Action)),
                key=lambda action: self._hex_distance(action.target_hex, enemy_lion),
                default=None,
            )
        return None

    def _hex_distance(self, a, b) -> int:
        aq, ar, as_ = a.q, a.r, -a.q - a.r
        bq, br, bs = b.q, b.r, -b.q - b.r
        return int((abs(aq - bq) + abs(ar - br) + abs(as_ - bs)) / 2)
