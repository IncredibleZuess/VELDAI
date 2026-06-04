from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

from veld.core.board import Hex, HexGrid, HexType
from veld.core.pieces import Piece, PieceType, Player
from veld.core.rules import Action, get_all_legal_actions, is_terminal
from veld.core.state import GameState


PLACEMENT_ORDER: tuple[PieceType, ...] = (
    PieceType.LION,
    PieceType.LEOPARD,
    PieceType.LEOPARD,
    PieceType.IMPALA,
    PieceType.IMPALA,
    PieceType.IMPALA,
    PieceType.IMPALA,
    PieceType.EAGLE,
)


@dataclass(frozen=True)
class PlacementAction:
    piece_type: PieceType
    target_hex: Hex


class Agent(Protocol):
    def choose_action(self, state: GameState):
        ...


def next_placement_piece(player: Player, state: GameState) -> PieceType | None:
    count = state.placement_counts.get(player, 0)
    if count >= len(PLACEMENT_ORDER):
        return None
    return PLACEMENT_ORDER[count]


def get_legal_placements(player: Player, state: GameState) -> list[PlacementAction]:
    if state.done or state.phase != "placement" or player != state.current_player:
        return []
    piece_type = next_placement_piece(player, state)
    if piece_type is None:
        return []
    legal_rows = {0, 1} if player == Player.RANGER_A else {5, 6}
    occupied = {piece.position for piece in state.pieces}
    actions: list[PlacementAction] = []
    for hex_, terrain in state.board.hexes.items():
        if hex_.r not in legal_rows or terrain == HexType.THICKET or hex_ in occupied:
            continue
        actions.append(PlacementAction(piece_type, hex_))
    return sorted(actions, key=lambda a: (a.target_hex.r, a.target_hex.q))


def apply_placement(state: GameState, placement_action: PlacementAction) -> GameState:
    legal = get_legal_placements(state.current_player, state)
    if placement_action not in legal:
        raise ValueError(f"Illegal placement: {placement_action}")
    player = state.current_player
    pieces = tuple(list(state.pieces) + [Piece(placement_action.piece_type, player, placement_action.target_hex)])
    counts = dict(state.placement_counts)
    counts[player] = counts.get(player, 0) + 1
    phase = "movement" if all(counts[p] >= len(PLACEMENT_ORDER) for p in Player) else "placement"
    next_player = player.opponent()
    if phase == "movement":
        next_player = Player.RANGER_A
    return state.with_updates(
        pieces=pieces,
        placement_counts=counts,
        current_player=next_player,
        phase=phase,
        last_action=placement_action,
    )


class RandomAgent:
    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def choose_action(self, state: GameState):
        if state.phase == "placement":
            actions = get_legal_placements(state.current_player, state)
        else:
            actions = get_all_legal_actions(state)
        if not actions:
            return None
        return self.rng.choice(actions)


class Game:
    def __init__(self, seed: int | None = None, max_turns: int = 200) -> None:
        self.seed = seed
        self.max_turns = max_turns
        self.rng = random.Random(seed)
        self.history: list[GameState] = []
        self.state = self.reset()

    def reset(self) -> GameState:
        self.history = []
        self.state = GameState(
            board=HexGrid(),
            placement_counts={Player.RANGER_A: 0, Player.RANGER_B: 0},
            current_player=Player.RANGER_A,
            phase="placement",
        )
        return self.state

    def step(self, action) -> GameState:
        if self.state.done:
            return self.state
        self.history.append(self.state)
        try:
            if self.state.phase == "placement":
                next_state = apply_placement(self.state, action)
            else:
                next_state = self.state.apply_action(action)
        except Exception:
            self.history.pop()
            raise

        done, winner = is_terminal(next_state) if next_state.phase == "movement" else (False, None)
        if next_state.phase == "movement" and next_state.turn_count >= self.max_turns:
            done, winner = True, None
        self.state = next_state.with_updates(done=done, winner=winner, draw=done and winner is None)
        if not self.state.done and self.state.phase == "movement" and not get_all_legal_actions(self.state):
            self.state = self.state.with_updates(done=True, winner=self.state.current_player.opponent())
        return self.state

    def undo(self) -> GameState:
        if self.history:
            self.state = self.history.pop()
        return self.state

    def play_episode(self, agent_a: Agent, agent_b: Agent) -> GameState:
        self.reset()
        agents = {Player.RANGER_A: agent_a, Player.RANGER_B: agent_b}
        guard = 0
        while not self.state.done and guard < self.max_turns + 40:
            action = agents[self.state.current_player].choose_action(self.state)
            if action is None:
                self.state = self.state.with_updates(done=True, winner=self.state.current_player.opponent())
                break
            self.step(action)
            guard += 1
        if not self.state.done:
            self.state = self.state.with_updates(done=True, draw=True, winner=None)
        return self.state

    def render(self) -> str:
        lines: list[str] = []
        piece_by_hex = {piece.position: piece for piece in self.state.pieces}
        for r in range(self.state.board.size):
            indent = " " * r
            cells: list[str] = []
            for q in range(self.state.board.size):
                hex_ = Hex(q, r)
                terrain = self.state.board.terrain_at(hex_)
                piece = piece_by_hex.get(hex_)
                if piece:
                    cell = f"{piece.owner.label}{piece.piece_type.label}"
                elif terrain == HexType.THICKET:
                    cell = "##"
                elif terrain == HexType.WATERING_HOLE:
                    cell = "~~"
                elif hex_ in self.state.claims:
                    cell = f"{self.state.claims[hex_].label}."
                else:
                    cell = ".."
                cells.append(cell)
            lines.append(indent + " ".join(cells))
        status = f"phase={self.state.phase} player={self.state.current_player.label} turn={self.state.turn_count}"
        if self.state.done:
            status += f" winner={self.state.winner.label if self.state.winner else 'draw'}"
        return status + "\n" + "\n".join(lines)
