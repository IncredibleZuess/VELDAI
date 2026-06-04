from __future__ import annotations

from dataclasses import dataclass

from veld.core.board import Hex
from veld.core.game import PlacementAction, get_legal_placements, next_placement_piece
from veld.core.rules import Action, get_all_legal_actions
from veld.core.state import GameState

BOARD_CELLS = 49
MAX_PIECES = 16
MOVEMENT_ACTIONS = MAX_PIECES * BOARD_CELLS
PLACEMENT_OFFSET = MOVEMENT_ACTIONS
ACTION_SPACE_SIZE = MOVEMENT_ACTIONS + BOARD_CELLS


def hex_to_flat(hex_: Hex) -> int:
    return hex_.r * 7 + hex_.q


def flat_to_hex(index: int) -> Hex:
    return Hex(index % 7, index // 7)


@dataclass(frozen=True)
class ActionCodec:
    action_space_size: int = ACTION_SPACE_SIZE

    def encode(self, action: Action | PlacementAction) -> int:
        if isinstance(action, PlacementAction):
            return PLACEMENT_OFFSET + hex_to_flat(action.target_hex)
        return action.piece_id * BOARD_CELLS + hex_to_flat(action.target_hex)

    def decode(self, index: int, state: GameState) -> Action | PlacementAction:
        if state.phase == "placement":
            piece_type = next_placement_piece(state.current_player, state)
            if piece_type is None:
                raise ValueError("No placement piece available")
            return PlacementAction(piece_type, flat_to_hex(index - PLACEMENT_OFFSET if index >= PLACEMENT_OFFSET else index))
        piece_id = index // BOARD_CELLS
        target = flat_to_hex(index % BOARD_CELLS)
        return Action(piece_id, target)

    def legal_indices(self, state: GameState) -> list[int]:
        if state.phase == "placement":
            return [self.encode(action) for action in get_legal_placements(state.current_player, state)]
        return [self.encode(action) for action in get_all_legal_actions(state)]
