from __future__ import annotations

from dataclasses import dataclass

from veld.core.board import AXIAL_DIRECTIONS, Hex, HexGrid, HexType
from veld.core.pieces import Piece, PieceType, Player
from veld.core.state import GameState


@dataclass(frozen=True)
class Action:
    piece_id: int
    target_hex: Hex


class MovementRule:
    pass


class CaptureRule:
    pass


CAPTURE_RANK = {
    PieceType.LION: 4,
    PieceType.LEOPARD: 3,
    PieceType.EAGLE: 2,
    PieceType.IMPALA: 1,
}


def get_hex_neighbors(hex_: Hex, grid: HexGrid) -> list[Hex]:
    return grid.neighbors(hex_)


def can_capture(attacker: Piece, defender: Piece) -> bool:
    if attacker.owner == defender.owner:
        return False
    if attacker.piece_type == PieceType.LION:
        return True
    if attacker.piece_type == PieceType.LEOPARD:
        return defender.piece_type in {PieceType.LEOPARD, PieceType.EAGLE, PieceType.IMPALA}
    if attacker.piece_type == PieceType.EAGLE:
        return defender.piece_type == PieceType.IMPALA
    if attacker.piece_type == PieceType.IMPALA:
        return defender.piece_type == PieceType.IMPALA
    return False


def _target_is_legal(piece: Piece, target: Hex, state: GameState) -> bool:
    if not state.board.contains(target) or state.board.is_blocked(target):
        return False
    _, occupant = state.piece_at(target)
    if occupant is None:
        return True
    return can_capture(piece, occupant)


def _line_moves(piece_id: int, piece: Piece, state: GameState, max_distance: int) -> list[Action]:
    moves: list[Action] = []
    for dq, dr in AXIAL_DIRECTIONS:
        for distance in range(1, max_distance + 1):
            target = Hex(piece.position.q + dq * distance, piece.position.r + dr * distance)
            if not state.board.contains(target) or state.board.is_blocked(target):
                break
            _, occupant = state.piece_at(target)
            if occupant is None:
                moves.append(Action(piece_id, target))
                continue
            if can_capture(piece, occupant):
                moves.append(Action(piece_id, target))
            break
    return moves


def _eagle_moves(piece_id: int, piece: Piece, state: GameState) -> list[Action]:
    moves: list[Action] = []
    for dq, dr in AXIAL_DIRECTIONS:
        jumped = Hex(piece.position.q + dq, piece.position.r + dr)
        target = Hex(piece.position.q + 2 * dq, piece.position.r + 2 * dr)
        if not state.board.contains(jumped) or not state.board.contains(target):
            continue
        if state.board.is_blocked(jumped) or state.board.is_blocked(target):
            continue
        _, jumped_piece = state.piece_at(jumped)
        if jumped_piece is None:
            continue
        if _target_is_legal(piece, target, state):
            moves.append(Action(piece_id, target))
    return moves


def get_legal_moves(piece: Piece, state: GameState) -> list[Action]:
    try:
        piece_id = state.pieces.index(piece)
    except ValueError:
        return []
    if state.done or state.phase != "movement" or piece.owner != state.current_player:
        return []
    if piece.piece_type == PieceType.LION:
        return _line_moves(piece_id, piece, state, 1)
    if piece.piece_type == PieceType.LEOPARD:
        return _line_moves(piece_id, piece, state, 2)
    if piece.piece_type == PieceType.IMPALA:
        return _line_moves(piece_id, piece, state, 1)
    if piece.piece_type == PieceType.EAGLE:
        return _eagle_moves(piece_id, piece, state)
    return []


def get_all_legal_actions(state: GameState) -> list[Action]:
    if state.done or state.phase != "movement":
        return []
    actions: list[Action] = []
    for piece_id, piece in enumerate(state.pieces):
        if piece.owner == state.current_player:
            actions.extend(get_legal_moves(piece, state))
    return actions


def is_terminal(state: GameState) -> tuple[bool, Player | None]:
    lion_owners = {piece.owner for piece in state.pieces if piece.piece_type == PieceType.LION}
    if Player.RANGER_A not in lion_owners:
        return True, Player.RANGER_B
    if Player.RANGER_B not in lion_owners:
        return True, Player.RANGER_A

    claimable = state.board.all_claimable_hexes()
    if all(hex_ in state.claims for hex_ in claimable):
        counts = state.claim_counts()
        if counts[Player.RANGER_A] > counts[Player.RANGER_B]:
            return True, Player.RANGER_A
        if counts[Player.RANGER_B] > counts[Player.RANGER_A]:
            return True, Player.RANGER_B
        return True, None
    return False, None


def apply_action(state: GameState, action: Action) -> GameState:
    if state.done:
        return state
    legal_actions = get_all_legal_actions(state)
    if action not in legal_actions:
        raise ValueError(f"Illegal action: {action}")

    moving_piece = state.pieces[action.piece_id]
    target_idx, target_piece = state.piece_at(action.target_hex)
    new_pieces: list[Piece] = list(state.pieces)
    captured = list(state.captured)

    if target_piece is not None and target_idx is not None:
        captured.append(target_piece)
        del new_pieces[target_idx]
        move_idx = action.piece_id - (1 if target_idx < action.piece_id else 0)
    else:
        move_idx = action.piece_id

    new_pieces[move_idx] = Piece(moving_piece.piece_type, moving_piece.owner, action.target_hex)

    new_claims = dict(state.claims)
    terrain = state.board.terrain_at(action.target_hex)
    if moving_piece.piece_type == PieceType.IMPALA and terrain in {HexType.VELD, HexType.WATERING_HOLE}:
        new_claims.setdefault(action.target_hex, moving_piece.owner)

    provisional = state.with_updates(
        pieces=tuple(new_pieces),
        claims=new_claims,
        current_player=state.current_player.opponent(),
        turn_count=state.turn_count + 1,
        last_action=action,
        captured=tuple(captured),
    )
    done, winner = is_terminal(provisional)
    return provisional.with_updates(done=done, winner=winner, draw=done and winner is None)
