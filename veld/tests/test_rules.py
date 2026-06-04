import pytest

from veld.core.board import Hex, HexGrid
from veld.core.pieces import Piece, PieceType, Player
from veld.core.rules import Action, apply_action, can_capture, get_all_legal_actions, get_legal_moves, is_terminal
from veld.core.state import GameState


def movement_state(*pieces, current=Player.RANGER_A):
    return GameState(board=HexGrid(), pieces=pieces, current_player=current, phase="movement")


def test_lion_moves_one_hex():
    lion = Piece(PieceType.LION, Player.RANGER_A, Hex(3, 3))
    state = movement_state(lion, Piece(PieceType.LION, Player.RANGER_B, Hex(6, 6)))
    moves = get_legal_moves(lion, state)
    assert all(abs(action.target_hex.q - 3) <= 1 and abs(action.target_hex.r - 3) <= 1 for action in moves)


def test_capture_hierarchy():
    leopard = Piece(PieceType.LEOPARD, Player.RANGER_A, Hex(0, 0))
    lion = Piece(PieceType.LION, Player.RANGER_B, Hex(1, 0))
    eagle = Piece(PieceType.EAGLE, Player.RANGER_B, Hex(1, 0))
    impala = Piece(PieceType.IMPALA, Player.RANGER_A, Hex(0, 1))
    assert not can_capture(leopard, lion)
    assert can_capture(leopard, eagle)
    assert not can_capture(impala, eagle)


def test_apply_action_is_immutable_and_impala_claims():
    impala = Piece(PieceType.IMPALA, Player.RANGER_A, Hex(0, 0))
    state = movement_state(impala, Piece(PieceType.LION, Player.RANGER_A, Hex(6, 6)), Piece(PieceType.LION, Player.RANGER_B, Hex(5, 6)))
    next_state = apply_action(state, Action(0, Hex(1, 0)))
    assert state.pieces[0].position == Hex(0, 0)
    assert next_state.pieces[0].position == Hex(1, 0)
    assert next_state.claims[Hex(1, 0)] == Player.RANGER_A


def test_terminal_when_lion_missing():
    state = movement_state(Piece(PieceType.LION, Player.RANGER_A, Hex(0, 0)))
    done, winner = is_terminal(state)
    assert done
    assert winner == Player.RANGER_A


def test_illegal_action_raises():
    state = movement_state(Piece(PieceType.LION, Player.RANGER_A, Hex(0, 0)), Piece(PieceType.LION, Player.RANGER_B, Hex(6, 6)))
    with pytest.raises(ValueError):
        apply_action(state, Action(0, Hex(6, 6)))
