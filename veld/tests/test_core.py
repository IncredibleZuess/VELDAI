from veld.core.board import Hex, HexGrid, THICKETS, WATERING_HOLES, HexType
from veld.core.pieces import Piece, PieceType, Player
from veld.core.state import GameState


def test_hex_grid_has_49_hexes_and_terrain_constants():
    grid = HexGrid()
    assert len(grid.hexes) == 49
    assert all(grid.terrain_at(hex_) == HexType.WATERING_HOLE for hex_ in WATERING_HOLES)
    assert all(grid.terrain_at(hex_) == HexType.THICKET for hex_ in THICKETS)


def test_dataclasses_instantiate():
    state = GameState(pieces=(Piece(PieceType.LION, Player.RANGER_A, Hex(0, 0)),))
    assert state.pieces[0].piece_type == PieceType.LION
    assert state.current_player == Player.RANGER_A
