from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

from veld.core.board import Hex, HexGrid
from veld.core.pieces import Piece, Player


@dataclass(frozen=True)
class GameState:
    board: HexGrid = field(default_factory=HexGrid)
    pieces: tuple[Piece, ...] = field(default_factory=tuple)
    claims: Mapping[Hex, Player] = field(default_factory=dict)
    current_player: Player = Player.RANGER_A
    turn_count: int = 0
    done: bool = False
    winner: Player | None = None
    phase: str = "placement"
    placement_counts: Mapping[Player, int] = field(default_factory=dict)
    last_action: object | None = None
    captured: tuple[Piece, ...] = field(default_factory=tuple)
    draw: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "pieces", tuple(self.pieces))
        object.__setattr__(self, "claims", dict(self.claims))
        counts = {
            Player.RANGER_A: int(self.placement_counts.get(Player.RANGER_A, 0)),
            Player.RANGER_B: int(self.placement_counts.get(Player.RANGER_B, 0)),
        }
        object.__setattr__(self, "placement_counts", counts)

    def piece_at(self, hex_: Hex) -> tuple[int, Piece] | tuple[None, None]:
        for idx, piece in enumerate(self.pieces):
            if piece.position == hex_:
                return idx, piece
        return None, None

    def pieces_for(self, player: Player) -> list[tuple[int, Piece]]:
        return [(idx, p) for idx, p in enumerate(self.pieces) if p.owner == player]

    def claim_counts(self) -> dict[Player, int]:
        return {
            Player.RANGER_A: sum(1 for owner in self.claims.values() if owner == Player.RANGER_A),
            Player.RANGER_B: sum(1 for owner in self.claims.values() if owner == Player.RANGER_B),
        }

    def with_updates(self, **kwargs: object) -> "GameState":
        return replace(self, **kwargs)

    def apply_action(self, action: object) -> "GameState":
        from veld.core.rules import apply_action

        return apply_action(self, action)
