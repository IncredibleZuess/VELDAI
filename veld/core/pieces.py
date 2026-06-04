from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from veld.core.board import Hex


class PieceType(str, Enum):
    LION = "lion"
    LEOPARD = "leopard"
    IMPALA = "impala"
    EAGLE = "eagle"

    @property
    def label(self) -> str:
        return {
            PieceType.LION: "L",
            PieceType.LEOPARD: "P",
            PieceType.IMPALA: "I",
            PieceType.EAGLE: "E",
        }[self]


class Player(str, Enum):
    RANGER_A = "ranger_a"
    RANGER_B = "ranger_b"

    def opponent(self) -> "Player":
        return Player.RANGER_B if self == Player.RANGER_A else Player.RANGER_A

    @property
    def label(self) -> str:
        return "A" if self == Player.RANGER_A else "B"


@dataclass(frozen=True)
class Piece:
    piece_type: PieceType
    owner: Player
    position: Hex
