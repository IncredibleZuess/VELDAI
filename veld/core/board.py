from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True, order=True)
class Hex:
    """Axial coordinate on the 7x7 VELD board."""

    q: int
    r: int

    def __str__(self) -> str:
        return f"({self.q},{self.r})"


class HexType(str, Enum):
    VELD = "veld"
    WATERING_HOLE = "watering_hole"
    THICKET = "thicket"


AXIAL_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
)

WATERING_HOLES: frozenset[Hex] = frozenset(
    {
        Hex(3, 3),
        Hex(1, 1),
        Hex(5, 1),
        Hex(1, 5),
        Hex(5, 5),
    }
)

THICKETS: frozenset[Hex] = frozenset(
    {
        Hex(3, 1),
        Hex(1, 3),
        Hex(5, 3),
        Hex(3, 5),
    }
)


class HexGrid:
    """A rectangular 7x7 flat-top axial hex grid."""

    def __init__(self, size: int = 7) -> None:
        self.size = size
        self.hexes: dict[Hex, HexType] = {}
        for q in range(size):
            for r in range(size):
                coord = Hex(q, r)
                if coord in WATERING_HOLES:
                    terrain = HexType.WATERING_HOLE
                elif coord in THICKETS:
                    terrain = HexType.THICKET
                else:
                    terrain = HexType.VELD
                self.hexes[coord] = terrain

    def contains(self, hex_: Hex) -> bool:
        return hex_ in self.hexes

    def terrain_at(self, hex_: Hex) -> HexType:
        return self.hexes[hex_]

    def is_blocked(self, hex_: Hex) -> bool:
        return self.terrain_at(hex_) == HexType.THICKET

    def neighbors(self, hex_: Hex) -> list[Hex]:
        result: list[Hex] = []
        for dq, dr in AXIAL_DIRECTIONS:
            nxt = Hex(hex_.q + dq, hex_.r + dr)
            if self.contains(nxt):
                result.append(nxt)
        return result

    def all_claimable_hexes(self) -> list[Hex]:
        return [h for h, t in self.hexes.items() if t != HexType.THICKET]
