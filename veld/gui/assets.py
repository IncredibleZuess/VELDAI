from __future__ import annotations

from veld.core.pieces import Player

COLORS = {
    "background": (30, 36, 30),
    "hud": (24, 28, 24),
    "text": (236, 232, 214),
    "muted": (162, 166, 149),
    "veld": (89, 131, 76),
    "water": (58, 137, 172),
    "thicket": (61, 57, 48),
    "grid": (31, 44, 28),
    "selected": (255, 225, 92),
    "legal": (241, 202, 82),
    "last": (252, 146, 73),
    "ranger_a": (233, 202, 96),
    "ranger_b": (113, 178, 224),
    "claim_a": (170, 135, 45),
    "claim_b": (60, 127, 175),
    "danger": (211, 79, 71),
}


def player_color(player: Player) -> tuple[int, int, int]:
    return COLORS["ranger_a"] if player == Player.RANGER_A else COLORS["ranger_b"]


def claim_color(player: Player) -> tuple[int, int, int]:
    return COLORS["claim_a"] if player == Player.RANGER_A else COLORS["claim_b"]
