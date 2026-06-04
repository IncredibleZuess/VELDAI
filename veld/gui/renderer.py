from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import pygame

from veld.core.board import Hex, HexGrid, HexType
from veld.core.game import PlacementAction
from veld.core.pieces import Player
from veld.core.rules import Action
from veld.core.state import GameState
from veld.gui.assets import COLORS, claim_color, player_color


class HexLayout:
    def __init__(self, grid: HexGrid, width: int, height: int, hud_width: int = 320) -> None:
        self.grid = grid
        self.width = width
        self.height = height
        self.hud_width = hud_width
        board_width = max(360, width - hud_width - 80)
        board_height = max(360, height - 80)
        self.size = min(board_width / 12.0, board_height / 13.0)
        self.origin = self._center_origin(width - hud_width, height)

    def _raw_center(self, hex_: Hex) -> tuple[float, float]:
        x = self.size * 1.5 * hex_.q
        y = self.size * (math.sqrt(3) * (hex_.r + hex_.q / 2.0))
        return x, y

    def _center_origin(self, board_width: int, height: int) -> tuple[float, float]:
        centers = [self._raw_center(hex_) for hex_ in self.grid.hexes]
        min_x = min(x for x, _ in centers) - self.size
        max_x = max(x for x, _ in centers) + self.size
        min_y = min(y for _, y in centers) - self.size
        max_y = max(y for _, y in centers) + self.size
        return (
            (board_width - (max_x - min_x)) / 2 - min_x,
            (height - (max_y - min_y)) / 2 - min_y,
        )

    def hex_to_pixel(self, hex_: Hex) -> tuple[float, float]:
        x, y = self._raw_center(hex_)
        return x + self.origin[0], y + self.origin[1]

    def pixel_to_hex(self, pos: tuple[float, float]) -> Hex | None:
        x = pos[0] - self.origin[0]
        y = pos[1] - self.origin[1]
        qf = (2.0 / 3.0 * x) / self.size
        rf = (-1.0 / 3.0 * x + math.sqrt(3) / 3.0 * y) / self.size
        sf = -qf - rf
        rq, rr, rs = round(qf), round(rf), round(sf)
        q_diff = abs(rq - qf)
        r_diff = abs(rr - rf)
        s_diff = abs(rs - sf)
        if q_diff > r_diff and q_diff > s_diff:
            rq = -rr - rs
        elif r_diff > s_diff:
            rr = -rq - rs
        hex_ = Hex(int(rq), int(rr))
        return hex_ if self.grid.contains(hex_) else None

    def corners(self, hex_: Hex) -> list[tuple[int, int]]:
        cx, cy = self.hex_to_pixel(hex_)
        points: list[tuple[int, int]] = []
        for i in range(6):
            angle = math.radians(60 * i)
            points.append((int(cx + self.size * math.cos(angle)), int(cy + self.size * math.sin(angle))))
        return points


class PygameRenderer:
    def __init__(self, width: int = 1280, height: int = 800) -> None:
        self.width = width
        self.height = height
        self.hud_width = 320
        self.fonts_ready = False
        self.font = None
        self.big_font = None

    def _ensure_fonts(self) -> None:
        if not self.fonts_ready:
            pygame.font.init()
            self.font = pygame.font.SysFont("arial", 20)
            self.big_font = pygame.font.SysFont("arial", 30, bold=True)
            self.fonts_ready = True

    def layout_for(self, state: GameState) -> HexLayout:
        return HexLayout(state.board, self.width, self.height, self.hud_width)

    def draw(
        self,
        surface: pygame.Surface,
        state: GameState,
        selected_piece_id: int | None = None,
        legal_actions: Iterable[Action | PlacementAction] = (),
        mode: str = "human_vs_human",
    ) -> None:
        self._ensure_fonts()
        layout = self.layout_for(state)
        surface.fill(COLORS["background"])
        legal_targets = {action.target_hex for action in legal_actions}
        last_target = getattr(state.last_action, "target_hex", None)
        piece_by_hex = {piece.position: (idx, piece) for idx, piece in enumerate(state.pieces)}

        for hex_, terrain in state.board.hexes.items():
            color = COLORS["veld"]
            if terrain == HexType.WATERING_HOLE:
                color = COLORS["water"]
            elif terrain == HexType.THICKET:
                color = COLORS["thicket"]
            pygame.draw.polygon(surface, color, layout.corners(hex_))
            if hex_ in state.claims:
                overlay = pygame.Surface((int(layout.size * 2), int(layout.size * 2)), pygame.SRCALPHA)
                overlay.fill((*claim_color(state.claims[hex_]), 90))
                cx, cy = layout.hex_to_pixel(hex_)
                surface.blit(overlay, (cx - layout.size, cy - layout.size))
            if hex_ == last_target:
                pygame.draw.polygon(surface, COLORS["last"], layout.corners(hex_), 4)
            elif hex_ in legal_targets:
                pygame.draw.polygon(surface, COLORS["legal"], layout.corners(hex_), 4)
            else:
                pygame.draw.polygon(surface, COLORS["grid"], layout.corners(hex_), 2)

            if terrain == HexType.WATERING_HOLE:
                self._center_text(surface, "~", layout.hex_to_pixel(hex_), COLORS["text"], self.big_font)
            elif terrain == HexType.THICKET:
                self._center_text(surface, "#", layout.hex_to_pixel(hex_), COLORS["muted"], self.big_font)

        for hex_, (idx, piece) in piece_by_hex.items():
            cx, cy = layout.hex_to_pixel(hex_)
            radius = int(layout.size * 0.48)
            pygame.draw.circle(surface, player_color(piece.owner), (int(cx), int(cy)), radius)
            border = COLORS["selected"] if idx == selected_piece_id else COLORS["grid"]
            pygame.draw.circle(surface, border, (int(cx), int(cy)), radius, 4)
            self._center_text(surface, piece.piece_type.label, (cx, cy - 2), (18, 20, 18), self.big_font)
            self._center_text(surface, piece.owner.label, (cx + radius * 0.45, cy + radius * 0.45), (18, 20, 18), self.font)

        self._draw_hud(surface, state, mode)

    def _center_text(self, surface: pygame.Surface, text: str, center: tuple[float, float], color, font) -> None:
        img = font.render(text, True, color)
        rect = img.get_rect(center=(int(center[0]), int(center[1])))
        surface.blit(img, rect)

    def _draw_hud(self, surface: pygame.Surface, state: GameState, mode: str) -> None:
        x = self.width - self.hud_width
        pygame.draw.rect(surface, COLORS["hud"], pygame.Rect(x, 0, self.hud_width, self.height))
        y = 28
        rows = [
            ("VELD", self.big_font, COLORS["text"]),
            (f"phase: {state.phase}", self.font, COLORS["text"]),
            (f"current: Ranger {state.current_player.label}", self.font, COLORS["text"]),
            (f"turn: {state.turn_count}", self.font, COLORS["text"]),
            (f"mode: {mode}", self.font, COLORS["muted"]),
        ]
        counts = state.claim_counts()
        rows.extend(
            [
                (f"claims A: {counts[Player.RANGER_A]}", self.font, COLORS["ranger_a"]),
                (f"claims B: {counts[Player.RANGER_B]}", self.font, COLORS["ranger_b"]),
                (f"captured: {len(state.captured)}", self.font, COLORS["muted"]),
            ]
        )
        if state.done:
            result = f"winner: Ranger {state.winner.label}" if state.winner else "winner: draw"
            rows.append((result, self.big_font, COLORS["danger"]))
        rows.extend(
            [
                ("R reset", self.font, COLORS["muted"]),
                ("U undo", self.font, COLORS["muted"]),
                ("A autoplay", self.font, COLORS["muted"]),
                ("Space step AI", self.font, COLORS["muted"]),
                ("S screenshot", self.font, COLORS["muted"]),
                ("Esc clear/quit", self.font, COLORS["muted"]),
            ]
        )
        for text, font, color in rows:
            img = font.render(text, True, color)
            surface.blit(img, (x + 24, y))
            y += img.get_height() + 14

    def save_screenshot(self, surface: pygame.Surface, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(surface, str(path))
