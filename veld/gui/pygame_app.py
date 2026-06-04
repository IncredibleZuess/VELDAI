from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pygame

from veld.gui.input import GuiController
from veld.gui.renderer import PygameRenderer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play or watch VELD in Pygame.")
    parser.add_argument("--mode", default="human_vs_human", choices=[
        "human_vs_human",
        "human_vs_random",
        "random_vs_random",
        "human_vs_mcts",
        "human_vs_ppo",
        "ppo_vs_mcts",
        "ppo_watch",
    ])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--agent", default="random", choices=["random", "mcts"])
    parser.add_argument("--checkpoint", default=None)
    return parser


def run_app(args: argparse.Namespace) -> None:
    pygame.init()
    screen = pygame.display.set_mode((args.width, args.height))
    pygame.display.set_caption("VELD")
    clock = pygame.time.Clock()
    controller = GuiController(
        mode=args.mode,
        seed=args.seed,
        watch_opponent_type=args.agent,
        checkpoint=args.checkpoint,
    )
    renderer = PygameRenderer(args.width, args.height)
    running = True

    while running:
        layout = renderer.layout_for(controller.state)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                controller.handle_hex_click(layout.pixel_to_hex(event.pos))
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if controller.selected_piece_id is not None:
                        controller.selected_piece_id = None
                        controller.refresh_legal_actions()
                    else:
                        running = False
                elif event.key == pygame.K_r:
                    controller.reset()
                elif event.key == pygame.K_u:
                    controller.undo()
                elif event.key == pygame.K_a:
                    controller.autoplay = not controller.autoplay
                elif event.key == pygame.K_SPACE:
                    controller.step_ai()
                elif event.key == pygame.K_s:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    renderer.save_screenshot(screen, Path("screenshots") / f"veld_{timestamp}.png")

        if controller.auto_step_enabled:
            controller.step_ai()

        renderer.draw(
            screen,
            controller.state,
            selected_piece_id=controller.selected_piece_id,
            legal_actions=controller.legal_actions,
            mode=controller.mode,
        )
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_app(args)


if __name__ == "__main__":
    main(sys.argv[1:])
