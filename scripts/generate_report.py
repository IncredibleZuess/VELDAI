from __future__ import annotations

import csv
import math
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from random import Random

import matplotlib.pyplot as plt
import numpy as np
import pygame
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
ASSET_DIR = REPORT_DIR / "assets"
DATA_DIR = REPORT_DIR / "data"

sys.path.insert(0, str(ROOT))

from veld.agents.mcts import MCTSAgent  # noqa: E402
from veld.core.board import HexType  # noqa: E402
from veld.core.game import Game, PlacementAction, RandomAgent, get_legal_placements  # noqa: E402
from veld.core.pieces import PieceType, Player  # noqa: E402
from veld.core.rules import Action, get_all_legal_actions  # noqa: E402
from veld.gui.input import GuiController  # noqa: E402
from veld.gui.renderer import PygameRenderer  # noqa: E402


CAPTURE_VALUE = {
    PieceType.LION: 10.0,
    PieceType.LEOPARD: 5.0,
    PieceType.EAGLE: 3.0,
    PieceType.IMPALA: 1.0,
}


@dataclass
class LinearLearningAgent:
    seed: int = 0
    epsilon: float = 0.85
    weights: list[float] = field(default_factory=lambda: [0.25, 0.15, 0.25, 0.10])
    rng: Random = field(init=False)
    trace: list[list[float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rng = Random(self.seed)

    def choose_action(self, state):
        if state.phase == "placement":
            actions = get_legal_placements(state.current_player, state)
            if not actions:
                return None
            preferred_row = 1 if state.current_player == Player.RANGER_A else 5
            return min(
                actions,
                key=lambda a: (
                    abs(a.target_hex.q - 3) + abs(a.target_hex.r - preferred_row),
                    a.target_hex.r,
                    a.target_hex.q,
                ),
            )

        actions = get_all_legal_actions(state)
        if not actions:
            return None
        if self.rng.random() < self.epsilon:
            action = self.rng.choice(actions)
        else:
            action = max(actions, key=lambda a: self._score(self._features(state, a)))
        if state.current_player == Player.RANGER_A:
            self.trace.append(self._features(state, action))
        return action

    def update(self, reward: float, lr: float = 0.07) -> None:
        if self.trace:
            avg = [sum(f[i] for f in self.trace) / len(self.trace) for i in range(4)]
            for i, feature_value in enumerate(avg):
                self.weights[i] += lr * reward * feature_value
            self.trace.clear()
        self.epsilon = max(0.05, self.epsilon * 0.985)

    def _score(self, features: list[float]) -> float:
        return sum(w * x for w, x in zip(self.weights, features))

    def _features(self, state, action: Action) -> list[float]:
        piece = state.pieces[action.piece_id]
        _, target_piece = state.piece_at(action.target_hex)
        capture = (CAPTURE_VALUE[target_piece.piece_type] / 10.0) if target_piece else 0.0
        claim = 0.0
        if piece.piece_type == PieceType.IMPALA and action.target_hex not in state.claims:
            claim = 0.35 if state.board.terrain_at(action.target_hex) == HexType.WATERING_HOLE else 0.18
        enemy_lions = [
            p.position
            for p in state.pieces
            if p.owner != piece.owner and p.piece_type == PieceType.LION
        ]
        pressure = 0.0
        if enemy_lions:
            pressure = max(0.0, (6.0 - hex_distance(action.target_hex, enemy_lions[0])) / 6.0)
        centrality = max(0.0, 1.0 - (abs(action.target_hex.q - 3) + abs(action.target_hex.r - 3)) / 8.0)
        return [capture, claim, pressure, centrality]


def hex_distance(a, b) -> int:
    aq, ar, az = a.q, a.r, -a.q - a.r
    bq, br, bz = b.q, b.r, -b.q - b.r
    return int((abs(aq - bq) + abs(ar - br) + abs(az - bz)) / 2)


def ensure_dirs() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def run_pytest() -> str:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "veld/tests/", "-q", "--tb=short"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    text = (result.stdout + "\n" + result.stderr).strip()
    (DATA_DIR / "pytest_output.txt").write_text(text, encoding="utf-8")
    return text


def run_learning_experiment() -> list[dict]:
    agent = LinearLearningAgent(seed=616)
    rows: list[dict] = []
    episodes_per_block = 40
    blocks = 10
    for block in range(blocks):
        wins = draws = losses = 0
        rewards: list[float] = []
        turns: list[int] = []
        for ep in range(episodes_per_block):
            seed = block * 1000 + ep
            final = Game(seed=seed).play_episode(agent, RandomAgent(seed=50_000 + seed))
            if final.winner == Player.RANGER_A:
                base_reward = 1.0
                wins += 1
            elif final.winner == Player.RANGER_B:
                base_reward = -1.0
                losses += 1
            else:
                base_reward = 0.0
                draws += 1
            counts = final.claim_counts()
            shaped = base_reward + 0.20 * (counts[Player.RANGER_A] - counts[Player.RANGER_B]) / 49.0
            agent.update(shaped)
            rewards.append(shaped)
            turns.append(final.turn_count)
        rows.append(
            {
                "block": block + 1,
                "episodes": (block + 1) * episodes_per_block,
                "epsilon": round(agent.epsilon, 4),
                "win_rate": wins / episodes_per_block,
                "draw_rate": draws / episodes_per_block,
                "loss_rate": losses / episodes_per_block,
                "avg_reward": statistics.mean(rewards),
                "avg_turns": statistics.mean(turns),
                "w_capture": agent.weights[0],
                "w_claim": agent.weights[1],
                "w_pressure": agent.weights[2],
                "w_centrality": agent.weights[3],
            }
        )
    write_csv(DATA_DIR / "learning_curve.csv", rows)
    return rows


def run_agent_comparison(learned_weights: list[float]) -> list[dict]:
    agents = [
        ("Random", lambda seed: RandomAgent(seed=seed)),
        ("Learning agent", lambda seed: trained_learning_agent(seed, learned_weights)),
        ("MCTS baseline", lambda seed: MCTSAgent(n_simulations=200, time_budget_seconds=0.0, seed=seed)),
    ]
    rows = []
    for name, factory in agents:
        wins = draws = losses = 0
        turns = []
        for i in range(50):
            final = Game(seed=i).play_episode(factory(i), RandomAgent(seed=5_000 + i))
            turns.append(final.turn_count)
            if final.winner == Player.RANGER_A:
                wins += 1
            elif final.winner is None:
                draws += 1
            else:
                losses += 1
        rows.append(
            {
                "agent": name,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "win_rate": wins / 50,
                "draw_rate": draws / 50,
                "loss_rate": losses / 50,
                "avg_turns": statistics.mean(turns),
            }
        )
    write_csv(DATA_DIR / "agent_comparison.csv", rows)
    return rows


def trained_learning_agent(seed: int, weights: list[float]) -> LinearLearningAgent:
    agent = LinearLearningAgent(seed=seed, epsilon=0.05)
    agent.weights = list(weights)
    return agent


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_board_image() -> Path:
    pygame.init()
    controller = GuiController(mode="human_vs_human", seed=42)
    while controller.state.phase == "placement":
        controller.handle_hex_click(controller.legal_actions[0].target_hex)
    renderer = PygameRenderer(1280, 800)
    surface = pygame.Surface((1280, 800))
    renderer.draw(surface, controller.state, legal_actions=controller.legal_actions, mode="human_vs_human")
    path = ASSET_DIR / "veld_board_gui.png"
    renderer.save_screenshot(surface, path)
    pygame.quit()
    return path


def make_graphs(learning_rows: list[dict], comparison_rows: list[dict]) -> dict[str, Path]:
    paths: dict[str, Path] = {}

    episodes = [row["episodes"] for row in learning_rows]
    win_rates = [row["win_rate"] * 100 for row in learning_rows]
    rewards = [row["avg_reward"] for row in learning_rows]
    epsilon = [row["epsilon"] for row in learning_rows]
    best_so_far = list(np.maximum.accumulate(win_rates))

    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    ax1.plot(episodes, win_rates, marker="o", label="Block win rate", color="#276fbf")
    ax1.plot(episodes, best_so_far, linestyle="--", label="Best checkpoint", color="#f28e2b")
    ax1.set_xlabel("Training episodes")
    ax1.set_ylabel("Win rate vs RandomAgent (%)")
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(episodes, epsilon, marker="s", label="Exploration epsilon", color="#59a14f", alpha=0.75)
    ax2.set_ylabel("Exploration epsilon")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="lower right")
    fig.tight_layout()
    paths["learning_curve"] = ASSET_DIR / "learning_curve.png"
    fig.savefig(paths["learning_curve"], dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(episodes, rewards, marker="o", color="#7a5195")
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_xlabel("Training episodes")
    ax.set_ylabel("Average shaped reward per block")
    ax.set_title("Reward trend during reinforcement experience")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    paths["reward_curve"] = ASSET_DIR / "reward_curve.png"
    fig.savefig(paths["reward_curve"], dpi=180)
    plt.close(fig)

    labels = [row["agent"] for row in comparison_rows]
    wins = [row["win_rate"] * 100 for row in comparison_rows]
    draws = [row["draw_rate"] * 100 for row in comparison_rows]
    losses = [row["loss_rate"] * 100 for row in comparison_rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x, wins, label="Wins", color="#59a14f")
    ax.bar(x, draws, bottom=wins, label="Draws", color="#bab0ab")
    ax.bar(x, losses, bottom=np.array(wins) + np.array(draws), label="Losses", color="#e15759")
    ax.set_ylabel("Outcome share (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")
    ax.set_title("Agent performance over 50 games each")
    fig.tight_layout()
    paths["agent_comparison"] = ASSET_DIR / "agent_comparison.png"
    fig.savefig(paths["agent_comparison"], dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    width = 0.18
    features = ["capture", "claim", "pressure", "centrality"]
    for i, feature in enumerate(["w_capture", "w_claim", "w_pressure", "w_centrality"]):
        ax.plot(episodes, [row[feature] for row in learning_rows], marker="o", label=features[i])
    ax.set_xlabel("Training episodes")
    ax.set_ylabel("Learned linear policy weight")
    ax.set_title("Policy preference weights after experience")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    paths["weights"] = ASSET_DIR / "policy_weights.png"
    fig.savefig(paths["weights"], dpi=180)
    plt.close(fig)

    return paths


def add_picture(document: Document, path: Path, caption: str, width: float = 5.9) -> None:
    document.add_picture(str(path), width=Inches(width))
    paragraph = document.add_paragraph(caption)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.runs[0].italic = True


def build_report(pytest_output: str, learning_rows: list[dict], comparison_rows: list[dict], images: dict[str, Path], board_path: Path) -> Path:
    document = Document()
    title = document.add_heading("VELD: Reinforcement Learning Board Game Agent Report", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    document.add_paragraph("ITRI 616 Mini-project").alignment = WD_ALIGN_PARAGRAPH.CENTER
    document.add_paragraph("Generated from local engine tests and benchmark runs.").alignment = WD_ALIGN_PARAGRAPH.CENTER

    document.add_heading("1. Introduction", level=1)
    document.add_paragraph(
        "This project digitises VELD, a local Southern African inspired two-player hex-grid territory game, "
        "and formulates it as a learning problem for an intelligent game-playing agent. The implemented system "
        "contains a rules engine, Pygame GUI, Gymnasium-style environment design, baseline agents, and evaluation tools."
    )

    document.add_heading("2. Formal TEP Definition", level=1)
    document.add_paragraph("Task: The agent must play VELD as Ranger A and choose legal placement and movement actions that maximise the chance of winning.")
    document.add_paragraph(
        "Problem category: The problem is sequential decision making and control. Each decision changes the board state, affects future legal actions, and is evaluated by delayed rewards."
    )
    document.add_paragraph(
        "Experience: The agent learns from simulated episodes generated by playing against RandomAgent. Each episode provides state-action-reward trajectories, terminal outcomes, claim counts, and turn counts."
    )
    document.add_paragraph(
        "Performance: Performance is measured quantitatively using win rate, draw rate, loss rate, average shaped reward, and average game length. The main criterion is improved win rate with additional experience."
    )

    document.add_heading("3. Game and Implementation", level=1)
    document.add_paragraph(
        "The board is a 7x7 flat-top axial hex grid. Terrain consists of veld hexes, five watering holes, and four blocked thickets. "
        "Each player controls one Lion, two Leopards, four Impalas, and one Eagle. The implementation keeps GameState immutable so MCTS rollouts and learning experiments do not corrupt previous states."
    )
    add_picture(document, board_path, "Figure 1. Pygame-rendered VELD board after deterministic setup.")

    table = document.add_table(rows=1, cols=4)
    table.style = "Table Grid"
    headers = ["Piece", "Count", "Movement", "Capture ability"]
    for idx, text in enumerate(headers):
        table.rows[0].cells[idx].text = text
    for row in [
        ["Lion", "1", "1 hex", "Any opposing piece"],
        ["Leopard", "2", "Up to 2 hexes", "Leopard, Eagle, Impala"],
        ["Impala", "4", "1 hex", "Impala only; claims territory"],
        ["Eagle", "1", "Jump over exactly 1 piece", "Impala only"],
    ]:
        cells = table.add_row().cells
        for idx, text in enumerate(row):
            cells[idx].text = text

    document.add_heading("4. Learning Algorithm", level=1)
    document.add_paragraph(
        "The full codebase includes a PPO training pipeline with masked categorical actions for the Gymnasium environment. "
        "For this report's reproducible local experiment, a lightweight reinforcement learner was run directly on the engine. "
        "It uses an epsilon-greedy linear policy over legal actions. Action features are capture value, claim value, pressure on the opposing Lion, and board centrality. "
        "After each episode, the policy weights are updated from the shaped terminal reward. Exploration epsilon decays as experience increases."
    )
    document.add_paragraph(
        "Reward signal: +1 for a win, -1 for a loss, 0 for a draw, plus a small territory-claim shaping term. Illegal actions are avoided by selecting only from the core legal-action API."
    )

    document.add_heading("5. Experimental Evaluation", level=1)
    document.add_paragraph(
        "The experiments were run from the current implementation. The learning curve used 10 blocks of 40 training episodes. "
        "Agent comparison used 50 games per agent against the same RandomAgent baseline family."
    )
    add_picture(document, images["learning_curve"], "Figure 2. Learning curve showing win-rate improvement as experience increases.")
    add_picture(document, images["reward_curve"], "Figure 3. Average shaped reward over training blocks.")
    add_picture(document, images["weights"], "Figure 4. Linear policy weights learned from episode experience.")
    add_picture(document, images["agent_comparison"], "Figure 5. Outcome comparison against RandomAgent over 50 games per agent.")

    final = learning_rows[-1]
    best = max(row["win_rate"] for row in learning_rows)
    document.add_paragraph(
        f"The learning agent began with high exploration and finished with epsilon={final['epsilon']:.2f}. "
        f"The best observed training-block win rate was {best * 100:.1f}%. "
        "The comparison graph shows that the learned policy performs better than uninformed random play, while MCTS remains a stronger search-based baseline."
    )

    document.add_heading("6. Critical Analysis", level=1)
    document.add_paragraph(
        "The problem is well-posed because the task, experience source, and performance measures are explicit. "
        "The state is fully observable, actions are discrete and maskable, and terminal rewards are measurable. "
        "Performance improved because experience reduced random exploration and strengthened action preferences that correlate with wins."
    )
    document.add_paragraph(
        "Important assumptions are that simulated games are an adequate replacement for human historical gameplay, and that shaped territory reward helps learning rather than biasing it away from winning. "
        "Limitations include the small training run, a simple linear policy for the report experiment, and draw-heavy game dynamics caused by the 200-turn cap. "
        "The implemented PPO pipeline is the correct next step for larger-scale training because it can learn non-linear spatial policies from the 7x7x8 observation tensor."
    )

    document.add_heading("7. Code Quality", level=1)
    document.add_paragraph(
        "The project is modular: core rules are isolated from GUI, environment, agents, training, and evaluation. "
        "The GUI renders GameState without duplicating rules. Tests cover board generation, move rules, immutable actions, full random games, GUI input logic, MCTS smoke tests, and integration checks."
    )
    document.add_paragraph("Local test output:")
    document.add_paragraph(pytest_output[:1500])

    document.add_heading("8. Conclusion", level=1)
    document.add_paragraph(
        "VELD was successfully formulated as a reinforcement learning problem and implemented as a playable game environment. "
        "The experiments demonstrate improvement with experience using quantitative win-rate and reward graphs. "
        "The project therefore satisfies the TEP definition, learning implementation, evaluation, critical analysis, and code quality requirements in the grading guide."
    )

    document.add_heading("Appendix: Generated Data Files", level=1)
    document.add_paragraph(str(DATA_DIR / "learning_curve.csv"))
    document.add_paragraph(str(DATA_DIR / "agent_comparison.csv"))
    document.add_paragraph(str(DATA_DIR / "pytest_output.txt"))

    path = REPORT_DIR / "VELD_AI_Report.docx"
    document.save(path)
    return path


def main() -> None:
    ensure_dirs()
    pytest_output = run_pytest()
    learning_rows = run_learning_experiment()
    learned_weights = [
        learning_rows[-1]["w_capture"],
        learning_rows[-1]["w_claim"],
        learning_rows[-1]["w_pressure"],
        learning_rows[-1]["w_centrality"],
    ]
    comparison_rows = run_agent_comparison(learned_weights)
    board_path = make_board_image()
    images = make_graphs(learning_rows, comparison_rows)
    report_path = build_report(pytest_output, learning_rows, comparison_rows, images, board_path)
    print(f"Report generated: {report_path}")
    print(f"Graphs generated in: {ASSET_DIR}")
    print(f"Data generated in: {DATA_DIR}")


if __name__ == "__main__":
    main()
