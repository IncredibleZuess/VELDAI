from __future__ import annotations

import csv
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
ASSET_DIR = REPORT_DIR / "assets"
DATA_DIR = REPORT_DIR / "data"
OUT = REPORT_DIR / "VELD_AI_Presentation.pptx"


BG = RGBColor(30, 36, 30)
PANEL = RGBColor(244, 241, 229)
TEXT = RGBColor(36, 42, 34)
MUTED = RGBColor(92, 99, 84)
ACCENT = RGBColor(65, 129, 79)
ACCENT_2 = RGBColor(52, 117, 153)
GOLD = RGBColor(218, 172, 62)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def set_text(frame, paragraphs: list[str], font_size: int = 20, color: RGBColor = TEXT, bold_first: bool = False) -> None:
    frame.clear()
    for i, text in enumerate(paragraphs):
        p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        p.text = text
        p.level = 0
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.font.name = "Aptos"
        p.font.bold = bold_first and i == 0
        p.space_after = Pt(8)


def add_title(slide, title: str, subtitle: str | None = None) -> None:
    box = slide.shapes.add_textbox(Inches(0.45), Inches(0.25), Inches(12.4), Inches(0.65))
    tf = box.text_frame
    set_text(tf, [title], font_size=30, color=PANEL, bold_first=True)
    if subtitle:
        sub = slide.shapes.add_textbox(Inches(0.48), Inches(0.88), Inches(12.0), Inches(0.35))
        set_text(sub.text_frame, [subtitle], font_size=13, color=RGBColor(205, 214, 196))


def add_footer(slide, number: int) -> None:
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(7.22), Inches(13.34), Inches(0.28))
    line.fill.solid()
    line.fill.fore_color.rgb = RGBColor(22, 27, 22)
    line.line.fill.background()
    box = slide.shapes.add_textbox(Inches(11.85), Inches(7.23), Inches(1.0), Inches(0.2))
    tf = box.text_frame
    set_text(tf, [f"{number}"], font_size=9, color=RGBColor(190, 200, 183))
    tf.paragraphs[0].alignment = PP_ALIGN.RIGHT


def add_panel(slide, x, y, w, h, fill=PANEL):
    panel = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    panel.fill.solid()
    panel.fill.fore_color.rgb = fill
    panel.line.color.rgb = RGBColor(216, 213, 199)
    return panel


def add_bullets(slide, x, y, w, h, bullets: list[str], size: int = 18, color: RGBColor = TEXT) -> None:
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.clear()
    for i, bullet in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = bullet
        p.font.size = Pt(size)
        p.font.name = "Aptos"
        p.font.color.rgb = color
        p.level = 0
        p.space_after = Pt(9)


def add_stat(slide, x, y, label: str, value: str, color: RGBColor = ACCENT) -> None:
    add_panel(slide, x, y, 2.45, 0.95, RGBColor(255, 253, 245))
    value_box = slide.shapes.add_textbox(Inches(x + 0.1), Inches(y + 0.08), Inches(2.2), Inches(0.45))
    set_text(value_box.text_frame, [value], font_size=24, color=color, bold_first=True)
    label_box = slide.shapes.add_textbox(Inches(x + 0.1), Inches(y + 0.55), Inches(2.2), Inches(0.28))
    set_text(label_box.text_frame, [label], font_size=10, color=MUTED)


def add_notes(slide, notes: str) -> None:
    # python-pptx does not expose speaker notes. Put concise prompt text off-slide
    # so it remains editable in PowerPoint's selection pane if needed.
    box = slide.shapes.add_textbox(Inches(13.6), Inches(0.2), Inches(4), Inches(5))
    set_text(box.text_frame, [notes], font_size=10, color=RGBColor(0, 0, 0))


def make_presentation() -> Path:
    learning = load_csv(DATA_DIR / "learning_curve.csv")
    comparison = load_csv(DATA_DIR / "agent_comparison.csv")
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    def new_slide(title: str, subtitle: str | None = None):
        slide = prs.slides.add_slide(blank)
        bg = slide.background
        bg.fill.solid()
        bg.fill.fore_color.rgb = BG
        add_title(slide, title, subtitle)
        add_footer(slide, len(prs.slides))
        return slide

    # 1
    slide = new_slide("VELD AI Agent", "ITRI616 Mini-project oral presentation")
    slide.shapes.add_picture(str(ASSET_DIR / "veld_board_gui.png"), Inches(6.65), Inches(1.2), width=Inches(5.9))
    add_bullets(
        slide,
        0.75,
        1.55,
        5.55,
        3.2,
        [
            "Digitised a local Southern African-inspired hex board game.",
            "Built an intelligent agent using reinforcement learning concepts.",
            "Goal: show that agent performance improves with experience.",
        ],
        22,
        PANEL,
    )
    add_stat(slide, 0.8, 5.45, "Board size", "7 x 7", GOLD)
    add_stat(slide, 3.45, 5.45, "Agents compared", "3", ACCENT_2)
    add_stat(slide, 6.1, 5.45, "Training episodes", "400", ACCENT)
    add_notes(slide, "Opening: explain VELD and state that the talk follows the rubric: TEP, method, results, critical thinking.")

    # 2
    slide = new_slide("Problem Definition", "What the intelligent agent must solve")
    add_panel(slide, 0.65, 1.25, 12.0, 5.55)
    add_bullets(
        slide,
        1.0,
        1.55,
        11.2,
        4.95,
        [
            "Agent role: play as Ranger A and select legal placement and movement actions.",
            "Game objective: capture the opponent's Lion or win by territory claims before the 200-turn cap.",
            "Problem type: sequential decision making and control, not classification.",
            "Challenge: actions have delayed consequences and the legal action set changes after every move.",
            "State source: immutable GameState containing terrain, pieces, claims, current player, phase, and terminal status.",
        ],
        20,
    )
    add_notes(slide, "Emphasise why this is a learning problem: the agent needs experience to prefer actions that improve future outcomes.")

    # 3
    slide = new_slide("TEP Framework", "Task, Experience, Performance")
    labels = ["Task", "Experience", "Performance"]
    texts = [
        "Choose legal actions that maximise win probability in VELD.",
        "Simulated self-generated episodes against RandomAgent using the implemented rules engine.",
        "Measured with win rate, draw/loss rate, shaped reward, average turns, and comparison against baselines.",
    ]
    for i, (label, text) in enumerate(zip(labels, texts)):
        x = 0.75 + i * 4.15
        add_panel(slide, x, 1.45, 3.65, 4.65, RGBColor(255, 253, 245))
        title = slide.shapes.add_textbox(Inches(x + 0.25), Inches(1.75), Inches(3.1), Inches(0.45))
        set_text(title.text_frame, [label], font_size=28, color=[ACCENT, ACCENT_2, GOLD][i], bold_first=True)
        add_bullets(slide, x + 0.25, 2.45, 3.1, 2.8, [text], 18)
    add_notes(slide, "This slide directly answers the 4 marks for problem definition and TEP framework.")

    # 4
    slide = new_slide("Technical Methodology", "Engine, GUI, environment, agents")
    add_panel(slide, 0.65, 1.2, 5.8, 5.7)
    add_bullets(
        slide,
        0.95,
        1.55,
        5.2,
        4.95,
        [
            "Core engine: board.py, pieces.py, rules.py, state.py, game.py.",
            "Immutable transitions prevent rollout corruption.",
            "Pygame GUI renders state and calls core legal actions only.",
            "Gymnasium wrapper exposes 7x7x8 observations and action masks.",
            "MCTS baseline uses UCB1 and random rollouts.",
        ],
        17,
    )
    slide.shapes.add_picture(str(ASSET_DIR / "veld_board_gui.png"), Inches(6.85), Inches(1.35), width=Inches(5.7))
    add_notes(slide, "Explain modularity and why no rule logic is duplicated in the GUI.")

    # 5
    slide = new_slide("Learning Algorithm", "Reinforcement learning experiment used for the report")
    add_panel(slide, 0.65, 1.2, 12.0, 5.7)
    add_bullets(
        slide,
        1.0,
        1.55,
        11.2,
        4.95,
        [
            "Implemented PPO scaffold for the full RL environment; report experiment uses a lightweight reproducible RL learner.",
            "Policy: epsilon-greedy linear scoring over legal actions.",
            "Features: capture value, claim value, pressure on opposing Lion, and centrality.",
            "Reward: +1 win, -1 loss, 0 draw, plus small claim-difference shaping.",
            "Update: episode reward adjusts feature weights; epsilon decays from exploration toward exploitation.",
        ],
        19,
    )
    add_notes(slide, "Be clear that this is reinforcement learning because actions are updated from rewards after simulated experience.")

    # 6
    slide = new_slide("Training Results", "Does performance improve with experience?")
    slide.shapes.add_picture(str(ASSET_DIR / "learning_curve.png"), Inches(0.65), Inches(1.25), width=Inches(6.2))
    slide.shapes.add_picture(str(ASSET_DIR / "reward_curve.png"), Inches(6.95), Inches(1.25), width=Inches(5.85))
    final = learning[-1]
    add_stat(slide, 1.05, 6.05, "Final block win rate", f"{float(final['win_rate']) * 100:.0f}%", ACCENT)
    add_stat(slide, 3.7, 6.05, "Final epsilon", f"{float(final['epsilon']):.2f}", ACCENT_2)
    add_stat(slide, 6.35, 6.05, "Final avg reward", f"{float(final['avg_reward']):.2f}", GOLD)
    add_notes(slide, "Point out that results are noisy but show stronger final performance and reduced exploration.")

    # 7
    slide = new_slide("What the Agent Learned", "Policy weights after experience")
    slide.shapes.add_picture(str(ASSET_DIR / "policy_weights.png"), Inches(0.95), Inches(1.25), width=Inches(7.0))
    add_panel(slide, 8.35, 1.45, 4.15, 4.85)
    add_bullets(
        slide,
        8.65,
        1.75,
        3.55,
        4.2,
        [
            "Pressure and centrality weights increased the most.",
            "The agent learned to move toward tactically important board areas.",
            "Capture and claim weights also increased, but more gradually.",
            "This supports the claim that experience changed policy behaviour.",
        ],
        17,
    )
    add_notes(slide, "Use this for technical understanding: weights make learning interpretable.")

    # 8
    slide = new_slide("Baseline Comparison", "Performance over 50 games per agent")
    slide.shapes.add_picture(str(ASSET_DIR / "agent_comparison.png"), Inches(0.75), Inches(1.25), width=Inches(6.85))
    add_panel(slide, 8.0, 1.35, 4.55, 5.25)
    rows = {r["agent"]: r for r in comparison}
    bullets = [
        f"RandomAgent win rate: {float(rows['Random']['win_rate']) * 100:.0f}%",
        f"Learning agent win rate: {float(rows['Learning agent']['win_rate']) * 100:.0f}%",
        f"MCTS baseline win rate: {float(rows['MCTS baseline']['win_rate']) * 100:.0f}%",
        "Interpretation: learned policy beats random play, while MCTS remains stronger because it searches future states.",
    ]
    add_bullets(slide, 8.3, 1.75, 3.95, 4.35, bullets, 18)
    add_notes(slide, "This addresses results presentation and analysis: compare against baselines, not just training curve.")

    # 9
    slide = new_slide("Critical Analysis", "Assumptions, limitations, and validity")
    add_panel(slide, 0.65, 1.2, 12.0, 5.7)
    add_bullets(
        slide,
        1.0,
        1.5,
        11.1,
        5.0,
        [
            "Well-posed: explicit state, legal actions, reward, experience source, and quantitative metrics.",
            "Improvement: final learned agent outperformed random baseline in win rate and loss avoidance.",
            "Assumptions: simulated random opponents are a useful learning source; reward shaping reflects useful strategy.",
            "Limitations: small training budget, simple linear policy, high draw rate, and no human gameplay dataset.",
            "Future work: train the PPO pipeline longer, evaluate against MCTS at multiple strengths, tune rewards, and add human play logs.",
        ],
        19,
    )
    add_notes(slide, "Prepare to answer whether the problem is well-posed and whether improvement is convincing.")

    # 10
    slide = new_slide("Communication Plan", "How the presentation is organised")
    add_panel(slide, 0.8, 1.3, 11.75, 5.5)
    add_bullets(
        slide,
        1.1,
        1.65,
        11.0,
        4.8,
        [
            "1. Define VELD and the learning problem.",
            "2. Explain the TEP framework.",
            "3. Show the implementation architecture and learning method.",
            "4. Present training and comparison graphs.",
            "5. Critically evaluate assumptions, limitations, and next steps.",
        ],
        22,
    )
    add_notes(slide, "This slide helps satisfy communication and organisation marks.")

    # 11
    slide = new_slide("Questions and Critical Thinking", "Prepared responses")
    add_panel(slide, 0.65, 1.2, 12.0, 5.7)
    add_bullets(
        slide,
        1.0,
        1.45,
        11.25,
        5.15,
        [
            "Why reinforcement learning? The agent learns action choices from delayed game rewards in a sequential environment.",
            "Why action masks? They prevent the learner from wasting probability on impossible moves.",
            "Why MCTS? It is a strong planning baseline that does not require training data.",
            "Did performance improve? Yes: learned agent achieved 46% wins vs 10% for random in the comparison run.",
            "What is the biggest weakness? The report experiment is lightweight; PPO should be trained longer for a stronger final agent.",
        ],
        18,
    )
    add_notes(slide, "Use this as backup if asked about methodology, validity, or limitations.")

    # 12
    slide = new_slide("Conclusion", "Rubric checklist")
    add_panel(slide, 0.75, 1.25, 11.8, 5.55)
    add_bullets(
        slide,
        1.1,
        1.6,
        11.0,
        4.9,
        [
            "Problem definition and TEP: VELD framed as sequential decision making.",
            "Methodology: modular game engine, legal-action API, RL learner, MCTS baseline.",
            "Results: graphs show learning, reward trend, learned weights, and baseline comparison.",
            "Critical thinking: assumptions and limitations are explicit.",
            "Main conclusion: experience improved agent performance, but search-based MCTS remains the stronger baseline.",
        ],
        20,
    )
    add_notes(slide, "Close by restating the key evidence: learned agent beats random baseline and graphs show policy improvement.")

    prs.save(OUT)
    return OUT


def main() -> None:
    missing = [
        path
        for path in [
            DATA_DIR / "learning_curve.csv",
            DATA_DIR / "agent_comparison.csv",
            ASSET_DIR / "learning_curve.png",
            ASSET_DIR / "reward_curve.png",
            ASSET_DIR / "policy_weights.png",
            ASSET_DIR / "agent_comparison.png",
            ASSET_DIR / "veld_board_gui.png",
        ]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Run scripts/generate_report.py first. Missing: {missing}")
    path = make_presentation()
    print(f"Presentation generated: {path}")


if __name__ == "__main__":
    main()
