# VELD — RL Board Game Project Blueprint

**Objective:** Build the VELD board game engine, a Pygame GUI for human/agent play, a Gymnasium-compatible RL environment, and a self-play PPO/MCTS training pipeline from scratch.

**Mode:** Direct (no git/gh assumed — edit-in-place)
**Estimated steps:** 9
**Parallelism:** Steps 4, 5, and 6 can run in parallel after Step 3 is complete. Step 7 requires Steps 5 and 6. Step 9 requires all prior steps.

---

## Dependency Graph

```
Step 1 → Step 2 → Step 3 ─┬─→ Step 4 ───────────────┐
                            ├─→ Step 5 ─→ Step 7 ─→ Step 8 ─→ Step 9
                            └─→ Step 6 ───────────┘

Step 4 = Pygame GUI
Step 5 = Gymnasium Env
Step 6 = MCTS Agent
```

---

## Step 1 — Project Scaffold & Data Model

**Model tier:** Default
**Depends on:** Nothing
**Produces:** `veld/`, `veld/core/`, `pyproject.toml`, base data classes

### Context Brief
This is the first step of a multi-session project to build VELD, a South African-themed two-player hex-grid territory board game, and an RL training pipeline on top of it. There is no existing code. Start from scratch.

### Task List
- [ ] Create project directory structure:
  ```
  veld/
    core/
      __init__.py
      board.py       # HexGrid, Hex, HexType
      pieces.py      # Piece, PieceType, Player enum
      rules.py       # CaptureRule, MovementRule
      state.py       # GameState dataclass
    gui/
      __init__.py
      pygame_app.py  # interactive Pygame GUI
      assets.py      # generated/drawn board colors, labels, sprites
      renderer.py    # board geometry + drawing helpers
      input.py       # mouse/keyboard input mapping
    env/
      __init__.py
    agents/
      __init__.py
    tests/
      __init__.py
  pyproject.toml
  README.md
  ```
- [ ] Implement `Hex` (axial coordinates `q, r`) and `HexGrid` (7×7 flat-top hex grid) in `board.py`
- [ ] Hard-code the 5 watering hole positions and 4 thicket positions as constants
- [ ] Implement `PieceType` enum: `LION, LEOPARD, IMPALA, EAGLE`
- [ ] Implement `Player` enum: `RANGER_A, RANGER_B`
- [ ] Implement `Piece` dataclass: `piece_type, owner, position`
- [ ] Implement `GameState` dataclass: `board, pieces, claims, current_player, turn_count, done, winner`
- [ ] Write `pyproject.toml` with deps: `numpy`, `gymnasium`, `torch`, `pygame`, `pytest`, `stable-baselines3`

### Verification
```bash
python -c "from veld.core.board import HexGrid; g = HexGrid(); print(len(g.hexes))"
# Expected: 49
python -m pytest veld/tests/ -q
```

### Exit Criteria
- `HexGrid` correctly generates 49 hexes (7×7)
- Watering holes and thickets are addressable by coordinate
- All dataclasses instantiate without error

---

## Step 2 — Movement & Capture Rules Engine

**Model tier:** Default
**Depends on:** Step 1
**Produces:** `veld/core/rules.py` fully implemented, `veld/core/state.py` with `apply_action()`

### Context Brief
VELD is a two-player hex-grid board game. The data model (HexGrid, Piece, GameState) was built in Step 1. This step implements the rules engine: legal move generation, capture resolution, and state transition.

**Movement rules:**
- **Lion** (1 per player): moves 1 hex in any direction; captures any piece
- **Leopard** (2 per player): moves up to 2 hexes; captures same-tier or lower (not Lion)
- **Impala** (4 per player): moves 1 hex; claims territory on landing; captures Impala only
- **Eagle** (1 per player): jumps over exactly 1 piece to the hex beyond; captures Impala only; cannot be captured by Impala

**Capture hierarchy:** Lion > Leopard > Eagle > Impala (Eagle captures Impalas, Impalas cannot capture Eagles)

**Claiming:** An Impala that lands on neutral Veld hex claims it for its owner.

### Task List
- [ ] Implement `get_hex_neighbors(hex, grid)` — returns up to 6 adjacent hexes
- [ ] Implement `get_legal_moves(piece, state)` for each PieceType
- [ ] Implement `can_capture(attacker, defender)` per capture hierarchy
- [ ] Implement `apply_action(state, action) -> GameState` — returns new state (immutable transition)
- [ ] Implement `is_terminal(state) -> (bool, winner_or_None)` — Lion captured OR all Veld claimed
- [ ] Implement `Action` dataclass: `piece_id, target_hex`
- [ ] Implement `get_all_legal_actions(state) -> List[Action]`
- [ ] Unit tests: move generation for all piece types, capture resolution, terminal detection

### Verification
```bash
python -m pytest veld/tests/test_rules.py -v
```

### Exit Criteria
- All piece types generate correct legal moves
- Capture hierarchy is correctly enforced
- `apply_action` returns a new `GameState` without mutating the original
- Terminal detection correctly identifies both win conditions

---

## Step 3 — Setup Phase & Game Loop

**Model tier:** Default
**Depends on:** Step 2
**Produces:** `veld/core/game.py` with full game loop, placement phase, CLI playthrough

### Context Brief
The rules engine is complete (Step 2). This step adds the setup phase (free placement in back 2 rows) and a `Game` orchestrator class that runs a full episode between two callable agents.

### Task List
- [ ] Implement `PlacementAction` dataclass: `piece_type, target_hex`
- [ ] Implement `get_legal_placements(player, state)` — only back 2 rows, unoccupied non-thicket hexes
- [ ] Implement `apply_placement(state, placement_action) -> GameState`
- [ ] Implement `Game` class:
  - `reset()` → fresh `GameState` in placement phase
  - `step(action)` → delegates to placement or movement phase
  - `render()` → ASCII board print (for debugging)
- [ ] Implement a `RandomAgent` that selects uniformly from legal actions
- [ ] Write a CLI script `scripts/play_random.py` that runs 10 random-vs-random games and prints outcomes
- [ ] Add integration test: full random game runs to terminal state without error

### Verification
```bash
python scripts/play_random.py
# Should print 10 game outcomes with winner and turn count
python -m pytest veld/tests/ -q
```

### Exit Criteria
- Full game (placement + movement) runs to completion with two `RandomAgent`s
- No infinite loops (enforce `max_turns=200` hard cap)
- ASCII render shows board state legibly

---


## Step 4 — Pygame GUI: Human Play, Agent Play & Visual Debugging

**Model tier:** Default
**Depends on:** Step 3
**Produces:** `veld/gui/`, `scripts/play_gui.py` — a working graphical interface built with Pygame

### Context Brief
The engine can now run complete games through code and ASCII rendering. This step adds a real GUI so the project is usable and debuggable without reading terminal output. The GUI should be implemented in **Pygame** and should call the same `Game`, `GameState`, legal-action, placement, and agent APIs from the core engine. Do not duplicate the game rules inside the GUI.

The GUI is not the RL environment. It is a human-facing visual tool for:
- Human vs Human play
- Human vs RandomAgent play
- RandomAgent vs RandomAgent watch mode
- Human vs MCTSAgent once Step 6 exists
- Loading a trained PPO checkpoint for visual inspection once Step 7 exists

### UI / UX Requirements
- Show a 7×7 flat-top hex board centered in the window.
- Draw terrain clearly:
  - Veld as the default hex
  - Watering holes with a distinct icon or fill
  - Thickets as blocked hexes
  - Claimed hexes tinted/marked by player
- Draw pieces with readable labels or icons:
  - `L` = Lion
  - `P` = Leopard
  - `I` = Impala
  - `E` = Eagle
  - use ownership markings for Ranger A vs Ranger B
- Highlight:
  - current player
  - selected piece
  - legal target hexes
  - last move
  - captured pieces / removed pieces in a side panel
- Include a right-side HUD with:
  - phase: placement or movement
  - current player
  - turn count
  - score / claimed hex count per player
  - winner / draw state when terminal
  - active control mode
- Support mouse input:
  - click a piece to select it
  - click a highlighted legal hex to move
  - during placement phase, click a valid back-row hex to place the next required piece
- Support keyboard shortcuts:
  - `R` reset game
  - `U` undo last move if history is implemented
  - `A` toggle autoplay for agent-vs-agent mode
  - `Space` advance one AI move in step mode
  - `Esc` clear selection or quit menu
- Make the GUI deterministic when a seed is passed.

### Task List
- [ ] Add `pygame` to `pyproject.toml`
- [ ] Create GUI package:
  ```
  veld/gui/
    __init__.py
    pygame_app.py
    renderer.py
    input.py
    assets.py
  scripts/play_gui.py
  ```
- [ ] Implement `HexLayout` / board geometry helper:
  - axial `Hex(q, r)` → pixel center
  - pixel coordinate → nearest hex
  - polygon corners for flat-top hex drawing
- [ ] Implement `PygameRenderer`:
  - draw board
  - draw terrain
  - draw claims
  - draw pieces
  - draw legal move highlights
  - draw HUD
- [ ] Implement `GuiController`:
  - owns a `Game`
  - stores selected piece / selected placement piece
  - translates mouse clicks into `PlacementAction` or `Action`
  - validates through existing core legal-action functions only
- [ ] Implement control modes:
  - `human_vs_human`
  - `human_vs_random`
  - `random_vs_random`
  - `human_vs_mcts` placeholder until Step 6 is complete
  - `ppo_watch` placeholder until Step 7 checkpoints exist
- [ ] Add optional CLI args to `scripts/play_gui.py`:
  - `--mode human_vs_human`
  - `--seed 42`
  - `--width 1280 --height 800`
  - `--agent random|mcts|ppo`
  - `--checkpoint path/to/model.pt`
- [ ] Add lightweight tests for non-visual GUI logic:
  - hex-to-pixel and pixel-to-hex round trip
  - click on occupied hex selects the correct piece
  - illegal click does not mutate state
  - legal click applies exactly one action
- [ ] Add a screenshot/export helper:
  - `S` saves current board view to `screenshots/veld_<timestamp>.png`

### Verification
```bash
pip install -e .
python scripts/play_gui.py --mode human_vs_human --seed 42
python scripts/play_gui.py --mode random_vs_random --seed 42

python -m pytest veld/tests/test_gui.py -v
```

### Exit Criteria
- GUI opens a Pygame window without crashing.
- A full human-vs-human game can be played using only mouse input.
- Random-vs-random watch mode can run visually to a terminal state.
- GUI never duplicates rule logic; it only calls core legal-action and state-transition APIs.
- GUI can reset cleanly and show winner/draw information.
- Non-visual GUI tests pass in headless mode.

---
## Step 5 — Gymnasium Environment Wrapper

**Model tier:** Default
**Depends on:** Step 3
**Produces:** `veld/env/veld_env.py` — a `gymnasium.Env` subclass

### Context Brief
The VELD game engine and loop are complete (Steps 1–3). This step wraps the game in a standard `gymnasium.Env` interface so any RL library (Stable-Baselines3, CleanRL, etc.) can train against it.

**Design decisions:**
- **Observation space:** `Box(shape=(7, 7, C), dtype=float32)` where C = feature channels (terrain, piece ownership, piece type, claims)
- **Action space:** `Discrete(N)` where N = max possible actions; illegal actions masked via `action_masks()`
- **Reward:** +0.1 per hex claimed, +0.2 per watering hole claimed, +1.0 for winning, −1.0 for losing, −0.5 for losing the Lion
- **Opponent:** configurable — defaults to `RandomAgent` during training

### Task List
- [ ] Implement `encode_state(state) -> np.ndarray` — shape `(7, 7, 8)` with channels:
  - terrain type (one-hot: veld/water/thicket)
  - claims (player A, player B)
  - pieces by type and owner (4 types × 2 players = 8 channels total, merge with terrain)
- [ ] Implement `VeldEnv(gymnasium.Env)`:
  - `observation_space`, `action_space`
  - `reset()`, `step(action)`, `render()`
  - `action_masks()` → `np.ndarray` boolean mask over action space
- [ ] Write `veld/env/action_codec.py` — bidirectional mapping between flat action index and `Action`/`PlacementAction`
- [ ] Add env registration: `gymnasium.register(id="Veld-v0", ...)`
- [ ] Unit test: `check_env(VeldEnv())` from `stable_baselines3.common.env_checker`

### Verification
```bash
python -c "
import gymnasium as gym
import veld
env = gym.make('Veld-v0')
obs, _ = env.reset()
print(obs.shape)  # (7, 7, 8)
obs, reward, done, trunc, info = env.step(env.action_space.sample())
print(reward, done)
"
python -m pytest veld/tests/test_env.py -v
```

### Exit Criteria
- `check_env` passes with no warnings
- Observation shape is `(7, 7, 8)`
- Action masking correctly excludes illegal actions
- Reward function matches spec

---

## Step 6 — MCTS Agent (Baseline)

**Model tier:** Strongest
**Depends on:** Step 3
**Produces:** `veld/agents/mcts.py` — a working MCTS agent usable as a strong baseline opponent

### Context Brief
The VELD game engine is complete (Steps 1–3). This step builds a Monte Carlo Tree Search agent as a strong baseline opponent for RL training. The MCTS agent must be self-contained and not depend on Step 4's Gymnasium wrapper.

**MCTS spec:**
- UCB1 selection with `C = sqrt(2)`
- Random rollout policy
- Configurable `n_simulations` (default 200)
- Compatible with `Game.step()` interface

### Task List
- [ ] Implement `MCTSNode`: `state, parent, children, visits, value, action`
- [ ] Implement `MCTSAgent`:
  - `select(node)` — UCB1 traversal
  - `expand(node)` — add one child per legal action
  - `rollout(state)` — random play to terminal
  - `backprop(node, result)` — update visits/value up to root
  - `choose_action(state, n_simulations) -> Action`
- [ ] Add `scripts/mcts_vs_random.py` — 50 games MCTS vs Random, print win rate
- [ ] Add configurable time budget alternative to `n_simulations`

### Verification
```bash
python scripts/mcts_vs_random.py
# MCTS (200 sims) should win >70% vs Random
```

### Exit Criteria
- MCTS wins >70% of games against `RandomAgent` at 200 simulations
- No crashes on terminal states
- `choose_action` completes within 5 seconds per move at 500 simulations

---

## Step 7 — PPO Training Pipeline

**Model tier:** Strongest
**Depends on:** Step 5 (env), Step 6 (MCTS baseline opponent)
**Produces:** `veld/train/ppo_train.py`, trained model checkpoint

### Context Brief
The Gymnasium environment (`Veld-v0`) and MCTS baseline agent are complete. This step builds a PPO training pipeline using **CleanRL**-style single-file implementation (no SB3 dependency) with:
- Action masking (invalid action logits → −inf before softmax)
- Self-play curriculum: starts vs `RandomAgent`, switches to `MCTSAgent(n=100)` after 500k steps, then self-play after 1M steps
- CNN feature extractor over the `(7, 7, 8)` observation

### Task List
- [ ] Implement `VeldCNN(nn.Module)`: 3-layer CNN → flatten → Linear for policy + value heads
- [ ] Implement `MaskedCategorical` distribution: applies boolean action mask before sampling
- [ ] Implement PPO update loop (clip ratio 0.2, entropy coef 0.01, value coef 0.5, GAE λ=0.95)
- [ ] Implement opponent curriculum scheduler
- [ ] Add checkpoint saving every 100k steps to `checkpoints/`
- [ ] Add TensorBoard logging: episode reward, win rate, entropy, value loss
- [ ] Write `scripts/train_ppo.py` — entry point with argparse (steps, lr, batch_size, opponent)

### Verification
```bash
python scripts/train_ppo.py --steps 50000 --batch_size 256
# Should run 50k steps without crash, produce a checkpoint
tensorboard --logdir runs/
```

### Exit Criteria
- Training runs 50k steps on CPU without OOM or crash
- Checkpoint loads and produces valid action distributions
- Win rate vs `RandomAgent` improves monotonically over first 200k steps

---

## Step 8 — Evaluation Suite & Elo Tracker

**Model tier:** Default
**Depends on:** Step 7
**Produces:** `veld/eval/`, `scripts/eval.py`, Elo rating CSV

### Context Brief
A trained PPO checkpoint exists. This step builds an evaluation harness to measure agent strength, compute Elo ratings across checkpoints, and produce a leaderboard.

### Task List
- [ ] Implement `TournamentEvaluator`:
  - Load N checkpoints + RandomAgent + MCTSAgent(200) + MCTSAgent(500)
  - Run round-robin: 200 games per pair
  - Compute Elo from results (K=32, initial=1000)
- [ ] Output `results/elo_leaderboard.csv` and print table
- [ ] Implement `scripts/eval.py` — configurable checkpoint dir, n_games, opponent list
- [ ] Add `veld/eval/replay.py` — replay a saved game state sequence with ASCII render
- [ ] Save game logs as JSON for replay: `{turns: [{action, state_hash, reward}]}`

### Verification
```bash
python scripts/eval.py --checkpoints checkpoints/ --n_games 100
# Prints Elo leaderboard; trained agent should beat Random
```

### Exit Criteria
- Elo leaderboard CSV is generated
- Trained PPO agent has higher Elo than `RandomAgent`
- Replay loads and renders a saved game correctly

---

## Step 9 — Packaging, Docs & Final Integration Test

**Model tier:** Default
**Depends on:** Steps 1–8 (all)
**Produces:** Installable package, `README.md`, full test suite passing

### Context Brief
All components of VELD are built: game engine, Pygame GUI, Gymnasium env, MCTS agent, PPO pipeline, evaluation suite. This step finalises packaging, writes user-facing documentation, and runs the full integration test suite.

### Task List
- [ ] Update `pyproject.toml`: correct version, authors, license (MIT), entry points
- [ ] Write `README.md`:
  - Game rules summary with piece table
  - Installation instructions (`pip install -e .`)
  - Quickstart: play GUI, play random vs random, train PPO, run eval
  - Architecture diagram (ASCII)
- [ ] Add `CONTRIBUTING.md` with anti-pattern warnings (see below)
- [ ] Write full integration test: `tests/test_integration.py`
  - Random vs Random: 100 games, all terminate, no crashes
  - Pygame GUI logic: headless renderer/input tests pass
  - MCTS vs Random: win rate > 60%
  - Env: `check_env` passes
  - PPO checkpoint: loads, produces legal actions on fresh board
- [ ] Run `python -m pytest veld/tests/ -v --tb=short` — all green
- [ ] Verify `pip install -e .` works in a clean venv

### Verification
```bash
pip install -e .
python -m pytest veld/tests/ -v --tb=short
# All tests green
python -c "import veld; print(veld.__version__)"
```

### Exit Criteria
- All tests pass
- Package installs cleanly
- README covers all quickstart paths
- No circular imports

---

## Anti-Pattern Catalog

Avoid these known failure modes during implementation:

| Anti-Pattern | Why It Fails | Fix |
|---|---|---|
| Mutating `GameState` in-place | Breaks MCTS rollout tree | Always return new state from `apply_action` |
| Flat action index without masking | RL agent wastes capacity on illegal actions | Always apply `action_masks()` before sampling |
| Hardcoding hex neighbors | Breaks at grid edges | Use axial coordinate arithmetic |
| Infinite game loops | PPO episodes never terminate, reward explodes | Enforce `max_turns=200` hard cap with draw result |
| Storing full state in MCTS node | Memory explosion during rollout | Store only `state_hash` + `action`; reconstruct on demand |
| Using `gym` instead of `gymnasium` | Deprecated API, SB3 3.x breaks | Use `gymnasium` throughout |
| Re-implementing rules inside Pygame GUI | GUI diverges from engine and creates hidden bugs | GUI must call core legal-action and transition APIs only |
| Pixel-only GUI state | Makes undo/replay/testing impossible | Keep `GameState` as the source of truth and render from it |

---

## Plan Mutation Protocol

If a step needs to change after execution has started:

- **Split:** Insert a new step between two existing steps; update dependency edges.
- **Skip:** Mark step as `[SKIPPED]` with reason; verify downstream steps don't depend on its outputs.
- **Reorder:** Only allowed if dependency graph still holds after reordering.
- **Abandon:** Document reason in `plans/abandoned/`; do not delete original plan file.

All mutations must be logged at the bottom of this file under `## Mutation Log`.

---

## Mutation Log

- Added Step 4 for a Pygame GUI, including board rendering, mouse controls, HUD, agent modes, screenshot export, and headless GUI tests.
- Renumbered downstream steps from Gymnasium onward and updated the dependency graph.
- Added `pygame` to the project dependencies and GUI quickstart/testing requirements.
