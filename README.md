# VELD

VELD is a two-player South African-themed hex territory board game with:

- immutable core game state and rules
- Pygame human/agent GUI
- Gymnasium RL environment with action masks
- MCTS baseline agent
- PPO training and Elo evaluation scripts

## Rules Summary

The board is a 7x7 flat-top axial hex grid with veld, watering holes, and blocked thickets. Ranger A and Ranger B each place:

| Piece | Count | Movement | Capture |
|---|---:|---|---|
| Lion | 1 | 1 hex | any piece |
| Leopard | 2 | up to 2 hexes | Leopard, Eagle, Impala |
| Impala | 4 | 1 hex | Impala only; claims territory on landing |
| Eagle | 1 | jumps over exactly 1 piece | Impala only |

The game ends when a lion is captured, all claimable hexes are claimed, or the 200-turn cap is reached.

## Install

```bash
pip install -e .
```

## Quickstart

```bash
python scripts/play_random.py
python scripts/play_gui.py --mode human_vs_human --seed 42
python scripts/play_gui.py --mode random_vs_random --seed 42
python scripts/mcts_vs_random.py
python scripts/train_ppo.py --steps 50000 --batch_size 256
python scripts/eval.py --checkpoints checkpoints/ --n_games 100
```

## Architecture

```text
veld.core        board, pieces, immutable state, rules, setup, game loop
veld.gui         Pygame renderer and mouse/keyboard controller
veld.env         Gymnasium Veld-v0 wrapper and flat action codec
veld.agents      RandomAgent and MCTSAgent
veld.train       masked PPO training loop
veld.eval        tournament, Elo, and replay helpers
```

## Tests

```bash
python -m pytest veld/tests/ -v --tb=short
```
