# Contributing

Keep the core game state as the source of truth. GUI, environment, MCTS, PPO, and evaluation code should call the shared legal-action and transition APIs instead of duplicating rules.

Avoid these patterns:

- mutating `GameState` in place
- sampling flat RL actions without applying `action_masks()`
- hardcoding hex neighbors instead of using axial directions
- allowing episodes to run beyond the 200-turn cap
- adding circular imports between `core`, `env`, `gui`, and `agents`
- storing GUI-only state as the authoritative game state

Use `python -m pytest veld/tests/ -v --tb=short` before relying on a change.
