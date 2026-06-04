"""VELD package."""

__version__ = "0.1.0"

try:
    from gymnasium.envs.registration import register

    register(
        id="Veld-v0",
        entry_point="veld.env.veld_env:VeldEnv",
    )
except Exception:
    # Gymnasium is optional at import time for core/game-only usage.
    pass
