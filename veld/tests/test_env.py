import pytest

gymnasium = pytest.importorskip("gymnasium")

import veld  # noqa: E402,F401
from veld.env.veld_env import VeldEnv, encode_state  # noqa: E402


def test_env_reset_shape_and_mask():
    env = VeldEnv(seed=1)
    obs, info = env.reset()
    assert obs.shape == (7, 7, 8)
    assert info["action_mask"].dtype == bool
    assert info["action_mask"].any()


def test_registered_env_step():
    env = gymnasium.make("Veld-v0")
    obs, info = env.reset(seed=1)
    obs, reward, done, truncated, info = env.step(env.action_space.sample())
    assert obs.shape == (7, 7, 8)
    assert isinstance(reward, float)
