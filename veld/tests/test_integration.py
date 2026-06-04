import pytest

from veld.agents.mcts import MCTSAgent
from veld.core.game import Game, RandomAgent


def test_random_vs_random_batch_terminates():
    for idx in range(20):
        final = Game(seed=idx).play_episode(RandomAgent(seed=idx), RandomAgent(seed=idx + 1))
        assert final.done
        assert final.turn_count <= 200


def test_mcts_vs_random_smoke():
    final = Game(seed=1).play_episode(MCTSAgent(n_simulations=5, seed=1), RandomAgent(seed=2))
    assert final.done


def test_ppo_checkpoint_interface_imports_when_torch_available():
    pytest.importorskip("torch")
    from veld.train.ppo_train import VeldCNN

    model = VeldCNN(10)
    assert model.policy.out_features == 10
