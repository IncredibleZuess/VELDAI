from veld.agents.mcts import MCTSAgent
from veld.core.game import Game


def test_mcts_choose_action_smoke():
    game = Game(seed=1)
    action = MCTSAgent(n_simulations=2, seed=1).choose_action(game.state)
    assert action is not None
