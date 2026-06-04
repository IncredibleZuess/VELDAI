from veld.core.game import Game, RandomAgent, get_legal_placements


def test_random_game_reaches_terminal():
    final = Game(seed=1).play_episode(RandomAgent(seed=2), RandomAgent(seed=3))
    assert final.done
    assert final.turn_count <= 200


def test_placements_are_back_rows_and_not_empty():
    game = Game(seed=1)
    placements = get_legal_placements(game.state.current_player, game.state)
    assert placements
    assert all(action.target_hex.r in {0, 1} for action in placements)
