import pytest

from veld.agents.mcts import MCTSAgent
from veld.core.board import Hex
from veld.core.game import RandomAgent
from veld.gui.input import GuiController
from veld.gui.renderer import HexLayout


def test_hex_pixel_round_trip():
    controller = GuiController(seed=42)
    layout = HexLayout(controller.state.board, 1280, 800)
    for hex_ in [Hex(0, 0), Hex(3, 3), Hex(6, 6)]:
        assert layout.pixel_to_hex(layout.hex_to_pixel(hex_)) == hex_


def test_legal_placement_click_applies_one_action():
    controller = GuiController(seed=42)
    before_count = len(controller.state.pieces)
    target = controller.legal_actions[0].target_hex
    assert controller.handle_hex_click(target)
    assert len(controller.state.pieces) == before_count + 1


def test_illegal_click_does_not_mutate_state():
    controller = GuiController(seed=42)
    before = controller.state
    controller.handle_hex_click(Hex(3, 3))
    assert controller.state is before


def test_click_on_occupied_hex_selects_piece_after_setup():
    controller = GuiController(seed=42)
    while controller.state.phase == "placement":
        controller.handle_hex_click(controller.legal_actions[0].target_hex)
    own_piece_id, own_piece = controller.state.pieces_for(controller.state.current_player)[0]
    changed = controller.handle_hex_click(own_piece.position)
    assert not changed
    assert controller.selected_piece_id == own_piece_id


def test_human_vs_ppo_requires_checkpoint():
    with pytest.raises(ValueError, match="checkpoint"):
        GuiController(mode="human_vs_ppo", seed=42)


def test_ppo_vs_mcts_requires_checkpoint():
    with pytest.raises(ValueError, match="checkpoint"):
        GuiController(mode="ppo_vs_mcts", seed=42)


def test_human_vs_ppo_uses_checkpoint(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.write_bytes(b"dummy")

    class DummyPPO:
        def __init__(self, checkpoint):
            self.checkpoint = checkpoint

        def choose_action(self, state):
            return None

    monkeypatch.setattr("veld.gui.input.PPOCheckpointAgent", DummyPPO)
    controller = GuiController(mode="human_vs_ppo", seed=42, checkpoint=checkpoint_path)
    assert isinstance(controller.agent_b, DummyPPO)
    assert controller.agent_b.checkpoint == checkpoint_path


def test_ppo_vs_mcts_uses_expected_agents(monkeypatch):
    fake_ppo = RandomAgent(seed=99)
    monkeypatch.setattr(GuiController, "_build_ppo_agent", lambda self: fake_ppo)
    controller = GuiController(mode="ppo_vs_mcts", seed=42)
    assert controller.agent_a is fake_ppo
    assert isinstance(controller.agent_b, MCTSAgent)


def test_ppo_watch_rejects_invalid_watch_opponent(monkeypatch):
    monkeypatch.setattr(GuiController, "_build_ppo_agent", lambda self: RandomAgent(seed=7))
    with pytest.raises(ValueError, match="Invalid watch opponent type"):
        GuiController(mode="ppo_watch", seed=42, watch_opponent_type="bad")
