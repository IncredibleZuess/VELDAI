from __future__ import annotations

from pathlib import Path

import torch

from veld.env.action_codec import ActionCodec
from veld.env.veld_env import encode_state
from veld.train.ppo_train import MaskedCategorical, VeldCNN


class PPOCheckpointAgent:
    def __init__(self, checkpoint: str | Path) -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = VeldCNN(payload["n_actions"])
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()
        self.codec = ActionCodec()

    def choose_action(self, state):
        legal = self.codec.legal_indices(state)
        if not legal:
            return None
        mask = torch.zeros((1, self.codec.action_space_size), dtype=torch.bool)
        mask[0, legal] = True
        obs = torch.tensor(encode_state(state))
        with torch.no_grad():
            logits, _ = self.model(obs)
            dist = MaskedCategorical(logits, mask)
            idx = int(dist.probs.argmax(dim=-1).item())
        if idx not in legal:
            idx = legal[0]
        return self.codec.decode(idx, state)
