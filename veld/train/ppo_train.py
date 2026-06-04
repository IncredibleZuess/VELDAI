from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from veld.agents.mcts import MCTSAgent
from veld.core.game import RandomAgent
from veld.env.veld_env import VeldEnv


class VeldCNN(nn.Module):
    def __init__(self, n_actions: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
        )
        self.policy = nn.Linear(256, n_actions)
        self.value = nn.Linear(256, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)
        obs = obs.permute(0, 3, 1, 2).float()
        features = self.features(obs)
        return self.policy(features), self.value(features).squeeze(-1)


class MaskedCategorical(Categorical):
    def __init__(self, logits: torch.Tensor, mask: torch.Tensor):
        logits = logits.masked_fill(~mask.bool(), torch.finfo(logits.dtype).min)
        super().__init__(logits=logits)


def make_opponent(step: int, self_agent=None):
    if step >= 1_000_000 and self_agent is not None:
        return self_agent
    if step >= 500_000:
        return MCTSAgent(n_simulations=100)
    return RandomAgent()


@dataclass
class RolloutBatch:
    obs: list[np.ndarray]
    actions: list[int]
    logprobs: list[float]
    rewards: list[float]
    dones: list[bool]
    values: list[float]
    masks: list[np.ndarray]


def compute_gae(rewards, dones, values, gamma=0.99, gae_lambda=0.95):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    lastgaelam = 0.0
    next_value = 0.0
    for t in reversed(range(len(rewards))):
        next_nonterminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * next_nonterminal * lastgaelam
        next_value = values[t]
    returns = advantages + np.asarray(values, dtype=np.float32)
    return advantages, returns


def train(args: argparse.Namespace) -> Path:
    device = torch.device("cpu")
    env = VeldEnv(seed=args.seed)
    obs, info = env.reset(seed=args.seed)
    model = VeldCNN(env.action_space.n).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    writer = SummaryWriter(args.logdir)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0
    last_checkpoint = checkpoint_dir / "ppo_step_0.pt"

    while global_step < args.steps:
        batch = RolloutBatch([], [], [], [], [], [], [])
        for _ in range(args.batch_size):
            mask_np = env.action_masks()
            with torch.no_grad():
                logits, value = model(torch.tensor(obs, device=device))
                dist = MaskedCategorical(logits, torch.tensor(mask_np[None, :], device=device))
                action = dist.sample()
                logprob = dist.log_prob(action)
            next_obs, reward, done, truncated, info = env.step(int(action.item()))
            batch.obs.append(obs)
            batch.actions.append(int(action.item()))
            batch.logprobs.append(float(logprob.item()))
            batch.rewards.append(float(reward))
            batch.dones.append(bool(done or truncated))
            batch.values.append(float(value.item()))
            batch.masks.append(mask_np)
            obs = next_obs
            global_step += 1
            if done or truncated:
                writer.add_scalar("charts/episode_reward", sum(batch.rewards), global_step)
                obs, info = env.reset()
            if global_step >= args.steps:
                break

        advantages, returns = compute_gae(batch.rewards, batch.dones, batch.values)
        obs_t = torch.tensor(np.asarray(batch.obs), device=device)
        actions_t = torch.tensor(batch.actions, device=device)
        old_logprobs_t = torch.tensor(batch.logprobs, device=device)
        advantages_t = torch.tensor(advantages, device=device)
        returns_t = torch.tensor(returns, device=device)
        masks_t = torch.tensor(np.asarray(batch.masks), device=device)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        for _ in range(args.update_epochs):
            logits, values = model(obs_t)
            dist = MaskedCategorical(logits, masks_t)
            new_logprobs = dist.log_prob(actions_t)
            entropy = dist.entropy().mean()
            ratio = (new_logprobs - old_logprobs_t).exp()
            policy_loss = -torch.min(
                advantages_t * ratio,
                advantages_t * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef),
            ).mean()
            value_loss = 0.5 * (returns_t - values).pow(2).mean()
            loss = policy_loss - args.entropy_coef * entropy + args.value_coef * value_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("losses/value_loss", value_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy.item(), global_step)
        if global_step % args.save_every < args.batch_size or global_step >= args.steps:
            last_checkpoint = checkpoint_dir / f"ppo_step_{global_step}.pt"
            torch.save({"model_state_dict": model.state_dict(), "n_actions": env.action_space.n}, last_checkpoint)

    writer.close()
    return last_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--logdir", default="runs")
    parser.add_argument("--save_every", type=int, default=100_000)
    parser.add_argument("--opponent", default="random", choices=["random", "mcts", "self"])
    return parser


def main(argv: list[str] | None = None) -> None:
    path = train(build_parser().parse_args(argv))
    print(f"saved checkpoint: {path}")


if __name__ == "__main__":
    main()
