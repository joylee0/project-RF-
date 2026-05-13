from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import random

from .env import FINISH, START, YutEnv, advance, distance_to_finish


class RandomAgent:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def act(self, env: YutEnv) -> int:
        return self.rng.choice(env.legal_actions())


class RuleBasedAgent:
    """Simple hand-written strategy used as a stronger baseline."""

    def act(self, env: YutEnv) -> int:
        player = env.current_player
        opponent = 1 - player
        steps = env.pending_steps[0]
        legal = env.legal_actions()
        opp_positions = set(env.positions[opponent])

        scored = []
        for piece in legal:
            old_pos = env.positions[player][piece]
            new_pos = advance(old_pos, steps)
            score = 0.0
            if new_pos == FINISH:
                score += 100
            if new_pos in opp_positions:
                score += 50
            if old_pos == START:
                score += 5
            stack_size = env.positions[player].count(old_pos)
            score += 4 * max(0, stack_size - 1)
            score -= 0.5 * distance_to_finish(new_pos)
            scored.append((score, piece))
        return max(scored)[1]


@dataclass
class Transition:
    state: list[float]
    action: int
    reward: float
    next_state: list[float]
    done: bool
    next_legal: list[int]


@dataclass
class ValueTransition:
    state: list[float]
    reward: float
    next_state: list[float]
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.items: deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.items.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.items, batch_size)

    def __len__(self) -> int:
        return len(self.items)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 4,
        lr: float = 1e-3,
        gamma: float = 0.97,
        seed: int | None = None,
    ):
        import torch
        import torch.nn as nn

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        self.torch = torch
        self.gamma = gamma
        self.action_dim = action_dim
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )
        self.target = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )
        self.target.load_state_dict(self.model.state_dict())
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

    def act(self, env: YutEnv, epsilon: float = 0.0) -> int:
        legal = env.legal_actions()
        if random.random() < epsilon:
            return random.choice(legal)
        state = self.torch.tensor(env.observe(), dtype=self.torch.float32).unsqueeze(0)
        with self.torch.no_grad():
            q_values = self.model(state).squeeze(0)
        illegal = set(range(self.action_dim)) - set(legal)
        for action in illegal:
            q_values[action] = -1e9
        return int(self.torch.argmax(q_values).item())

    def train_batch(self, replay: ReplayBuffer, batch_size: int = 64) -> float | None:
        if len(replay) < batch_size:
            return None

        batch = replay.sample(batch_size)
        torch = self.torch
        states = torch.tensor([t.state for t in batch], dtype=torch.float32)
        actions = torch.tensor([t.action for t in batch], dtype=torch.long)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32)
        next_states = torch.tensor([t.next_state for t in batch], dtype=torch.float32)
        dones = torch.tensor([t.done for t in batch], dtype=torch.bool)

        q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_all = self.target(next_states)
            masks = torch.full_like(next_q_all, -1e9)
            for row, transition in enumerate(batch):
                if transition.next_legal:
                    masks[row, transition.next_legal] = 0
            next_q = (next_q_all + masks).max(dim=1).values
            next_q[dones] = 0
            target = rewards + self.gamma * next_q

        loss = self.loss_fn(q, target)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return float(loss.item())

    def sync_target(self) -> None:
        self.target.load_state_dict(self.model.state_dict())

    def save(self, path) -> None:
        self.torch.save(
            {
                "model": self.model.state_dict(),
                "target": self.target.state_dict(),
                "gamma": self.gamma,
                "action_dim": self.action_dim,
            },
            path,
        )

    def load(self, path) -> None:
        checkpoint = self.torch.load(path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])
        self.target.load_state_dict(checkpoint.get("target", checkpoint["model"]))


class ValueNetworkAgent:
    """State-value agent that chooses actions by comparing simulated next states."""

    def __init__(
        self,
        state_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.97,
        seed: int | None = None,
    ):
        import torch
        import torch.nn as nn

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        self.torch = torch
        self.gamma = gamma
        self.model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )
        self.target = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )
        self.target.load_state_dict(self.model.state_dict())
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

    def act(self, env: YutEnv, epsilon: float = 0.0) -> int:
        legal = env.legal_actions()
        if random.random() < epsilon:
            return random.choice(legal)

        player = env.current_player
        scored = []
        for action in legal:
            candidate = env.clone()
            result = candidate.step(action)
            if result.done:
                winner = candidate.winner()
                score = 1.0 if winner == player else -1.0
            else:
                state = self.torch.tensor(candidate.observe_for(player), dtype=self.torch.float32).unsqueeze(0)
                with self.torch.no_grad():
                    value = float(self.model(state).item())
                score = result.reward + self.gamma * value
            scored.append((score, action))
        return max(scored)[1]

    def train_batch(self, replay: ReplayBuffer, batch_size: int = 64) -> float | None:
        if len(replay) < batch_size:
            return None

        batch = replay.sample(batch_size)
        torch = self.torch
        states = torch.tensor([t.state for t in batch], dtype=torch.float32)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32)
        next_states = torch.tensor([t.next_state for t in batch], dtype=torch.float32)
        dones = torch.tensor([t.done for t in batch], dtype=torch.bool)

        values = self.model(states).squeeze(1)
        with torch.no_grad():
            next_values = self.target(next_states).squeeze(1)
            next_values[dones] = 0
            targets = rewards + self.gamma * next_values
            targets = targets.clamp(-1.0, 1.0)

        loss = self.loss_fn(values, targets)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return float(loss.item())

    def sync_target(self) -> None:
        self.target.load_state_dict(self.model.state_dict())

    def save(self, path) -> None:
        self.torch.save(
            {
                "model": self.model.state_dict(),
                "target": self.target.state_dict(),
                "gamma": self.gamma,
            },
            path,
        )

    def load(self, path) -> None:
        checkpoint = self.torch.load(path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])
        self.target.load_state_dict(checkpoint.get("target", checkpoint["model"]))
