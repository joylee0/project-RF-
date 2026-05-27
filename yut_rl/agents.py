from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import ast
import json
import random

from .env import ACTION_DIM, FINISH, START, YutEnv, advance, decode_action, distance_to_finish


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
        legal = env.legal_actions()
        opp_positions = set(env.positions[opponent])

        scored = []
        for action in legal:
            piece, steps = decode_action(action)
            old_pos = env.positions[player][piece]
            new_pos = advance(old_pos, steps)
            score = 0.0
            if new_pos == FINISH:
                score += 100
            if new_pos != FINISH and new_pos in opp_positions:
                score += 50
            if old_pos == START:
                score += 5
            stack_size = 1 if old_pos == START else env.positions[player].count(old_pos)
            score += 4 * max(0, stack_size - 1)
            score -= 0.5 * distance_to_finish(new_pos)
            scored.append((score, action))
        return max(scored)[1]


class TabularQAgent:
    """Dependency-free Q-learning baseline for first experiments.

    The state space is still large, but this agent is useful as a runnable
    first RL draft before moving to neural models such as DQN or PPO.
    """

    def __init__(
        self,
        alpha: float = 0.15,
        gamma: float = 0.97,
        seed: int | None = None,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.rng = random.Random(seed)
        self.q: dict[tuple, list[float]] = {}

    def act(self, env: YutEnv, epsilon: float = 0.0) -> int:
        legal = env.legal_actions()
        if self.rng.random() < epsilon:
            return self.rng.choice(legal)

        values = self._values(self.state_key(env))
        return max(legal, key=lambda action: values[action])

    def update(
        self,
        state_key: tuple,
        action: int,
        reward: float,
        next_state_key: tuple | None,
        next_legal: list[int],
        done: bool,
    ) -> None:
        values = self._values(state_key)
        if done or next_state_key is None or not next_legal:
            target = reward
        else:
            next_values = self._values(next_state_key)
            target = reward + self.gamma * max(next_values[action] for action in next_legal)
        values[action] += self.alpha * (target - values[action])

    def state_key(self, env: YutEnv, player: int | None = None) -> tuple:
        if player is None:
            player = env.current_player
        opponent = 1 - player
        return (
            tuple(env.positions[player]),
            tuple(env.positions[opponent]),
            tuple(env.pending_steps),
            env.current_player == player,
        )

    def save(self, path) -> None:
        payload = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "q": [
                {"state": repr(state), "values": values}
                for state, values in self.q.items()
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def load(self, path) -> None:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        self.alpha = payload.get("alpha", self.alpha)
        self.gamma = payload.get("gamma", self.gamma)
        self.q = {
            ast.literal_eval(item["state"]): item["values"]
            for item in payload.get("q", [])
        }

    def _values(self, state_key: tuple) -> list[float]:
        return self.q.setdefault(state_key, [0.0] * ACTION_DIM)


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
        action_dim: int = ACTION_DIM,
        hidden_dim: int = 256,
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
        self.state_dim = state_dim
        self.gamma = gamma
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.target = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
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
                "hidden_dim": self.hidden_dim,
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
        hidden_dim: int = 256,
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
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )
        self.target = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
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
                "hidden_dim": self.hidden_dim,
            },
            path,
        )

    def load(self, path) -> None:
        checkpoint = self.torch.load(path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])
        self.target.load_state_dict(checkpoint.get("target", checkpoint["model"]))

    def clone_frozen(self) -> "ValueNetworkAgent":
        clone = ValueNetworkAgent(state_dim=self.state_dim, hidden_dim=self.hidden_dim, gamma=self.gamma)
        clone.model.load_state_dict(self.model.state_dict())
        clone.target.load_state_dict(self.target.state_dict())
        return clone
