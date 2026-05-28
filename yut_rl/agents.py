from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import ast
import json
import random

from .env import ACTION_DIM, FINISH, START, YUT_OUTCOMES, YutEnv, advance, decode_action, distance_to_finish


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


class StrategicRuleBasedAgent:
    """Heuristic agent that scores tactical value and opponent counterplay."""

    def __init__(
        self,
        finish_weight: float = 130.0,
        capture_weight: float = 55.0,
        progress_weight: float = 3.0,
        stack_weight: float = 7.0,
        danger_weight: float = 28.0,
        shortcut_weight: float = 8.0,
        counterplay_weight: float = 18.0,
        seed: int | None = None,
    ):
        self.finish_weight = finish_weight
        self.capture_weight = capture_weight
        self.progress_weight = progress_weight
        self.stack_weight = stack_weight
        self.danger_weight = danger_weight
        self.shortcut_weight = shortcut_weight
        self.counterplay_weight = counterplay_weight
        self.rng = random.Random(seed)

    def act(self, env: YutEnv) -> int:
        legal = env.legal_actions()
        scored = [(self.score_action(env, action), self.rng.random(), action) for action in legal]
        return max(scored)[2]

    def score_action(self, env: YutEnv, action: int) -> float:
        return self.score_breakdown(env, action)["score"]

    def score_breakdown(self, env: YutEnv, action: int) -> dict[str, float]:
        player = env.current_player
        opponent = 1 - player
        piece, steps = decode_action(action)
        old_pos = env.positions[player][piece]
        before_me = env.positions[player][:]
        before_opp = env.positions[opponent][:]
        before_distance = sum(distance_to_finish(pos) for pos in before_me)
        before_finished = before_me.count(FINISH)
        before_opp_finished = before_opp.count(FINISH)

        candidate = env.clone()
        result = candidate.step(action)
        after_me = candidate.positions[player]
        after_opp = candidate.positions[opponent]
        new_pos = after_me[piece]
        moving_count = before_me.count(old_pos) if old_pos not in (START, FINISH) else 1
        captured_count = self._count_captured(before_opp, after_opp)
        finished_gain = after_me.count(FINISH) - before_finished
        opp_finished_gain = after_opp.count(FINISH) - before_opp_finished
        progress_gain = before_distance - sum(distance_to_finish(pos) for pos in after_me)
        danger = self._capture_danger(candidate, player)

        if result.done:
            terminal = 10_000.0 if candidate.winner() == player else -10_000.0
            return {
                "score": terminal,
                "terminal": terminal,
                "finish": 0.0,
                "capture": 0.0,
                "progress": 0.0,
                "stack": 0.0,
                "shortcut": 0.0,
                "danger": 0.0,
                "opponent_finish": 0.0,
                "distance": 0.0,
                "capture_bonus": 0.0,
                "counterplay": 0.0,
                "start_bonus": 0.0,
                "stack_risk": 0.0,
            }

        parts = {
            "terminal": 0.0,
            "finish": self.finish_weight * finished_gain,
            "capture": self.capture_weight * captured_count,
            "progress": self.progress_weight * progress_gain,
            "stack": self.stack_weight * max(0, moving_count - 1),
            "shortcut": self.shortcut_weight * self._shortcut_bonus(old_pos, new_pos),
            "danger": -self.danger_weight * danger,
            "opponent_finish": -15.0 * opp_finished_gain,
            "distance": -0.2 * distance_to_finish(new_pos),
            "capture_bonus": 0.0,
            "counterplay": 0.0,
            "start_bonus": 0.0,
            "stack_risk": 0.0,
        }

        if captured_count and candidate.current_player == player:
            parts["capture_bonus"] = 12.0
        if candidate.current_player == opponent:
            parts["counterplay"] = -self.counterplay_weight * self._best_counterplay(candidate, opponent)
        if old_pos == START and self._pieces_on_board(before_me) < 2:
            parts["start_bonus"] = 6.0
        if moving_count > 1 and danger > 0:
            parts["stack_risk"] = -8.0 * moving_count
        parts["score"] = sum(parts.values())
        return parts

    def _best_counterplay(self, env: YutEnv, opponent: int) -> float:
        legal = env.legal_actions()
        if not legal:
            return 0.0
        return max(self._counterplay_gain(env, action, opponent) for action in legal)

    def _counterplay_gain(self, env: YutEnv, action: int, opponent: int) -> float:
        before_opp = env.positions[opponent][:]
        before_me = env.positions[1 - opponent][:]
        before_distance = sum(distance_to_finish(pos) for pos in before_opp)
        before_finished = before_opp.count(FINISH)

        candidate = env.clone()
        result = candidate.step(action)
        if result.done:
            return 8.0 if candidate.winner() == opponent else 0.0

        after_opp = candidate.positions[opponent]
        after_me = candidate.positions[1 - opponent]
        captured_count = self._count_captured(before_me, after_me)
        finished_gain = after_opp.count(FINISH) - before_finished
        progress_gain = before_distance - sum(distance_to_finish(pos) for pos in after_opp)
        return 2.5 * finished_gain + 1.0 * captured_count + 0.04 * progress_gain

    def _capture_danger(self, env: YutEnv, player: int) -> float:
        opponent = 1 - player
        my_positions = [pos for pos in env.positions[player] if pos not in (START, FINISH)]
        if not my_positions:
            return 0.0

        danger = 0.0
        step_probs = {steps: prob for _, steps, prob, _ in YUT_OUTCOMES}
        for opp_pos in env.positions[opponent]:
            if opp_pos == FINISH:
                continue
            for steps, prob in step_probs.items():
                try:
                    target = advance(opp_pos, steps)
                except ValueError:
                    continue
                if target in my_positions:
                    stack_size = env.positions[player].count(target)
                    danger += prob * stack_size
        return danger

    @staticmethod
    def _pieces_on_board(positions: list[int]) -> int:
        return sum(pos not in (START, FINISH) for pos in positions)

    @staticmethod
    def _count_captured(before: list[int], after: list[int]) -> int:
        return sum(
            1
            for old, new in zip(before, after)
            if old not in (START, FINISH) and new == START
        )

    @staticmethod
    def _shortcut_bonus(old_pos: int, new_pos: int) -> float:
        if old_pos in (START, FINISH) or new_pos == FINISH:
            return 0.0
        if new_pos in {4, 9, 24}:
            return 1.0
        if new_pos in {20, 21, 22, 23, 25, 26}:
            return 0.5
        return 0.0


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
    discount_power: int = 1


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


def masked_argmax(values, legal: list[int], fill_value: float = -1e9) -> int:
    masked = values.clone()
    illegal = set(range(len(masked))) - set(legal)
    for action in illegal:
        masked[action] = fill_value
    return int(masked.argmax().item())


def masked_categorical(torch, logits, legal: list[int]):
    masked = logits.clone()
    illegal = set(range(masked.shape[-1])) - set(legal)
    for action in illegal:
        masked[..., action] = -1e9
    return torch.distributions.Categorical(logits=masked)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        gamma: float = 0.97,
        grad_clip: float = 1.0,
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
        self.grad_clip = grad_clip
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
        return masked_argmax(q_values, legal)

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
            discounts = torch.tensor([self.gamma ** t.discount_power for t in batch], dtype=torch.float32)
            target = rewards + discounts * next_q

        loss = self.loss_fn(q, target)
        self.optim.zero_grad()
        loss.backward()
        self.torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
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
                "grad_clip": self.grad_clip,
            },
            path,
        )

    def load(self, path) -> None:
        checkpoint = self.torch.load(path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])
        self.target.load_state_dict(checkpoint.get("target", checkpoint["model"]))


class DoubleDQNAgent(DQNAgent):
    """DQN variant that decouples next-action selection and target evaluation."""

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
            next_online = self.model(next_states)
            next_target = self.target(next_states)
            masks = torch.full_like(next_online, -1e9)
            for row, transition in enumerate(batch):
                if transition.next_legal:
                    masks[row, transition.next_legal] = 0
            next_actions = (next_online + masks).argmax(dim=1)
            next_q = next_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            next_q[dones] = 0
            discounts = torch.tensor([self.gamma ** t.discount_power for t in batch], dtype=torch.float32)
            target = rewards + discounts * next_q

        loss = self.loss_fn(q, target)
        self.optim.zero_grad()
        loss.backward()
        self.torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optim.step()
        return float(loss.item())


class DuelingNetwork:
    def __new__(cls, state_dim: int, hidden_dim: int, action_dim: int):
        import torch.nn as nn

        class _Dueling(nn.Module):
            def __init__(self):
                super().__init__()
                self.feature = nn.Sequential(
                    nn.Linear(state_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                )
                self.value = nn.Linear(hidden_dim, 1)
                self.advantage = nn.Linear(hidden_dim, action_dim)

            def forward(self, x):
                features = self.feature(x)
                value = self.value(features)
                advantage = self.advantage(features)
                return value + advantage - advantage.mean(dim=1, keepdim=True)

        return _Dueling()


class DuelingDQNAgent(DoubleDQNAgent):
    """Dueling Double DQN with separate value and advantage heads."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        gamma: float = 0.97,
        grad_clip: float = 1.0,
        seed: int | None = None,
    ):
        super().__init__(state_dim, action_dim, hidden_dim, lr, gamma, grad_clip=grad_clip, seed=seed)
        self.model = DuelingNetwork(state_dim, hidden_dim, action_dim)
        self.target = DuelingNetwork(state_dim, hidden_dim, action_dim)
        self.target.load_state_dict(self.model.state_dict())
        self.optim = self.torch.optim.Adam(self.model.parameters(), lr=lr)


class ValueNetworkAgent:
    """State-value agent that chooses actions by comparing simulated next states."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        lookahead_depth: int = 2,
        lr: float = 1e-3,
        gamma: float = 0.97,
        grad_clip: float = 1.0,
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
        self.lookahead_depth = lookahead_depth
        self.gamma = gamma
        self.grad_clip = grad_clip
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
            score = self._score_action(env, action, player, self.lookahead_depth)
            scored.append((score, action))
        return max(scored)[1]

    def _state_value(self, env: YutEnv, player: int) -> float:
        state = self.torch.tensor(env.observe_for(player), dtype=self.torch.float32).unsqueeze(0)
        with self.torch.no_grad():
            return float(self.model(state).item())

    def _score_action(self, env: YutEnv, action: int, player: int, depth: int) -> float:
        candidate = env.clone()
        result = candidate.step(action)
        if result.done:
            winner = candidate.winner()
            return 1.0 if winner == player else -1.0
        if depth <= 1 or candidate.current_player != player:
            return result.reward + self.gamma * self._state_value(candidate, player)

        next_legal = candidate.legal_actions()
        if not next_legal:
            return result.reward + self.gamma * self._state_value(candidate, player)
        next_score = max(self._score_action(candidate, next_action, player, depth - 1) for next_action in next_legal)
        return result.reward + self.gamma * next_score

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
        self.torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
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
                "lookahead_depth": self.lookahead_depth,
                "grad_clip": self.grad_clip,
            },
            path,
        )

    def load(self, path) -> None:
        checkpoint = self.torch.load(path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])
        self.target.load_state_dict(checkpoint.get("target", checkpoint["model"]))

    def clone_frozen(self) -> "ValueNetworkAgent":
        clone = ValueNetworkAgent(
            state_dim=self.state_dim,
            hidden_dim=self.hidden_dim,
            lookahead_depth=self.lookahead_depth,
            gamma=self.gamma,
            grad_clip=self.grad_clip,
        )
        clone.model.load_state_dict(self.model.state_dict())
        clone.target.load_state_dict(self.target.state_dict())
        return clone


class StrategicValueNetworkAgent(ValueNetworkAgent):
    """Value Network agent blended with tactical Yutnori heuristics."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        lookahead_depth: int = 2,
        lr: float = 1e-3,
        gamma: float = 0.97,
        grad_clip: float = 1.0,
        heuristic_weight: float = 0.35,
        heuristic_params: dict | None = None,
        seed: int | None = None,
    ):
        super().__init__(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            lookahead_depth=lookahead_depth,
            lr=lr,
            gamma=gamma,
            grad_clip=grad_clip,
            seed=seed,
        )
        self.heuristic_weight = heuristic_weight
        self.heuristic_params = heuristic_params or {}
        self.heuristic = StrategicRuleBasedAgent(seed=seed, **self.heuristic_params)

    def _score_action(self, env: YutEnv, action: int, player: int, depth: int) -> float:
        return self.score_components(env, action, player, depth)["score"]

    def score_components(self, env: YutEnv, action: int, player: int, depth: int | None = None) -> dict[str, float]:
        if depth is None:
            depth = self.lookahead_depth
        value_score = super()._score_action(env, action, player, depth)
        heuristic_parts = self.heuristic.score_breakdown(env, action)
        heuristic_score = max(-1.0, min(1.0, heuristic_parts["score"] / 150.0))
        heuristic_breakdown = {}
        for key, value in heuristic_parts.items():
            name = "heuristic_raw_score" if key == "score" else f"heuristic_{key}"
            heuristic_breakdown[name] = value
        return {
            "score": (1.0 - self.heuristic_weight) * value_score + self.heuristic_weight * heuristic_score,
            "value_score": value_score,
            "heuristic_score": heuristic_score,
            "weighted_value": (1.0 - self.heuristic_weight) * value_score,
            "weighted_heuristic": self.heuristic_weight * heuristic_score,
            **heuristic_breakdown,
        }

    def save(self, path) -> None:
        self.torch.save(
            {
                "model": self.model.state_dict(),
                "target": self.target.state_dict(),
                "gamma": self.gamma,
                "hidden_dim": self.hidden_dim,
                "lookahead_depth": self.lookahead_depth,
                "grad_clip": self.grad_clip,
                "heuristic_weight": self.heuristic_weight,
                "heuristic_params": self.heuristic_params,
            },
            path,
        )

    def clone_frozen(self) -> "StrategicValueNetworkAgent":
        clone = StrategicValueNetworkAgent(
            state_dim=self.state_dim,
            hidden_dim=self.hidden_dim,
            lookahead_depth=self.lookahead_depth,
            gamma=self.gamma,
            grad_clip=self.grad_clip,
            heuristic_weight=self.heuristic_weight,
            heuristic_params=self.heuristic_params,
        )
        clone.model.load_state_dict(self.model.state_dict())
        clone.target.load_state_dict(self.target.state_dict())
        return clone


class MCTSValueAgent:
    """Shallow Monte Carlo tree search guided by a trained ValueNetworkAgent."""

    def __init__(
        self,
        value_agent: ValueNetworkAgent,
        simulations: int = 32,
        rollout_depth: int = 6,
        seed: int | None = None,
    ):
        self.value_agent = value_agent
        self.simulations = simulations
        self.rollout_depth = rollout_depth
        self.rng = random.Random(seed)

    def act(self, env: YutEnv, epsilon: float = 0.0) -> int:
        legal = env.legal_actions()
        if self.rng.random() < epsilon:
            return self.rng.choice(legal)

        player = env.current_player
        scored = []
        for action in legal:
            total = 0.0
            for _ in range(max(1, self.simulations)):
                candidate = env.clone()
                result = candidate.step(action)
                total += self._rollout_value(candidate, player, result.reward, result.done)
            scored.append((total / max(1, self.simulations), action))
        return max(scored)[1]

    def _rollout_value(self, env: YutEnv, player: int, initial_reward: float, done: bool) -> float:
        if done:
            winner = env.winner()
            return 1.0 if winner == player else -1.0

        total = initial_reward
        discount = self.value_agent.gamma
        for depth in range(self.rollout_depth):
            if env.winner() is not None:
                return 1.0 if env.winner() == player else -1.0
            legal = env.legal_actions()
            if not legal:
                break
            result = env.step(self.rng.choice(legal))
            total += (discount ** (depth + 1)) * result.reward
            if result.done:
                winner = env.winner()
                return 1.0 if winner == player else -1.0
        return total + (discount ** (self.rollout_depth + 1)) * self.value_agent._state_value(env, player)

    def save(self, path) -> None:
        self.value_agent.save(path)


class ActorCriticAgent:
    """Shared actor-critic network used by REINFORCE, A2C, and PPO."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        gamma: float = 0.97,
        grad_clip: float = 1.0,
        seed: int | None = None,
    ):
        import torch
        import torch.nn as nn

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        self.torch = torch
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.grad_clip = grad_clip
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy = nn.Linear(hidden_dim, action_dim)
        self.value = nn.Linear(hidden_dim, 1)
        self.optim = torch.optim.Adam(list(self.net.parameters()) + list(self.policy.parameters()) + list(self.value.parameters()), lr=lr)

    def forward(self, states):
        features = self.net(states)
        return self.policy(features), self.value(features).squeeze(-1)

    def act(self, env: YutEnv, epsilon: float = 0.0) -> int:
        state = self.torch.tensor(env.observe(), dtype=self.torch.float32).unsqueeze(0)
        logits, _ = self.forward(state)
        dist = masked_categorical(self.torch, logits.squeeze(0), env.legal_actions())
        if random.random() < epsilon:
            return random.choice(env.legal_actions())
        return int(dist.sample().item())

    def evaluate_actions(self, states, actions, legal_actions: list[list[int]]):
        logits, values = self.forward(states)
        log_probs = []
        entropies = []
        for row, legal in enumerate(legal_actions):
            dist = masked_categorical(self.torch, logits[row], legal)
            log_probs.append(dist.log_prob(actions[row]))
            entropies.append(dist.entropy())
        return self.torch.stack(log_probs), values, self.torch.stack(entropies)

    def save(self, path) -> None:
        self.torch.save(
            {
                "net": self.net.state_dict(),
                "policy": self.policy.state_dict(),
                "value": self.value.state_dict(),
                "gamma": self.gamma,
                "hidden_dim": self.hidden_dim,
                "action_dim": self.action_dim,
                "grad_clip": self.grad_clip,
            },
            path,
        )

    def load(self, path) -> None:
        checkpoint = self.torch.load(path, map_location="cpu")
        self.net.load_state_dict(checkpoint["net"])
        self.policy.load_state_dict(checkpoint["policy"])
        self.value.load_state_dict(checkpoint["value"])


class ReinforceAgent(ActorCriticAgent):
    pass


class A2CAgent(ActorCriticAgent):
    pass


class PPOAgent(ActorCriticAgent):
    pass
