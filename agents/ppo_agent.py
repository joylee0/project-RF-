from __future__ import annotations

import random

import torch
import torch.nn as nn

from yut_rl.env import (
    ACTION_DIM,
    FINISH,
    MAX_STEPS,
    POSITION_FEATURES,
    POS_TO_INDEX,
    START,
    YUT_OUTCOMES,
    YutEnv,
    advance,
    decode_action,
    distance_to_finish,
)


SHORTCUT_ENTRY_POSITIONS = {4, 9, 24}


def encode_position(pos: int) -> list[float]:
    out = [0.0] * POSITION_FEATURES
    out[POS_TO_INDEX[pos]] = 1.0
    return out


def can_capture(env: YutEnv, player: int) -> float:
    opponent_positions = set(pos for pos in env.positions[1 - player] if pos not in (START, FINISH))
    for action in env.legal_actions():
        piece, steps = decode_action(action)
        if advance(env.positions[player][piece], steps) in opponent_positions:
            return 1.0
    return 0.0


def can_finish(env: YutEnv, player: int) -> float:
    for action in env.legal_actions():
        piece, steps = decode_action(action)
        if advance(env.positions[player][piece], steps) == FINISH:
            return 1.0
    return 0.0


def can_enter_shortcut(env: YutEnv, player: int) -> float:
    for action in env.legal_actions():
        piece, steps = decode_action(action)
        old_pos = env.positions[player][piece]
        new_pos = advance(old_pos, steps)
        if old_pos not in (START, FINISH) and new_pos in SHORTCUT_ENTRY_POSITIONS:
            return 1.0
    return 0.0


def capture_danger(env: YutEnv, player: int) -> float:
    my_positions = [pos for pos in env.positions[player] if pos not in (START, FINISH)]
    if not my_positions:
        return 0.0
    for opp_pos in env.positions[1 - player]:
        if opp_pos == FINISH:
            continue
        for _, steps, _, _ in YUT_OUTCOMES:
            if advance(opp_pos, steps) in my_positions:
                return 1.0
    return 0.0


def build_state(env: YutEnv, player: int | None = None) -> list[float]:
    if player is None:
        player = env.current_player
    opponent = 1 - player
    features: list[float] = []

    for pos in env.positions[player]:
        features.extend(encode_position(pos))
    for pos in env.positions[opponent]:
        features.extend(encode_position(pos))

    for steps in range(1, MAX_STEPS + 1):
        features.append(min(env.pending_steps.count(steps), 4) / 4)

    my_finished = env.positions[player].count(FINISH)
    opp_finished = env.positions[opponent].count(FINISH)
    features.extend([my_finished / 4, opp_finished / 4])
    features.extend(
        [
            can_capture(env, player),
            can_finish(env, player),
            can_enter_shortcut(env, player),
            capture_danger(env, player),
            1.0 if env.current_player == player else 0.0,
        ]
    )

    for pos in env.positions[player]:
        features.append(distance_to_finish(pos) / 27)
    for pos in env.positions[opponent]:
        features.append(distance_to_finish(pos) / 27)
    return features


def state_dim(seed: int = 0) -> int:
    env = YutEnv(seed=seed)
    env.reset()
    return len(build_state(env))


def masked_logits(logits, legal_actions: list[int]):
    mask = torch.full_like(logits, -1e9)
    mask[..., legal_actions] = 0
    return logits + mask


class MaskedPPOAgent:
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        grad_clip: float = 1.0,
        seed: int | None = None,
    ):
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
        self.state_dim = state_dim
        self.gamma = gamma
        self.clip = clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.grad_clip = grad_clip
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy = nn.Linear(hidden_dim, ACTION_DIM)
        self.value = nn.Linear(hidden_dim, 1)
        self.optim = torch.optim.Adam(self.parameters(), lr=lr)

    def parameters(self):
        return list(self.body.parameters()) + list(self.policy.parameters()) + list(self.value.parameters())

    def forward(self, states):
        h = self.body(states)
        return self.policy(h), self.value(h).squeeze(-1)

    def act(self, env: YutEnv, epsilon: float = 0.0, deterministic: bool = False) -> int:
        legal = env.legal_actions()
        if epsilon and random.random() < epsilon:
            return random.choice(legal)
        state = torch.tensor(build_state(env), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.forward(state)
            logits = masked_logits(logits.squeeze(0), legal)
            if deterministic:
                return int(torch.argmax(logits).item())
            return int(torch.distributions.Categorical(logits=logits).sample().item())

    def evaluate_actions(self, states, actions, legal_actions: list[list[int]]):
        logits, values = self.forward(states)
        log_probs, entropies = [], []
        for row, legal in enumerate(legal_actions):
            dist = torch.distributions.Categorical(logits=masked_logits(logits[row], legal))
            log_probs.append(dist.log_prob(actions[row]))
            entropies.append(dist.entropy())
        return torch.stack(log_probs), values, torch.stack(entropies)

    def save(self, path) -> None:
        torch.save(
            {
                "body": self.body.state_dict(),
                "policy": self.policy.state_dict(),
                "value": self.value.state_dict(),
                "gamma": self.gamma,
                "clip": self.clip,
                "value_coef": self.value_coef,
                "entropy_coef": self.entropy_coef,
                "grad_clip": self.grad_clip,
                "state_dim": self.state_dim,
            },
            path,
        )

    def load(self, path) -> None:
        checkpoint = torch.load(path, map_location="cpu")
        self.body.load_state_dict(checkpoint["body"])
        self.policy.load_state_dict(checkpoint["policy"])
        self.value.load_state_dict(checkpoint["value"])


class CaptureAwarePPOAgent(MaskedPPOAgent):
    """PPO policy with a tactical logit bias for capture-heavy Yutnori play."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        grad_clip: float = 1.0,
        tactical_weight: float = 2.5,
        seed: int | None = None,
    ):
        super().__init__(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            lr=lr,
            gamma=gamma,
            clip=clip,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            grad_clip=grad_clip,
            seed=seed,
        )
        self.tactical_weight = tactical_weight

    def act(self, env: YutEnv, epsilon: float = 0.0, deterministic: bool = False) -> int:
        legal = env.legal_actions()
        if epsilon and random.random() < epsilon:
            return random.choice(legal)
        state = torch.tensor(build_state(env), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.forward(state)
            logits = masked_logits(logits.squeeze(0), legal)
            bonus = torch.zeros_like(logits)
            for action in legal:
                bonus[action] = self._tactical_bonus(env, action)
            logits = logits + self.tactical_weight * bonus
            if deterministic:
                return int(torch.argmax(logits).item())
            return int(torch.distributions.Categorical(logits=logits).sample().item())

    def _tactical_bonus(self, env: YutEnv, action: int) -> float:
        player = env.current_player
        piece, _ = decode_action(action)
        before_pos = env.positions[player][piece]
        before_distance = distance_to_finish(before_pos)
        before_danger = capture_danger(env, player)
        candidate = env.clone()
        before_opp = candidate.positions[1 - player][:]
        before_finished = candidate.positions[player].count(FINISH)
        result = candidate.step(action)
        after_pos = candidate.positions[player][piece]
        after_me = candidate.positions[player]
        after_opp = candidate.positions[1 - player]

        captured = sum(
            1
            for old, new in zip(before_opp, after_opp)
            if old not in (START, FINISH) and new == START
        )
        finished_gain = after_me.count(FINISH) - before_finished
        progress = max(0, before_distance - distance_to_finish(after_pos))
        after_danger = 0.0 if result.done else capture_danger(candidate, player)

        score = 0.0
        if result.done and candidate.winner() == player:
            score += 20.0
        score += 10.0 * finished_gain
        score += 9.0 * captured
        score += 0.12 * progress
        if can_capture(env, player) and not captured:
            score -= 4.0
        if before_pos not in (START, FINISH) and after_pos in SHORTCUT_ENTRY_POSITIONS:
            score += 1.5
        if after_danger:
            score -= 7.0
        elif before_danger and not after_danger:
            score += 3.0
        if self._has_near_goal_piece(env, player) and before_pos == START:
            score -= 2.5
        if distance_to_finish(after_pos) <= 5 and after_pos != FINISH:
            score += 1.0
        if self._opponent_near_goal(env, player) and captured:
            score += 4.0
        if not result.done and candidate.current_player != player:
            score -= 2.5 * self._opponent_counterplay(candidate, player)
        return score

    @staticmethod
    def _has_near_goal_piece(env: YutEnv, player: int, max_distance: int = 5) -> bool:
        return any(
            pos not in (START, FINISH) and distance_to_finish(pos) <= max_distance
            for pos in env.positions[player]
        )

    @staticmethod
    def _opponent_near_goal(env: YutEnv, player: int, max_distance: int = 5) -> bool:
        return any(
            pos not in (START, FINISH) and distance_to_finish(pos) <= max_distance
            for pos in env.positions[1 - player]
        )

    @staticmethod
    def _opponent_counterplay(env: YutEnv, player: int) -> float:
        opponent = 1 - player
        if env.current_player != opponent:
            return 0.0
        legal = env.legal_actions()
        if not legal:
            return 0.0
        best = 0.0
        for action in legal:
            piece, _ = decode_action(action)
            before_pos = env.positions[opponent][piece]
            before_distance = distance_to_finish(before_pos)
            before_me = env.positions[player][:]
            before_finished = env.positions[opponent].count(FINISH)
            candidate = env.clone()
            result = candidate.step(action)
            after_pos = candidate.positions[opponent][piece]
            captured = sum(
                1
                for old, new in zip(before_me, candidate.positions[player])
                if old not in (START, FINISH) and new == START
            )
            finished_gain = candidate.positions[opponent].count(FINISH) - before_finished
            progress = max(0, before_distance - distance_to_finish(after_pos))
            value = 2.5 * captured + 4.0 * finished_gain + 0.03 * progress
            if result.done and candidate.winner() == opponent:
                value += 8.0
            best = max(best, value)
        return best

    def save(self, path) -> None:
        super().save(path)
        checkpoint = torch.load(path, map_location="cpu")
        checkpoint["tactical_weight"] = self.tactical_weight
        torch.save(checkpoint, path)

    def load(self, path) -> None:
        checkpoint = torch.load(path, map_location="cpu")
        self.body.load_state_dict(checkpoint["body"])
        self.policy.load_state_dict(checkpoint["policy"])
        self.value.load_state_dict(checkpoint["value"])
        self.tactical_weight = checkpoint.get("tactical_weight", self.tactical_weight)
