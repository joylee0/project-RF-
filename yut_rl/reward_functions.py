from __future__ import annotations

from dataclasses import dataclass


class RewardFunction:
    name = "base"

    def __call__(self, before, after, info: dict, done: bool, player: int) -> float:
        raise NotImplementedError


def _env_mod():
    from yut_rl import env as env_mod

    return env_mod


def _terminal(done: bool, after, player: int, win: float = 1.0, loss: float = -1.0) -> float:
    if not done:
        return 0.0
    winner = after.winner()
    if winner is None:
        return 0.0
    return win if winner == player else loss


def _count_captured(before_positions: list[int], after_positions: list[int]) -> int:
    env_mod = _env_mod()
    return sum(
        1
        for old, new in zip(before_positions, after_positions)
        if old not in (env_mod.START, env_mod.FINISH) and new == env_mod.START
    )


def _danger(env, player: int) -> bool:
    env_mod = _env_mod()
    my_positions = [pos for pos in env.positions[player] if pos not in (env_mod.START, env_mod.FINISH)]
    if not my_positions:
        return False
    for opp_pos in env.positions[1 - player]:
        if opp_pos == env_mod.FINISH:
            continue
        for _, steps, _, _ in env_mod.YUT_OUTCOMES:
            if env_mod.advance(opp_pos, steps) in my_positions:
                return True
    return False


def _opponent_counterplay(after, player: int) -> bool:
    if after.current_player == player:
        return False
    opponent = 1 - player
    env_mod = _env_mod()
    my_positions = {pos for pos in after.positions[player] if pos not in (env_mod.START, env_mod.FINISH)}
    for action in after.legal_actions():
        piece, steps = after.decode_action(action)
        new_pos = env_mod.advance(after.positions[opponent][piece], steps)
        if new_pos == env_mod.FINISH or new_pos in my_positions:
            return True
    return False


@dataclass(frozen=True)
class SparseReward(RewardFunction):
    name: str = "sparse"

    def __call__(self, before, after, info: dict, done: bool, player: int) -> float:
        return _terminal(done, after, player)


@dataclass(frozen=True)
class MinimalDenseReward(RewardFunction):
    name: str = "minimal_dense"

    def __call__(self, before, after, info: dict, done: bool, player: int) -> float:
        env_mod = _env_mod()
        reward = _terminal(done, after, player)
        reward += 0.3 * max(0, after.positions[player].count(env_mod.FINISH) - before.positions[player].count(env_mod.FINISH))
        reward += 0.2 * _count_captured(before.positions[1 - player], after.positions[1 - player])
        reward -= 0.2 * _count_captured(before.positions[player], after.positions[player])
        reward -= 0.001
        return reward


@dataclass(frozen=True)
class BalancedTacticalReward(RewardFunction):
    name: str = "balanced_tactical"

    def __call__(self, before, after, info: dict, done: bool, player: int) -> float:
        env_mod = _env_mod()
        old_pos = info.get("from")
        new_pos = info.get("to")
        moved_steps = max(0, env_mod.distance_to_finish(old_pos) - env_mod.distance_to_finish(new_pos)) if old_pos is not None and new_pos is not None else 0
        reward = MinimalDenseReward()(before, after, info, done, player)
        reward += 0.1 * _count_captured(before.positions[1 - player], after.positions[1 - player])
        if not _danger(before, player) and _danger(after, player):
            reward -= 0.15
        if _danger(before, player) and not _danger(after, player):
            reward += 0.1
        if old_pos not in (None, env_mod.START, env_mod.FINISH) and new_pos in {4, 9, 24}:
            reward += 0.08
        reward += 0.01 * moved_steps
        return reward


@dataclass(frozen=True)
class CaptureHeavyReward(RewardFunction):
    name: str = "capture_heavy"

    def __call__(self, before, after, info: dict, done: bool, player: int) -> float:
        env_mod = _env_mod()
        reward = _terminal(done, after, player)
        reward += 0.5 * _count_captured(before.positions[1 - player], after.positions[1 - player])
        reward += 0.25 * max(0, after.positions[player].count(env_mod.FINISH) - before.positions[player].count(env_mod.FINISH))
        reward -= 0.3 * _count_captured(before.positions[player], after.positions[player])
        reward -= 0.001
        return reward


@dataclass(frozen=True)
class RiskAwareReward(RewardFunction):
    name: str = "risk_aware"

    def __call__(self, before, after, info: dict, done: bool, player: int) -> float:
        env_mod = _env_mod()
        reward = _terminal(done, after, player)
        finished_gain = max(0, after.positions[player].count(env_mod.FINISH) - before.positions[player].count(env_mod.FINISH))
        captured = _count_captured(before.positions[1 - player], after.positions[1 - player])
        reward += 0.35 * finished_gain
        reward += 0.25 * captured
        reward -= 0.35 * _count_captured(before.positions[player], after.positions[player])
        if not _danger(before, player) and _danger(after, player):
            reward -= 0.3
        if _opponent_counterplay(after, player):
            reward -= 0.2
        if not _danger(after, player) and (finished_gain or captured):
            reward += 0.15
        reward -= 0.001
        return reward


def get_reward_function(name: str | RewardFunction | None) -> RewardFunction | None:
    if isinstance(name, RewardFunction):
        return name
    key = (name or "legacy").lower()
    if key == "legacy":
        return None
    if key == "sparse":
        return SparseReward()
    if key in {"minimal_dense", "minimal-dense"}:
        return MinimalDenseReward()
    if key in {"balanced", "balanced_tactical", "balanced-tactical"}:
        return BalancedTacticalReward()
    if key in {"capture_heavy", "capture-heavy"}:
        return CaptureHeavyReward()
    if key in {"risk", "risk_aware", "risk-aware"}:
        return RiskAwareReward()
    raise ValueError(f"unknown reward_function: {name}")
