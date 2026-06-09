from __future__ import annotations

from dataclasses import dataclass


class StateEncoder:
    name = "base"

    def encode(self, env, player: int | None = None) -> list[float]:
        raise NotImplementedError


def _env_mod():
    from yut_rl import env as env_mod

    return env_mod


def _one_hot_position(pos: int) -> list[float]:
    env_mod = _env_mod()
    encoded = [0.0] * env_mod.POSITION_FEATURES
    encoded[env_mod.POS_TO_INDEX[pos]] = 1.0
    return encoded


def _perspective(env, player: int | None) -> tuple[int, int]:
    actor = env.current_player if player is None else player
    return actor, 1 - actor


def _pending_features(env) -> list[float]:
    env_mod = _env_mod()
    return [min(env.pending_steps.count(steps), 4) / 4 for steps in range(1, env_mod.MAX_STEPS + 1)]


def _captures_available(env, player: int) -> float:
    if player != env.current_player:
        return 0.0
    env_mod = _env_mod()
    opponent_positions = {pos for pos in env.positions[1 - player] if pos not in (env_mod.START, env_mod.FINISH)}
    for action in env.legal_actions():
        piece, steps = env.decode_action(action)
        if env_mod.advance(env.positions[player][piece], steps) in opponent_positions:
            return 1.0
    return 0.0


def _finish_available(env, player: int) -> float:
    if player != env.current_player:
        return 0.0
    env_mod = _env_mod()
    for action in env.legal_actions():
        piece, steps = env.decode_action(action)
        if env_mod.advance(env.positions[player][piece], steps) == env_mod.FINISH:
            return 1.0
    return 0.0


def _shortcut_available(env, player: int) -> float:
    if player != env.current_player:
        return 0.0
    env_mod = _env_mod()
    for action in env.legal_actions():
        piece, steps = env.decode_action(action)
        old_pos = env.positions[player][piece]
        new_pos = env_mod.advance(old_pos, steps)
        if old_pos not in (env_mod.START, env_mod.FINISH) and new_pos in {4, 9, 24}:
            return 1.0
    return 0.0


def _capture_risk(env, player: int, max_steps: int = 1) -> float:
    env_mod = _env_mod()
    my_positions = [pos for pos in env.positions[player] if pos not in (env_mod.START, env_mod.FINISH)]
    if not my_positions:
        return 0.0
    step_probs = {steps: prob for _, steps, prob, _ in env_mod.YUT_OUTCOMES if steps <= max_steps}
    risk = 0.0
    for opp_pos in env.positions[1 - player]:
        if opp_pos == env_mod.FINISH:
            continue
        for steps, prob in step_probs.items():
            target = env_mod.advance(opp_pos, steps)
            if target in my_positions:
                risk += prob * env.positions[player].count(target)
    return min(1.0, risk)


def _is_in_danger(env, player: int) -> float:
    return 1.0 if _capture_risk(env, player, max_steps=5) > 0 else 0.0


def _safe_action_exists(env, player: int) -> float:
    if player != env.current_player:
        return 0.0
    env_mod = _env_mod()
    for action in env.legal_actions():
        candidate = env.clone()
        _make_simulation_safe(candidate)
        candidate.step(action)
        if _capture_risk(candidate, player, max_steps=5) == 0:
            return 1.0
    return 0.0


def _risky_action_ratio(env, player: int) -> float:
    if player != env.current_player:
        return 0.0
    legal = env.legal_actions()
    if not legal:
        return 0.0
    risky = 0
    for action in legal:
        candidate = env.clone()
        _make_simulation_safe(candidate)
        candidate.step(action)
        risky += int(_capture_risk(candidate, player, max_steps=5) > 0)
    return risky / len(legal)


def _make_simulation_safe(env) -> None:
    env.state_encoder = LegacyStateEncoder()
    env.state_encoder_name = "legacy"
    env.reward_function = None
    env.reward_function_name = "legacy"


@dataclass(frozen=True)
class LegacyStateEncoder(StateEncoder):
    name: str = "legacy"

    def encode(self, env, player: int | None = None) -> list[float]:
        actor, opponent = _perspective(env, player)
        encoded: list[float] = []
        for pos in env.positions[actor] + env.positions[opponent]:
            encoded.extend(_one_hot_position(pos))
        encoded.extend(_pending_features(env))
        encoded.append(1.0 if env.current_player == actor else 0.0)
        return encoded


@dataclass(frozen=True)
class RawStateEncoder(StateEncoder):
    name: str = "raw"

    def encode(self, env, player: int | None = None) -> list[float]:
        env_mod = _env_mod()
        actor, opponent = _perspective(env, player)
        features: list[float] = []
        for pos in env.positions[actor] + env.positions[opponent]:
            features.extend(_one_hot_position(pos))
        features.extend(_pending_features(env))
        features.extend(
            [
                env.positions[actor].count(env_mod.FINISH) / 4,
                env.positions[opponent].count(env_mod.FINISH) / 4,
                1.0 if env.current_player == actor else 0.0,
            ]
        )
        return features


@dataclass(frozen=True)
class BoardStateEncoder(StateEncoder):
    name: str = "board"

    def encode(self, env, player: int | None = None) -> list[float]:
        env_mod = _env_mod()
        actor, opponent = _perspective(env, player)
        features: list[float] = []
        for positions in (env.positions[actor], env.positions[opponent]):
            for pos in positions:
                features.extend(_one_hot_position(pos))
                route = env_mod.route_for(pos) if pos != env_mod.FINISH else [env_mod.FINISH]
                features.extend(
                    [
                        1.0 if route == env_mod.OUTER + [env_mod.FINISH] else 0.0,
                        1.0 if route == env_mod.SHORTCUT_A else 0.0,
                        1.0 if route == env_mod.SHORTCUT_B else 0.0,
                        1.0 if route == env_mod.CENTER_TO_FINISH else 0.0,
                    ]
                )
                features.append(env_mod.distance_to_finish(pos) / 28)
        features.extend(_pending_features(env))
        return features


@dataclass(frozen=True)
class EngineeredStateEncoder(RawStateEncoder):
    name: str = "engineered"

    def encode(self, env, player: int | None = None) -> list[float]:
        env_mod = _env_mod()
        actor, opponent = _perspective(env, player)
        features = super().encode(env, actor)
        features.extend(
            [
                _captures_available(env, actor),
                _finish_available(env, actor),
                _shortcut_available(env, actor),
                _is_in_danger(env, actor),
            ]
        )
        features.extend(env_mod.distance_to_finish(pos) / 28 for pos in env.positions[actor])
        features.extend(env_mod.distance_to_finish(pos) / 28 for pos in env.positions[opponent])
        features.extend(
            [
                max(env.positions[actor].count(pos) for pos in env.positions[actor] if pos not in (env_mod.START, env_mod.FINISH)) / 4
                if any(pos not in (env_mod.START, env_mod.FINISH) for pos in env.positions[actor])
                else 0.0,
                len(env.legal_actions()) / env.action_dim if actor == env.current_player else 0.0,
            ]
        )
        return features


@dataclass(frozen=True)
class RiskAwareStateEncoder(EngineeredStateEncoder):
    name: str = "risk_aware"

    def encode(self, env, player: int | None = None) -> list[float]:
        actor, _ = _perspective(env, player)
        features = super().encode(env, actor)
        features.extend(
            [
                _capture_risk(env, actor, max_steps=1),
                _capture_risk(env, actor, max_steps=2),
                1.0 if _opponent_counterplay_possible(env, actor) else 0.0,
                _safe_action_exists(env, actor),
                _risky_action_ratio(env, actor),
            ]
        )
        return features


def _opponent_counterplay_possible(env, player: int) -> bool:
    opponent = 1 - player
    if env.current_player != opponent:
        return False
    return bool(_captures_available(env, opponent) or _finish_available(env, opponent))


def get_state_encoder(name: str | StateEncoder | None) -> StateEncoder:
    if isinstance(name, StateEncoder):
        return name
    key = (name or "legacy").lower()
    if key == "legacy":
        return LegacyStateEncoder()
    if key == "raw":
        return RawStateEncoder()
    if key == "board":
        return BoardStateEncoder()
    if key == "engineered":
        return EngineeredStateEncoder()
    if key in {"risk", "risk_aware", "risk-aware"}:
        return RiskAwareStateEncoder()
    raise ValueError(f"unknown state_encoder: {name}")
