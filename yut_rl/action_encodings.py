from __future__ import annotations

from dataclasses import dataclass


PIECES_PER_PLAYER = 4
MAX_STEPS = 5
ACTION_DIM = PIECES_PER_PLAYER * MAX_STEPS
YUT_STEP_ORDER = (1, 2, 3, 4, 5)


class ActionEncoding:
    name = "base"
    action_dim = ACTION_DIM

    def encode(self, piece: int, steps: int) -> int:
        raise NotImplementedError

    def decode(self, action: int) -> tuple[int, int]:
        raise NotImplementedError

    def legal_actions(self, env) -> list[int]:
        env._ensure_pending_roll()
        legal = []
        available_steps = sorted(set(env.pending_steps))
        for piece, pos in enumerate(env.positions[env.current_player]):
            if pos == env.finish_position:
                continue
            if pos == env.start_position or env._is_stack_leader(env.current_player, piece):
                for steps in available_steps:
                    legal.append(self.encode(piece, steps))
        return sorted(legal)

    def action_mask(self, env) -> list[int]:
        mask = [0] * self.action_dim
        for action in self.legal_actions(env):
            mask[action] = 1
        return mask


@dataclass(frozen=True)
class StepActionEncoding(ActionEncoding):
    """Existing step-major encoding: action = (steps - 1) * 4 + piece_id."""

    name: str = "step"
    action_dim: int = ACTION_DIM

    def encode(self, piece: int, steps: int) -> int:
        return (steps - 1) * PIECES_PER_PLAYER + piece

    def decode(self, action: int) -> tuple[int, int]:
        return action % PIECES_PER_PLAYER, action // PIECES_PER_PLAYER + 1


@dataclass(frozen=True)
class PieceYutActionEncoding(ActionEncoding):
    """Piece-major encoding: action = piece_id * 5 + yut_type_id."""

    name: str = "piece_yut"
    action_dim: int = ACTION_DIM

    def encode(self, piece: int, steps: int) -> int:
        return piece * MAX_STEPS + (steps - 1)

    def decode(self, action: int) -> tuple[int, int]:
        return action // MAX_STEPS, action % MAX_STEPS + 1


def get_action_encoding(name: str | ActionEncoding | None) -> ActionEncoding:
    if isinstance(name, ActionEncoding):
        return name
    key = (name or "step").lower()
    if key in {"step", "steps", "step_action", "step_action_encoding"}:
        return StepActionEncoding()
    if key in {"piece_yut", "piece-yut", "piece", "piece_yut_action_encoding"}:
        return PieceYutActionEncoding()
    raise ValueError(f"unknown action_encoding: {name}")
