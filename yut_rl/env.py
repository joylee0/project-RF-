from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

from yut_rl.action_encodings import ACTION_DIM, get_action_encoding
from yut_rl.reward_functions import get_reward_function
from yut_rl.state_encoders import get_state_encoder

START = -1
FINISH = 99
HOME = 19
BOARD_POSITIONS = [START] + list(range(27)) + [FINISH]
POS_TO_INDEX = {pos: idx for idx, pos in enumerate(BOARD_POSITIONS)}
POSITION_FEATURES = len(BOARD_POSITIONS)
PIECES_PER_PLAYER = 4
MAX_STEPS = 5

YUT_OUTCOMES = (
    ("do", 1, 0.1536, False),
    ("gae", 2, 0.3456, False),
    ("geol", 3, 0.3456, False),
    ("yut", 4, 0.1296, True),
    ("mo", 5, 0.0256, True),
)

OUTER = list(range(20))
SHORTCUT_A = [4, 20, 21, 24, 25, 26, HOME, FINISH]
SHORTCUT_B = [9, 22, 23, 24, 25, 26, HOME, FINISH]
CENTER_TO_FINISH = [24, 25, 26, HOME, FINISH]


@dataclass(frozen=True)
class Move:
    piece: int
    steps: int


@dataclass
class StepResult:
    observation: list[float]
    reward: float
    done: bool
    info: dict


class YutEnv:
    """Two-player Yutnori environment with capture, stacking, bonus throws.

    This is intentionally small and dependency-free so it can be used for
    random, rule-based, tabular, or neural agents.
    """

    start_position = START
    finish_position = FINISH

    def __init__(
        self,
        seed: int | None = None,
        max_turns: int = 1000,
        state_encoder: str = "legacy",
        reward_function: str = "legacy",
        action_encoding: str = "step",
        enable_action_mask: bool = True,
    ):
        self.rng = random.Random(seed)
        self.max_turns = max_turns
        self.state_encoder_name = state_encoder
        self.reward_function_name = reward_function
        self.action_encoding_name = action_encoding
        self.state_encoder = get_state_encoder(state_encoder)
        self.reward_function = get_reward_function(reward_function)
        self.action_encoding = get_action_encoding(action_encoding)
        self.enable_action_mask = enable_action_mask
        self.action_dim = self.action_encoding.action_dim
        self.positions = [[START] * 4, [START] * 4]
        self.current_player = 0
        self.turn_count = 0
        self.pending_steps: list[int] = []
        self.last_roll_name: str | None = None

    def reset(self, seed: int | None = None) -> list[float]:
        if seed is not None:
            self.rng.seed(seed)
        self.positions = [[START] * 4, [START] * 4]
        self.current_player = 0
        self.turn_count = 0
        self.pending_steps = []
        self.last_roll_name = None
        self._ensure_pending_roll()
        return self.observe()

    def clone(self) -> "YutEnv":
        new = YutEnv(
            max_turns=self.max_turns,
            state_encoder=self.state_encoder_name,
            reward_function=self.reward_function_name,
            action_encoding=self.action_encoding_name,
            enable_action_mask=self.enable_action_mask,
        )
        new.rng.setstate(self.rng.getstate())
        new.positions = [row[:] for row in self.positions]
        new.current_player = self.current_player
        new.turn_count = self.turn_count
        new.pending_steps = self.pending_steps[:]
        new.last_roll_name = self.last_roll_name
        return new

    def observe(self) -> list[float]:
        return self.observe_for(self.current_player)

    def observe_for(self, player: int) -> list[float]:
        return self.state_encoder.encode(self, player)

    def legal_actions(self) -> list[int]:
        return self.action_encoding.legal_actions(self)

    def action_mask(self) -> list[int]:
        if not self.enable_action_mask:
            return [1] * self.action_dim
        return self.action_encoding.action_mask(self)

    def encode_action(self, piece: int, steps: int) -> int:
        return self.action_encoding.encode(piece, steps)

    def decode_action(self, action: int) -> tuple[int, int]:
        return self.action_encoding.decode(action)

    def step(self, action: int) -> StepResult:
        self._ensure_pending_roll()
        player = self.current_player
        opponent = 1 - player
        legal = self.legal_actions()
        if action not in legal:
            return StepResult(self.observe(), -1.0, False, {"illegal": True})

        before = self.clone() if self.reward_function is not None else None
        piece, steps = self.decode_action(action)
        self.pending_steps.remove(steps)
        before_finished = self.positions[player].count(FINISH)
        moving = self._stack_members(player, piece)
        old_pos = self.positions[player][piece]
        new_pos = advance(old_pos, steps)

        for piece in moving:
            self.positions[player][piece] = new_pos

        captured = False
        if new_pos != FINISH:
            captured = self._capture_at(opponent, new_pos)

        after_finished = self.positions[player].count(FINISH)
        done = after_finished == 4
        if captured:
            self.pending_steps.extend(self._roll_turn_results())

        if not done and not self.pending_steps:
            self.current_player = opponent
            self.turn_count += 1
            if self.turn_count >= self.max_turns:
                done = True
            else:
                self._ensure_pending_roll()

        info = {
            "player": player,
            "steps": steps,
            "from": old_pos,
            "to": new_pos,
            "captured": captured,
            "finished": done and after_finished == 4,
            "finished_count": max(0, after_finished - before_finished),
            "roll": self.last_roll_name,
        }
        if self.reward_function is None:
            reward = -0.01
            reward += 0.2 * max(0, after_finished - before_finished)
            if captured:
                reward += 0.1
            if done:
                reward += 1.0
            if done and self.turn_count >= self.max_turns:
                reward -= 0.5
        else:
            reward = self.reward_function(before, self, info, done, player)

        return StepResult(
            self.observe(),
            reward,
            done,
            info,
        )

    def winner(self) -> int | None:
        for player in (0, 1):
            if self.positions[player].count(FINISH) == 4:
                return player
        return None

    def _roll_once(self) -> tuple[str, int, bool]:
        r = self.rng.random()
        acc = 0.0
        for name, steps, prob, bonus in YUT_OUTCOMES:
            acc += prob
            if r <= acc:
                self.last_roll_name = name
                return name, steps, bonus

    def _roll_turn_results(self) -> list[int]:
        results = []
        while True:
            name, steps, bonus = self._roll_once()
            results.append(steps)
            if not bonus:
                return results

    def _ensure_pending_roll(self) -> None:
        if not self.pending_steps:
            self.pending_steps.extend(self._roll_turn_results())

    def _stack_members(self, player: int, piece: int) -> list[int]:
        pos = self.positions[player][piece]
        if pos in (START, FINISH):
            return [piece]
        return [i for i, p in enumerate(self.positions[player]) if p == pos]

    def _is_stack_leader(self, player: int, piece: int) -> bool:
        pos = self.positions[player][piece]
        if pos in (START, FINISH):
            return True
        return piece == min(i for i, p in enumerate(self.positions[player]) if p == pos)

    def _capture_at(self, opponent: int, pos: int) -> bool:
        captured = False
        for piece, opp_pos in enumerate(self.positions[opponent]):
            if opp_pos == pos:
                self.positions[opponent][piece] = START
                captured = True
        return captured

    @staticmethod
    def _encode_pos(pos: int) -> list[float]:
        encoded = [0.0] * POSITION_FEATURES
        encoded[POS_TO_INDEX[pos]] = 1.0
        return encoded


def encode_action(piece: int, steps: int) -> int:
    return (steps - 1) * PIECES_PER_PLAYER + piece


def decode_action(action: int) -> tuple[int, int]:
    piece = action % PIECES_PER_PLAYER
    steps = action // PIECES_PER_PLAYER + 1
    return piece, steps


def advance(pos: int, steps: int) -> int:
    route = route_for(pos)
    if pos == START:
        idx = -1
    else:
        idx = route.index(pos)
    target_idx = idx + steps
    if target_idx >= len(route):
        return FINISH
    return route[target_idx]


def route_for(pos: int) -> list[int]:
    if pos in SHORTCUT_A:
        return SHORTCUT_A
    if pos in SHORTCUT_B:
        return SHORTCUT_B
    if pos in CENTER_TO_FINISH:
        return CENTER_TO_FINISH
    if pos == START or pos in OUTER:
        return OUTER + [FINISH]
    raise ValueError(f"unknown board position: {pos}")


def distance_to_finish(pos: int) -> int:
    if pos == FINISH:
        return 0
    route = route_for(pos)
    idx = -1 if pos == START else route.index(pos)
    return len(route) - idx - 1


def occupied_positions(positions: Iterable[int]) -> set[int]:
    return {pos for pos in positions if pos not in (START, FINISH)}
