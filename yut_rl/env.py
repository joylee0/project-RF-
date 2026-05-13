from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

START = -1
FINISH = 99

YUT_OUTCOMES = (
    ("do", 1, 4 / 16, False),
    ("gae", 2, 6 / 16, False),
    ("geol", 3, 4 / 16, False),
    ("yut", 4, 1 / 16, True),
    ("mo", 5, 1 / 16, True),
)

OUTER = list(range(20))
SHORTCUT_A = [4, 20, 21, 24, 25, 26, FINISH]
SHORTCUT_B = [9, 22, 23, 24, 25, 26, FINISH]
CENTER_TO_FINISH = [24, 25, 26, FINISH]


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

    def __init__(self, seed: int | None = None, max_turns: int = 1000):
        self.rng = random.Random(seed)
        self.max_turns = max_turns
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
        new = YutEnv(max_turns=self.max_turns)
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
        me = self.positions[player]
        opp = self.positions[1 - player]
        encoded = [self._encode_pos(pos) for pos in me + opp]
        pending = self.pending_steps[0] if self.pending_steps else 0
        encoded.append(pending / 5)
        encoded.append(1.0 if self.current_player == player else 0.0)
        return encoded

    def legal_actions(self) -> list[int]:
        self._ensure_pending_roll()
        steps = self.pending_steps[0]
        legal = []
        for piece, pos in enumerate(self.positions[self.current_player]):
            if pos == FINISH:
                continue
            if pos == START or self._is_stack_leader(self.current_player, piece):
                legal.append(piece)
        return legal

    def step(self, action: int) -> StepResult:
        self._ensure_pending_roll()
        player = self.current_player
        opponent = 1 - player
        legal = self.legal_actions()
        if action not in legal:
            return StepResult(self.observe(), -1.0, False, {"illegal": True})

        steps = self.pending_steps.pop(0)
        before_finished = self.positions[player].count(FINISH)
        moving = self._stack_members(player, action)
        old_pos = self.positions[player][action]
        new_pos = advance(old_pos, steps)

        for piece in moving:
            self.positions[player][piece] = new_pos

        captured = False
        if new_pos != FINISH:
            captured = self._capture_at(opponent, new_pos)

        after_finished = self.positions[player].count(FINISH)
        done = after_finished == 4
        reward = -0.01
        reward += 0.2 * max(0, after_finished - before_finished)
        if captured:
            reward += 0.1
            self._roll_once()
        if done:
            reward += 1.0

        if not done and not self.pending_steps:
            self.current_player = opponent
            self.turn_count += 1
            if self.turn_count >= self.max_turns:
                done = True
                reward -= 0.5
            else:
                self._ensure_pending_roll()

        return StepResult(
            self.observe(),
            reward,
            done,
            {
                "player": player,
                "steps": steps,
                "from": old_pos,
                "to": new_pos,
                "captured": captured,
                "finished": done and after_finished == 4,
                "roll": self.last_roll_name,
            },
        )

    def winner(self) -> int | None:
        for player in (0, 1):
            if self.positions[player].count(FINISH) == 4:
                return player
        return None

    def _roll_once(self) -> None:
        r = self.rng.random()
        acc = 0.0
        for name, steps, prob, bonus in YUT_OUTCOMES:
            acc += prob
            if r <= acc:
                self.pending_steps.append(steps)
                self.last_roll_name = name
                if bonus:
                    self._roll_once()
                return

    def _ensure_pending_roll(self) -> None:
        if not self.pending_steps:
            self._roll_once()

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
    def _encode_pos(pos: int) -> float:
        if pos == START:
            return 0.0
        if pos == FINISH:
            return 1.0
        return (pos + 1) / 100


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
