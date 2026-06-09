from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Protocol

START = -1
FINISH = 99
HOME = 19
PIECES_PER_PLAYER = 4
MAX_STEPS = 5
ACTION_DIM = PIECES_PER_PLAYER * MAX_STEPS
BOARD_POSITIONS = [START] + list(range(27)) + [FINISH]
POS_TO_INDEX = {pos: idx for idx, pos in enumerate(BOARD_POSITIONS)}
POSITION_FEATURES = len(BOARD_POSITIONS)

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


@dataclass
class CommonStep:
    observation: list[float]
    reward: float
    done: bool
    info: dict


def encode_action(piece: int, steps: int) -> int:
    return (steps - 1) * PIECES_PER_PLAYER + piece


def decode_action(action: int) -> tuple[int, int]:
    return action % PIECES_PER_PLAYER, action // PIECES_PER_PLAYER + 1


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


def advance(pos: int, steps: int) -> int:
    route = route_for(pos)
    idx = -1 if pos == START else route.index(pos)
    target_idx = idx + steps
    if target_idx >= len(route):
        return FINISH
    return route[target_idx]


def distance_to_finish(pos: int) -> int:
    if pos == FINISH:
        return 0
    route = route_for(pos)
    idx = -1 if pos == START else route.index(pos)
    return len(route) - idx - 1


class CommonYutEnv:
    """Official common Rule-based evaluation environment.

    Agent-visible API:
    - reset(seed=None) -> observation
    - observe() -> observation
    - legal_actions() -> list[int]
    - action_mask() -> list[int]
    - step(action) -> CommonStep
    - winner() -> int | None

    The environment does not expose its RNG object. Agents should only use the
    observation, legal actions/action mask, public game state, and fixed model
    state. A model returning an illegal action is handled by the evaluator.
    """

    def __init__(self, seed: int | None = None, max_decisions: int = 10_000):
        self._rng = random.Random(seed)
        self.max_decisions = max_decisions
        self.positions = [[START] * PIECES_PER_PLAYER, [START] * PIECES_PER_PLAYER]
        self.current_player = 0
        self.pending_steps: list[int] = []
        self.decision_count = 0
        self.last_roll_name: str | None = None
        self.evaluation_error: str | None = None

    def reset(self, seed: int | None = None) -> list[float]:
        if seed is not None:
            self._rng.seed(seed)
        self.positions = [[START] * PIECES_PER_PLAYER, [START] * PIECES_PER_PLAYER]
        self.current_player = 0
        self.pending_steps = []
        self.decision_count = 0
        self.last_roll_name = None
        self.evaluation_error = None
        self._ensure_pending_roll()
        return self.observe()

    def clone(self) -> "CommonYutEnv":
        new = CommonYutEnv(max_decisions=self.max_decisions)
        new._rng.setstate(self._rng.getstate())
        new.positions = [row[:] for row in self.positions]
        new.current_player = self.current_player
        new.pending_steps = self.pending_steps[:]
        new.decision_count = self.decision_count
        new.last_roll_name = self.last_roll_name
        new.evaluation_error = self.evaluation_error
        return new

    def observe(self) -> list[float]:
        return self.observe_for(self.current_player)

    def observe_for(self, player: int) -> list[float]:
        me = self.positions[player]
        opp = self.positions[1 - player]
        encoded: list[float] = []
        for pos in me + opp:
            encoded.extend(self._encode_pos(pos))
        for steps in range(1, MAX_STEPS + 1):
            encoded.append(min(self.pending_steps.count(steps), 4) / 4)
        encoded.append(1.0 if self.current_player == player else 0.0)
        return encoded

    def legal_actions(self) -> list[int]:
        self._ensure_pending_roll()
        legal = []
        available_steps = sorted(set(self.pending_steps))
        for piece, pos in enumerate(self.positions[self.current_player]):
            if pos == FINISH:
                continue
            if pos == START or self._is_stack_leader(self.current_player, piece):
                for steps in available_steps:
                    legal.append(encode_action(piece, steps))
        return sorted(legal)

    def action_mask(self) -> list[int]:
        mask = [0] * ACTION_DIM
        for action in self.legal_actions():
            mask[action] = 1
        return mask

    def step(self, action: int) -> CommonStep:
        player = self.current_player
        legal = self.legal_actions()
        if action not in legal:
            return CommonStep(
                observation=self.observe(),
                reward=-1.0,
                done=True,
                info={"illegal": True, "player": player, "winner": 1 - player},
            )

        piece, steps = decode_action(action)
        self.pending_steps.remove(steps)
        opponent = 1 - player
        old_pos = self.positions[player][piece]
        moving = self._stack_members(player, piece)
        before_finished = self.positions[player].count(FINISH)
        new_pos = advance(old_pos, steps)

        for moving_piece in moving:
            self.positions[player][moving_piece] = new_pos

        captured_count = 0
        if new_pos != FINISH:
            captured_count = self._capture_at(opponent, new_pos)

        after_finished = self.positions[player].count(FINISH)
        finished_count = max(0, after_finished - before_finished)
        done = after_finished == PIECES_PER_PLAYER
        reward = 1.0 if done else 0.0
        if captured_count:
            reward += 0.1
            self.pending_steps.extend(self._roll_turn_results())

        if not done and not self.pending_steps:
            self.current_player = opponent
            self._ensure_pending_roll()

        self.decision_count += 1
        if not done and self.decision_count > self.max_decisions:
            self.evaluation_error = "max_decisions_exceeded"
            done = True

        return CommonStep(
            observation=self.observe(),
            reward=reward,
            done=done,
            info={
                "player": player,
                "steps": steps,
                "from": old_pos,
                "to": new_pos,
                "captured": captured_count > 0,
                "captured_count": captured_count,
                "finished_count": finished_count,
                "winner": self.winner(),
                "evaluation_error": self.evaluation_error,
            },
        )

    def winner(self) -> int | None:
        for player in (0, 1):
            if self.positions[player].count(FINISH) == PIECES_PER_PLAYER:
                return player
        return None

    def score_common_rule_action(self, action: int) -> float:
        player = self.current_player
        opponent = 1 - player
        piece, steps = decode_action(action)
        old_pos = self.positions[player][piece]
        new_pos = advance(old_pos, steps)
        opp_positions = set(pos for pos in self.positions[opponent] if pos != START and pos != FINISH)
        stack_size = 1 if old_pos in (START, FINISH) else self.positions[player].count(old_pos)
        score = 0.0
        if new_pos == FINISH:
            score += 100.0
        if new_pos in opp_positions:
            score += 50.0
        if old_pos == START:
            score += 5.0
        score += 4.0 * max(0, stack_size - 1)
        score -= 0.5 * distance_to_finish(new_pos)
        return score

    def _roll_once(self) -> tuple[str, int, bool]:
        r = self._rng.random()
        acc = 0.0
        for name, steps, prob, bonus in YUT_OUTCOMES:
            acc += prob
            if r <= acc:
                self.last_roll_name = name
                return name, steps, bonus
        name, steps, _, bonus = YUT_OUTCOMES[-1]
        self.last_roll_name = name
        return name, steps, bonus

    def _roll_turn_results(self) -> list[int]:
        results = []
        while True:
            _, steps, bonus = self._roll_once()
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
        return [idx for idx, item in enumerate(self.positions[player]) if item == pos]

    def _is_stack_leader(self, player: int, piece: int) -> bool:
        pos = self.positions[player][piece]
        if pos in (START, FINISH):
            return True
        return piece == min(idx for idx, item in enumerate(self.positions[player]) if item == pos)

    def _capture_at(self, opponent: int, pos: int) -> int:
        captured = 0
        for piece, opp_pos in enumerate(self.positions[opponent]):
            if opp_pos == pos:
                self.positions[opponent][piece] = START
                captured += 1
        return captured

    @staticmethod
    def _encode_pos(pos: int) -> list[float]:
        encoded = [0.0] * POSITION_FEATURES
        encoded[POS_TO_INDEX[pos]] = 1.0
        return encoded


class CommonAgent(Protocol):
    def select_action(self, observation: list[float], legal_actions: list[int]) -> int:
        ...


class CommonRuleBasedAgent:
    """Frozen official Rule-based opponent."""

    model_type = "Rule-based"

    def select_action(self, observation: list[float], legal_actions: list[int], env: CommonYutEnv | None = None) -> int:
        if env is None:
            return min(legal_actions)
        return min(legal_actions, key=lambda action: (-env.score_common_rule_action(action), action))
