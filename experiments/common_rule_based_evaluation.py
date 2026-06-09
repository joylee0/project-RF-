from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

from agents.ppo_agent import MaskedPPOAgent, build_state, masked_logits, state_dim
from common_rule_based_env import (
    CommonRuleBasedAgent,
    CommonYutEnv,
    FINISH,
    START,
    YUT_OUTCOMES,
    advance,
    decode_action,
    distance_to_finish,
)


@dataclass
class EvalResult:
    winner: int | None
    decisions: int
    captures: list[int]
    finished: list[int]
    rewards: list[float]
    illegal_player: int | None = None
    evaluation_error: str | None = None


class CommonPPOAdapter:
    """Pure PPO-logit adapter for the common evaluation engine."""

    model_type = "Pure RL"

    def __init__(self, checkpoint: str, seed: int = 0):
        self.agent = MaskedPPOAgent(state_dim=state_dim(seed), seed=seed)
        self.agent.load(checkpoint)
        self.agent.body.eval()
        self.agent.policy.eval()
        self.agent.value.eval()

    def select_action(self, observation: list[float], legal_actions: list[int], env: CommonYutEnv | None = None) -> int:
        if env is None:
            return min(legal_actions)
        with torch.no_grad():
            state = torch.tensor(build_state(env), dtype=torch.float32).unsqueeze(0)
            logits, _ = self.agent.forward(state)
            logits = masked_logits(logits.squeeze(0), legal_actions)
            return int(torch.argmax(logits).item())


class CommonTacticalPPOAdapter(CommonPPOAdapter):
    """PPO adapter with a tunable common-rule tactical prior."""

    model_type = "Pure RL + tactical prior"

    def __init__(self, checkpoint: str, seed: int = 0, tactical_weight: float = 4.0):
        super().__init__(checkpoint, seed=seed)
        self.tactical_weight = tactical_weight

    def select_action(self, observation: list[float], legal_actions: list[int], env: CommonYutEnv | None = None) -> int:
        if env is None:
            return min(legal_actions)
        with torch.no_grad():
            state = torch.tensor(build_state(env), dtype=torch.float32).unsqueeze(0)
            logits, _ = self.agent.forward(state)
            logits = masked_logits(logits.squeeze(0), legal_actions)
            bonus = torch.zeros_like(logits)
            for action in legal_actions:
                bonus[action] = self._tactical_bonus(env, action)
            logits = logits + self.tactical_weight * bonus
            return int(torch.argmax(logits).item())

    def _tactical_bonus(self, env: CommonYutEnv, action: int) -> float:
        player = env.current_player
        opponent = 1 - player
        piece, _ = decode_action(action)
        before_me = env.positions[player][:]
        before_opp = env.positions[opponent][:]
        old_pos = before_me[piece]
        before_distance = sum(distance_to_finish(pos) for pos in before_me)
        before_finished = before_me.count(FINISH)

        candidate = env.clone()
        result = candidate.step(action)
        after_me = candidate.positions[player]
        after_opp = candidate.positions[opponent]
        new_pos = after_me[piece]
        finished_gain = after_me.count(FINISH) - before_finished
        captured_count = count_captured(before_opp, after_opp)
        progress_gain = before_distance - sum(distance_to_finish(pos) for pos in after_me)
        moving_count = before_me.count(old_pos) if old_pos not in (START, FINISH) else 1

        if result.done and candidate.winner() == player:
            return 30.0

        score = 0.0
        score += 9.0 * finished_gain
        score += 7.0 * captured_count
        score += 0.10 * progress_gain
        score += 0.8 * max(0, moving_count - 1)
        if old_pos == START and self._pieces_on_board(before_me) < 2:
            score += 0.6
        if self._can_capture(env, player) and not captured_count:
            score -= 2.5
        if not result.done:
            danger = self._capture_danger(candidate, player)
            score -= 4.0 * danger
            if candidate.current_player == opponent:
                score -= 1.8 * self._best_counterplay(candidate, opponent)
        if new_pos != FINISH:
            score -= 0.03 * distance_to_finish(new_pos)
        return score

    @staticmethod
    def _pieces_on_board(positions: list[int]) -> int:
        return sum(pos not in (START, FINISH) for pos in positions)

    @staticmethod
    def _can_capture(env: CommonYutEnv, player: int) -> bool:
        opponent = 1 - player
        targets = {pos for pos in env.positions[opponent] if pos not in (START, FINISH)}
        for action in env.legal_actions():
            piece, steps = decode_action(action)
            if advance(env.positions[player][piece], steps) in targets:
                return True
        return False

    @staticmethod
    def _capture_danger(env: CommonYutEnv, player: int) -> float:
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
                target = advance(opp_pos, steps)
                if target in my_positions:
                    danger += prob * env.positions[player].count(target)
        return danger

    def _best_counterplay(self, env: CommonYutEnv, opponent: int) -> float:
        legal = env.legal_actions()
        if not legal:
            return 0.0
        return max(self._counterplay_value(env, action, opponent) for action in legal)

    @staticmethod
    def _counterplay_value(env: CommonYutEnv, action: int, opponent: int) -> float:
        before_opp = env.positions[opponent][:]
        before_me = env.positions[1 - opponent][:]
        before_finished = before_opp.count(FINISH)
        before_distance = sum(distance_to_finish(pos) for pos in before_opp)

        candidate = env.clone()
        result = candidate.step(action)
        if result.done and candidate.winner() == opponent:
            return 10.0

        after_opp = candidate.positions[opponent]
        captured_count = count_captured(before_me, candidate.positions[1 - opponent])
        finished_gain = after_opp.count(FINISH) - before_finished
        progress_gain = before_distance - sum(distance_to_finish(pos) for pos in after_opp)
        return 3.0 * finished_gain + 1.5 * captured_count + 0.03 * progress_gain


def count_captured(before: list[int], after: list[int]) -> int:
    return sum(
        1
        for old, new in zip(before, after)
        if old not in (START, FINISH) and new == START
    )


def call_agent(agent, env: CommonYutEnv) -> int:
    observation = env.observe()
    legal_actions = env.legal_actions()
    if not legal_actions:
        raise RuntimeError("non-terminal state has no legal actions")
    try:
        return agent.select_action(observation, legal_actions, env=env)
    except TypeError:
        return agent.select_action(observation, legal_actions)


def play_game(agent0, agent1, seed: int) -> EvalResult:
    env = CommonYutEnv(seed=seed)
    env.reset(seed=seed)
    agents = [agent0, agent1]
    captures = [0, 0]
    rewards = [0.0, 0.0]

    while True:
        player = env.current_player
        before = [row[:] for row in env.positions]
        try:
            action = call_agent(agents[player], env)
        except Exception as exc:
            return EvalResult(
                winner=1 - player,
                decisions=env.decision_count,
                captures=captures,
                finished=[env.positions[0].count(FINISH), env.positions[1].count(FINISH)],
                rewards=rewards,
                evaluation_error=f"agent_exception:{type(exc).__name__}",
            )

        legal = env.legal_actions()
        if action not in legal:
            return EvalResult(
                winner=1 - player,
                decisions=env.decision_count + 1,
                captures=captures,
                finished=[env.positions[0].count(FINISH), env.positions[1].count(FINISH)],
                rewards=rewards,
                illegal_player=player,
            )

        result = env.step(action)
        actor = result.info.get("player", player)
        captures[actor] += count_captured(before[1 - actor], env.positions[1 - actor])
        rewards[actor] += result.reward

        if result.info.get("evaluation_error"):
            return EvalResult(
                winner=None,
                decisions=env.decision_count,
                captures=captures,
                finished=[env.positions[0].count(FINISH), env.positions[1].count(FINISH)],
                rewards=rewards,
                evaluation_error=result.info["evaluation_error"],
            )

        if result.done:
            return EvalResult(
                winner=env.winner(),
                decisions=env.decision_count,
                captures=captures,
                finished=[env.positions[0].count(FINISH), env.positions[1].count(FINISH)],
                rewards=rewards,
            )


def confidence_interval_95(wins: int, games: int) -> float:
    if games <= 0:
        return 0.0
    p = wins / games
    return 1.96 * math.sqrt(p * (1.0 - p) / games)


def evaluate_against_common_rule(agent, seed_count: int, seed_start: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    start_time = time.perf_counter()
    for idx in range(seed_count):
        base_seed = seed_start + idx
        games = [
            ("first", 0, play_game(agent, CommonRuleBasedAgent(), base_seed * 2)),
            ("second", 1, play_game(CommonRuleBasedAgent(), agent, base_seed * 2 + 1)),
        ]
        for side, model_player, result in games:
            rows.append(
                {
                    "base_seed": base_seed,
                    "side": side,
                    "model_player": model_player,
                    "winner": result.winner,
                    "model_win": int(result.winner == model_player),
                    "decisions": result.decisions,
                    "model_captures": result.captures[model_player],
                    "model_finished_pieces": result.finished[model_player],
                    "model_reward": result.rewards[model_player],
                    "illegal_action": int(result.illegal_player == model_player),
                    "evaluation_error": result.evaluation_error or "",
                }
            )
    elapsed = time.perf_counter() - start_time
    games_df = pd.DataFrame(rows)
    valid_games = len(games_df)
    wins = int(games_df["model_win"].sum())
    errors = int((games_df["evaluation_error"] != "").sum())
    illegal = int(games_df["illegal_action"].sum())
    first = games_df[games_df["side"] == "first"]
    second = games_df[games_df["side"] == "second"]
    summary = pd.DataFrame(
        [
            {
                "games": valid_games,
                "seed_count": seed_count,
                "win_rate": wins / valid_games,
                "win_rate_ci95": confidence_interval_95(wins, valid_games),
                "first_player_win_rate": first["model_win"].mean(),
                "second_player_win_rate": second["model_win"].mean(),
                "avg_decisions": games_df["decisions"].mean(),
                "avg_turns": games_df["decisions"].mean(),
                "avg_captures": games_df["model_captures"].mean(),
                "avg_finished_pieces": games_df["model_finished_pieces"].mean(),
                "illegal_action_count": illegal,
                "evaluation_error_count": errors,
                "total_eval_seconds": elapsed,
                "avg_eval_seconds_per_game": elapsed / valid_games,
            }
        ]
    )
    return summary, games_df


def search_tactical_weight(args) -> tuple[CommonTacticalPPOAdapter, str]:
    rows = []
    best_row = None
    best_agent = None
    for weight in args.tactical_weights:
        agent = CommonTacticalPPOAdapter(args.checkpoint, seed=args.seed_start, tactical_weight=weight)
        summary, _ = evaluate_against_common_rule(
            agent,
            seed_count=args.search_seed_count,
            seed_start=args.seed_start,
        )
        row = summary.iloc[0].to_dict()
        row["tactical_weight"] = weight
        rows.append(row)
        if best_row is None or row["win_rate"] > best_row["win_rate"]:
            best_row = row
            best_agent = agent

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "ppo_tactical_weight_search.csv", index=False)
    print(pd.DataFrame(rows)[["tactical_weight", "games", "win_rate", "avg_captures", "avg_finished_pieces"]].to_string(index=False))
    print(f"best tactical_weight={best_row['tactical_weight']} search_win_rate={best_row['win_rate']:.3f}")
    return best_agent, "Pure RL + tactical prior"


def make_agent(args):
    if args.agent == "common_rule":
        return CommonRuleBasedAgent(), "Rule-based"
    if args.agent == "ppo_imitation":
        return CommonPPOAdapter(args.checkpoint, seed=args.seed_start), "Pure RL"
    if args.agent == "ppo_tactical":
        if args.search_tactical_weight:
            return search_tactical_weight(args)
        return CommonTacticalPPOAdapter(args.checkpoint, seed=args.seed_start, tactical_weight=args.tactical_weight), "Pure RL + tactical prior"
    raise ValueError(f"unknown agent: {args.agent}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="common_rule", choices=["common_rule", "ppo_imitation", "ppo_tactical"])
    parser.add_argument("--name", default=None)
    parser.add_argument("--checkpoint", default="results/ppo_training/ppo_imitation.pt")
    parser.add_argument("--seed-count", type=int, default=2500)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--out-dir", default="results/common_rule_based_eval")
    parser.add_argument("--tactical-weight", type=float, default=4.0)
    parser.add_argument("--search-tactical-weight", action="store_true")
    parser.add_argument("--search-seed-count", type=int, default=200)
    parser.add_argument("--tactical-weights", type=float, nargs="+", default=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    agent, model_type = make_agent(args)
    summary, games = evaluate_against_common_rule(agent, seed_count=args.seed_count, seed_start=args.seed_start)
    agent_name = args.name or args.agent
    summary.insert(0, "agent", agent_name)
    summary.insert(1, "model_type", model_type)
    summary.to_csv(out_dir / f"{agent_name}_summary.csv", index=False)
    games.to_csv(out_dir / f"{agent_name}_games.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
