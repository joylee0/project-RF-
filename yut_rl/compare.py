from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path

MPL_CONFIG_DIR = Path("results/.matplotlib").resolve()
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .agents import (
    A2CAgent,
    DQNAgent,
    MCTSValueAgent,
    PPOAgent,
    ReinforceAgent,
    RuleBasedAgent,
    StrategicRuleBasedAgent,
    StrategicValueNetworkAgent,
    ValueNetworkAgent,
)
from .env import FINISH, YutEnv


NEURAL_AGENT_TYPES = (
    DQNAgent,
    ValueNetworkAgent,
    MCTSValueAgent,
    ReinforceAgent,
    A2CAgent,
    PPOAgent,
)


@dataclass
class GameMetrics:
    winner_name: str | None
    turns: int
    captures: dict[str, int]
    finished: dict[str, int]
    rewards: dict[str, float]


def make_value_agents(seed: int, value_model: str | None = None, strategic_value_model: str | None = None):
    env = YutEnv(seed=seed)
    state_dim = len(env.reset())
    value = ValueNetworkAgent(state_dim=state_dim, seed=seed)
    strategic_value = StrategicValueNetworkAgent(state_dim=state_dim, seed=seed)
    if value_model:
        value.load(value_model)
        strategic_value.load(strategic_value_model or value_model)
    elif strategic_value_model:
        strategic_value.load(strategic_value_model)
    return value, strategic_value


def choose_action(agent, env: YutEnv) -> int:
    if isinstance(agent, NEURAL_AGENT_TYPES):
        return agent.act(env, epsilon=0.0)
    return agent.act(env)


def play_measured_game(agent0, name0: str, agent1, name1: str, seed: int) -> GameMetrics:
    env = YutEnv(seed=seed)
    env.reset()
    names = [name0, name1]
    agents = [agent0, agent1]
    captures = {name0: 0, name1: 0}
    rewards = {name0: 0.0, name1: 0.0}
    turns = 0

    while True:
        player = env.current_player
        action = choose_action(agents[player], env)
        result = env.step(action)
        turns += 1

        acting_name = names[player]
        rewards[acting_name] += result.reward
        if result.info.get("captured"):
            captures[acting_name] += 1

        if result.done:
            winner = env.winner()
            winner_name = names[winner] if winner is not None else None
            if winner is not None:
                rewards[names[1 - winner]] -= 1.0
            return GameMetrics(
                winner_name=winner_name,
                turns=turns,
                captures=captures,
                finished={
                    name0: env.positions[0].count(FINISH),
                    name1: env.positions[1].count(FINISH),
                },
                rewards=rewards,
            )


def run_matchup(baseline_name: str, baseline_agent, improved_name: str, improved_agent, games: int, seed: int) -> pd.DataFrame:
    if games < 1_000:
        raise ValueError("--games must be at least 1000 for the requested comparison.")

    totals = {
        baseline_name: {"wins": 0, "turns": 0, "captures": 0, "finished": 0, "rewards": 0.0},
        improved_name: {"wins": 0, "turns": 0, "captures": 0, "finished": 0, "rewards": 0.0},
    }

    for game in range(games):
        if game % 2 == 0:
            metrics = play_measured_game(baseline_agent, baseline_name, improved_agent, improved_name, seed + game)
        else:
            metrics = play_measured_game(improved_agent, improved_name, baseline_agent, baseline_name, seed + game)

        for name in (baseline_name, improved_name):
            totals[name]["wins"] += int(metrics.winner_name == name)
            totals[name]["turns"] += metrics.turns
            totals[name]["captures"] += metrics.captures[name]
            totals[name]["finished"] += metrics.finished[name]
            totals[name]["rewards"] += metrics.rewards[name]

    rows = []
    for kind, name in (("baseline", baseline_name), ("improved", improved_name)):
        rows.append(
            {
                "matchup": f"{baseline_name}_vs_{improved_name}",
                "agent_type": kind,
                "agent": name,
                "games": games,
                "win_rate": totals[name]["wins"] / games,
                "avg_turns": totals[name]["turns"] / games,
                "avg_captures": totals[name]["captures"] / games,
                "avg_finished": totals[name]["finished"] / games,
                "avg_reward": totals[name]["rewards"] / games,
            }
        )
    return pd.DataFrame(rows)


def plot_results(df: pd.DataFrame, output_path: Path) -> None:
    metrics = ["win_rate", "avg_turns", "avg_captures", "avg_finished", "avg_reward"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4), constrained_layout=True)
    for ax, metric in zip(axes, metrics):
        pivot = df.pivot(index="matchup", columns="agent_type", values=metric)
        pivot.plot(kind="bar", ax=ax, color=["#4C78A8", "#F58518"])
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelrotation=30)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="best")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=120_000)
    parser.add_argument("--value-model", type=str, default=None)
    parser.add_argument("--strategic-value-model", type=str, default=None)
    parser.add_argument("--csv", type=str, default="results/agent_comparison.csv")
    parser.add_argument("--plot", type=str, default="results/agent_comparison.png")
    args = parser.parse_args()

    value, strategic_value = make_value_agents(args.seed, args.value_model, args.strategic_value_model)
    frames = [
        run_matchup(
            "rule_based",
            RuleBasedAgent(),
            "strategic_rule_based",
            StrategicRuleBasedAgent(seed=args.seed),
            args.games,
            args.seed,
        ),
        run_matchup(
            "value",
            value,
            "strategic_value",
            strategic_value,
            args.games,
            args.seed + args.games + 10_000,
        ),
    ]
    df = pd.concat(frames, ignore_index=True)

    csv_path = Path(args.csv)
    plot_path = Path(args.plot)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    plot_results(df, plot_path)

    print(df.to_string(index=False))
    print()
    print(f"[saved csv] {csv_path}")
    print(f"[saved plot] {plot_path}")


if __name__ == "__main__":
    main()
