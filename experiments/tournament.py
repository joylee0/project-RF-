from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MPL_CONFIG_DIR = ROOT / "results" / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from yut_rl.agents import (
    A2CAgent,
    DQNAgent,
    DoubleDQNAgent,
    DuelingDQNAgent,
    MCTSValueAgent,
    PPOAgent,
    RandomAgent,
    ReinforceAgent,
    RuleBasedAgent,
    StrategicRuleBasedAgent,
    StrategicValueNetworkAgent,
    TabularQAgent,
    ValueNetworkAgent,
)
from yut_rl.env import FINISH, YutEnv
from yut_rl.train import train_dqn


NEURAL_TYPES = (
    DQNAgent,
    ValueNetworkAgent,
    MCTSValueAgent,
    ReinforceAgent,
    A2CAgent,
    PPOAgent,
)


@dataclass(frozen=True)
class AgentSpec:
    name: str
    kind: str
    factory: callable


def state_dim(seed: int) -> int:
    env = YutEnv(seed=seed)
    return len(env.reset())


def try_load(agent, path: str | None) -> None:
    if not path or not Path(path).exists() or not hasattr(agent, "load"):
        return
    try:
        agent.load(path)
    except Exception:
        return


def make_specs(seed: int, value_model: str | None, dqn_model: str | None, mcts_simulations: int) -> list[AgentSpec]:
    dim = state_dim(seed)

    def dqn_factory(cls, offset: int):
        def make():
            agent = cls(state_dim=dim, seed=seed + offset)
            try_load(agent, dqn_model)
            return agent

        return make

    def value_factory(cls, offset: int):
        def make():
            agent = cls(state_dim=dim, seed=seed + offset)
            try_load(agent, value_model)
            return agent

        return make

    def mcts_factory():
        base = ValueNetworkAgent(state_dim=dim, seed=seed + 80)
        try_load(base, value_model)
        return MCTSValueAgent(base, simulations=mcts_simulations, rollout_depth=4, seed=seed + 81)

    return [
        AgentSpec("random", "baseline", lambda: RandomAgent(seed=seed + 1)),
        AgentSpec("rule_based", "baseline", RuleBasedAgent),
        AgentSpec("strategic_rule", "scoring", lambda: StrategicRuleBasedAgent(seed=seed + 2)),
        AgentSpec("tabular", "tabular", lambda: TabularQAgent(seed=seed + 3)),
        AgentSpec("value", "value", value_factory(ValueNetworkAgent, 10)),
        AgentSpec("strategic_value", "value", value_factory(StrategicValueNetworkAgent, 11)),
        AgentSpec("dqn", "dqn", dqn_factory(DQNAgent, 20)),
        AgentSpec("double_dqn", "dqn", dqn_factory(DoubleDQNAgent, 21)),
        AgentSpec("dueling_dqn", "dqn", dqn_factory(DuelingDQNAgent, 22)),
        AgentSpec("reinforce", "policy", lambda: ReinforceAgent(state_dim=dim, seed=seed + 30)),
        AgentSpec("a2c", "policy", lambda: A2CAgent(state_dim=dim, seed=seed + 31)),
        AgentSpec("ppo", "policy", lambda: PPOAgent(state_dim=dim, seed=seed + 32)),
        AgentSpec("mcts_value", "search", mcts_factory),
    ]


def choose_action(agent, env: YutEnv) -> int:
    if isinstance(agent, NEURAL_TYPES):
        return agent.act(env, epsilon=0.0)
    return agent.act(env)


def play_game(agent0, name0: str, agent1, name1: str, seed: int) -> dict:
    env = YutEnv(seed=seed)
    env.reset()
    names = [name0, name1]
    agents = [agent0, agent1]
    captures = {name0: 0, name1: 0}
    turns = 0

    while True:
        player = env.current_player
        result = env.step(choose_action(agents[player], env))
        turns += 1
        actor = result.info.get("player")
        if actor is not None and result.info.get("captured"):
            captures[names[actor]] += 1

        if result.done:
            winner = env.winner()
            return {
                "winner": names[winner] if winner is not None else None,
                "turns": turns,
                "captures": captures,
                "finished": {
                    name0: env.positions[0].count(FINISH),
                    name1: env.positions[1].count(FINISH),
                },
            }


def play_match(spec_a: AgentSpec, spec_b: AgentSpec, games: int, seed: int) -> tuple[list[dict], list[dict]]:
    totals = {
        spec_a.name: {"wins": 0, "turns": 0, "captures": 0, "finished": 0},
        spec_b.name: {"wins": 0, "turns": 0, "captures": 0, "finished": 0},
    }
    rows = []
    agent_a = spec_a.factory()
    agent_b = spec_b.factory()

    for game in range(games):
        if game % 2 == 0:
            result = play_game(agent_a, spec_a.name, agent_b, spec_b.name, seed + game)
        else:
            result = play_game(agent_b, spec_b.name, agent_a, spec_a.name, seed + game)

        for name in (spec_a.name, spec_b.name):
            totals[name]["wins"] += int(result["winner"] == name)
            totals[name]["turns"] += result["turns"]
            totals[name]["captures"] += result["captures"][name]
            totals[name]["finished"] += result["finished"][name]

    for spec in (spec_a, spec_b):
        stats = totals[spec.name]
        rows.append(
            {
                "matchup": f"{spec_a.name}_vs_{spec_b.name}",
                "agent": spec.name,
                "kind": spec.kind,
                "games": games,
                "win_rate": stats["wins"] / games,
                "avg_turns": stats["turns"] / games,
                "avg_captures": stats["captures"] / games,
                "avg_finished": stats["finished"] / games,
            }
        )
    return rows, [{"agent": name, **stats} for name, stats in totals.items()]


def run_tournament(specs: list[AgentSpec], games: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    match_rows = []
    totals = {spec.name: {"wins": 0, "games": 0, "turns": 0, "captures": 0, "finished": 0} for spec in specs}
    for i, spec_a in enumerate(specs):
        for j, spec_b in enumerate(specs[i + 1 :], start=i + 1):
            rows, partial = play_match(spec_a, spec_b, games, seed + 10_000 * i + 500 * j)
            match_rows.extend(rows)
            for item in partial:
                name = item["agent"]
                totals[name]["wins"] += item["wins"]
                totals[name]["games"] += games
                totals[name]["turns"] += item["turns"]
                totals[name]["captures"] += item["captures"]
                totals[name]["finished"] += item["finished"]

    summary_rows = []
    for spec in specs:
        item = totals[spec.name]
        summary_rows.append(
            {
                "agent": spec.name,
                "kind": spec.kind,
                "games": item["games"],
                "overall_win_rate": item["wins"] / item["games"],
                "avg_turns": item["turns"] / item["games"],
                "avg_captures": item["captures"] / item["games"],
                "avg_finished": item["finished"] / item["games"],
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("overall_win_rate", ascending=False)
    return pd.DataFrame(match_rows), summary


def scoring_candidates(seed: int) -> list[tuple[str, StrategicRuleBasedAgent, dict]]:
    configs = [
        {"finish_weight": 130.0, "capture_weight": 55.0, "progress_weight": 3.0, "stack_weight": 7.0, "danger_weight": 28.0, "shortcut_weight": 8.0, "counterplay_weight": 18.0},
        {"finish_weight": 150.0, "capture_weight": 50.0, "progress_weight": 4.0, "stack_weight": 7.0, "danger_weight": 32.0, "shortcut_weight": 10.0, "counterplay_weight": 18.0},
        {"finish_weight": 170.0, "capture_weight": 45.0, "progress_weight": 4.5, "stack_weight": 8.0, "danger_weight": 36.0, "shortcut_weight": 12.0, "counterplay_weight": 20.0},
        {"finish_weight": 160.0, "capture_weight": 55.0, "progress_weight": 3.5, "stack_weight": 9.0, "danger_weight": 36.0, "shortcut_weight": 8.0, "counterplay_weight": 18.0},
    ]
    out = []
    for idx, cfg in enumerate(configs):
        out.append((f"scoring_cfg_{idx}", StrategicRuleBasedAgent(seed=seed + idx, **cfg), cfg))
    return out


def compare_against_field(candidate_name: str, candidate, specs: list[AgentSpec], games: int, seed: int) -> dict:
    wins = games_total = turns = captures = finished = 0
    for idx, opponent in enumerate(specs):
        if opponent.name == candidate_name:
            continue
        rows, _ = play_match(
            AgentSpec(candidate_name, "candidate", lambda candidate=candidate: candidate),
            opponent,
            games,
            seed + idx * 20_000,
        )
        row = next(item for item in rows if item["agent"] == candidate_name)
        wins += row["win_rate"] * games
        games_total += games
        turns += row["avg_turns"] * games
        captures += row["avg_captures"] * games
        finished += row["avg_finished"] * games
    return {
        "agent": candidate_name,
        "games": games_total,
        "overall_win_rate": wins / games_total,
        "avg_turns": turns / games_total,
        "avg_captures": captures / games_total,
        "avg_finished": finished / games_total,
    }


def tune_scoring(specs: list[AgentSpec], games: int, seed: int) -> pd.DataFrame:
    rows = []
    for name, agent, params in scoring_candidates(seed):
        result = compare_against_field(name, agent, specs, games, seed)
        rows.append({**result, **params})
    return pd.DataFrame(rows).sort_values("overall_win_rate", ascending=False)


def tune_dqn(specs: list[AgentSpec], games: int, seed: int, episodes: int) -> pd.DataFrame:
    configs = [
        {"lr": 1e-3, "gamma": 0.97, "epsilon_start": 0.8, "epsilon_end": 0.05},
        {"lr": 5e-4, "gamma": 0.97, "epsilon_start": 0.7, "epsilon_end": 0.03},
        {"lr": 1e-3, "gamma": 0.99, "epsilon_start": 0.7, "epsilon_end": 0.05},
    ]
    rows = []
    for idx, cfg in enumerate(configs):
        args = argparse.Namespace(
            agent="dqn",
            seed=seed + idx,
            hidden_dim=256,
            lr=cfg["lr"],
            gamma=cfg["gamma"],
            grad_clip=1.0,
            load_model=None,
            replay_capacity=50_000,
            episodes=episodes,
            epsilon_start=cfg["epsilon_start"],
            epsilon_end=cfg["epsilon_end"],
            batch_size=64,
            target_sync=100,
            eval_interval=0,
            eval_games=20,
            reward_mode="hybrid",
            dense_reward_scale=0.5,
            best_model=f"results/dqn_tuned_{idx}.pt",
        )
        agent = train_dqn(args)
        result = compare_against_field(f"dqn_tuned_{idx}", agent, specs, games, seed + idx * 30_000)
        rows.append({**result, **cfg, "episodes": episodes})
    return pd.DataFrame(rows).sort_values("overall_win_rate", ascending=False)


def plot_bars(df: pd.DataFrame, x: str, y: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    df.plot(kind="bar", x=x, y=y, ax=ax, legend=False, color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(y)
    ax.tick_params(axis="x", labelrotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=1_000)
    parser.add_argument("--fine-games", type=int, default=1_000)
    parser.add_argument("--dqn-episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=330_000)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--value-model", type=str, default="checkpoints/value.pt")
    parser.add_argument("--dqn-model", type=str, default=None)
    parser.add_argument("--mcts-simulations", type=int, default=4)
    parser.add_argument("--agents", type=str, default="all")
    parser.add_argument("--out-dir", type=str, default="results/tournament")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_specs = make_specs(args.seed, args.value_model, args.dqn_model, args.mcts_simulations)
    if args.agents == "all":
        specs = all_specs
    else:
        wanted = {name.strip() for name in args.agents.split(",") if name.strip()}
        specs = [spec for spec in all_specs if spec.name in wanted]

    matchups, summary = run_tournament(specs, args.games, args.seed)
    selected = summary.head(args.top_k)

    selected_names = set(selected["agent"])
    fine_field = [spec for spec in specs if spec.name in selected_names]
    if not fine_field:
        fine_field = specs

    scoring_before = summary[summary["agent"] == "strategic_rule"]
    scoring_after = tune_scoring(fine_field, args.fine_games, args.seed + 700_000)
    dqn_before = summary[summary["agent"] == "dqn"]
    dqn_after = tune_dqn(fine_field, args.fine_games, args.seed + 800_000, args.dqn_episodes)

    fine_summary = pd.concat(
        [
            scoring_before.assign(stage="before", family="scoring"),
            scoring_after.head(1).assign(stage="after", family="scoring"),
            dqn_before.assign(stage="before", family="dqn"),
            dqn_after.head(1).assign(stage="after", family="dqn"),
        ],
        ignore_index=True,
        sort=False,
    )

    matchups.to_csv(out_dir / "round_robin_matchups.csv", index=False)
    summary.to_csv(out_dir / "round_robin_summary.csv", index=False)
    selected.to_csv(out_dir / "selected_top_agents.csv", index=False)
    scoring_after.to_csv(out_dir / "scoring_grid_search.csv", index=False)
    dqn_after.to_csv(out_dir / "dqn_grid_search.csv", index=False)
    fine_summary.to_csv(out_dir / "fine_tuning_comparison.csv", index=False)

    plot_bars(summary, "agent", "overall_win_rate", "Round-robin win rate", out_dir / "round_robin_win_rate.png")
    plot_bars(fine_summary, "agent", "overall_win_rate", "Fine-tuning before/after", out_dir / "fine_tuning_win_rate.png")

    best_rows = [
        scoring_after.iloc[0].to_dict(),
        dqn_after.iloc[0].to_dict(),
        summary.iloc[0].to_dict(),
    ]
    best = max(best_rows, key=lambda row: row["overall_win_rate"])

    print("[implemented agents]")
    for spec in all_specs:
        print(f"- {spec.name} ({spec.kind})")
    print()
    print("[round-robin top agents]")
    print(selected.to_string(index=False))
    print()
    print("[fine-tuning comparison]")
    print(fine_summary[["family", "stage", "agent", "overall_win_rate", "avg_turns", "avg_captures", "avg_finished"]].to_string(index=False))
    print()
    print("[best agent]")
    print(best)
    print()
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()
