from __future__ import annotations

import argparse
import json
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

from agents.ppo_agent import CaptureAwarePPOAgent, MaskedPPOAgent, can_capture, capture_danger, state_dim
from yut_rl.agents import StrategicRuleBasedAgent, StrategicValueNetworkAgent
from yut_rl.env import FINISH, START, YutEnv


def count_captured(before: list[int], after: list[int]) -> int:
    return sum(
        1
        for old, new in zip(before, after)
        if old not in (START, FINISH) and new == START
    )


def make_strategic_value(seed: int, value_model: str | None):
    agent = StrategicValueNetworkAgent(
        state_dim=len(YutEnv(seed=seed).reset()),
        heuristic_weight=0.45,
        heuristic_params={
            "finish_weight": 150.0,
            "capture_weight": 50.0,
            "progress_weight": 4.0,
            "danger_weight": 34.0,
            "counterplay_weight": 20.0,
        },
        seed=seed,
    )
    if value_model and Path(value_model).exists():
        agent.load(value_model)
    return agent


def load_ppo(path: Path, seed: int, tactical: bool = False) -> MaskedPPOAgent:
    cls = CaptureAwarePPOAgent if tactical else MaskedPPOAgent
    agent = cls(state_dim=state_dim(seed), seed=seed)
    agent.load(path)
    return agent


def choose_action(agent, env: YutEnv) -> int:
    if isinstance(agent, MaskedPPOAgent):
        return agent.act(env, deterministic=True)
    return agent.act(env)


def play_game(agent0, name0: str, agent1, name1: str, seed: int) -> dict:
    env = YutEnv(seed=seed)
    env.reset()
    agents = [agent0, agent1]
    names = [name0, name1]
    captures = {name0: 0, name1: 0}
    finished_total = {name0: 0, name1: 0}
    rewards = {name0: 0.0, name1: 0.0}
    capture_opportunities = {name0: 0, name1: 0}
    capture_successes = {name0: 0, name1: 0}
    danger_opportunities = {name0: 0, name1: 0}
    danger_avoids = {name0: 0, name1: 0}
    turns = 0

    while True:
        player = env.current_player
        before = [row[:] for row in env.positions]
        actor_name = names[player]
        had_capture = bool(can_capture(env, player))
        was_in_danger = bool(capture_danger(env, player))
        action = choose_action(agents[player], env)
        result = env.step(action)
        actor = result.info.get("player", player)
        actor_name = names[actor]
        captured = count_captured(before[1 - actor], env.positions[1 - actor])
        captures[actor_name] += captured
        if had_capture:
            capture_opportunities[actor_name] += 1
            if captured:
                capture_successes[actor_name] += 1
        if was_in_danger:
            danger_opportunities[actor_name] += 1
            if not capture_danger(env, actor):
                danger_avoids[actor_name] += 1
        rewards[actor_name] += result.reward
        if result.done:
            winner = env.winner()
            for idx, name in enumerate(names):
                finished_total[name] = env.positions[idx].count(FINISH)
            return {
                "winner": names[winner] if winner is not None else None,
                "winner_side": winner,
                "turns": turns + 1,
                "captures": captures,
                "finished": finished_total,
                "rewards": rewards,
                "capture_opportunities": capture_opportunities,
                "capture_successes": capture_successes,
                "danger_opportunities": danger_opportunities,
                "danger_avoids": danger_avoids,
            }
        turns += 1


def run_tournament(agents: list[tuple[str, object]], games: int, seed: int, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    totals = {
        name: {
            "wins": 0,
            "games": 0,
            "first_games": 0,
            "first_wins": 0,
            "second_games": 0,
            "second_wins": 0,
            "turns": 0,
            "captures": 0,
            "finished": 0,
            "reward": 0.0,
            "capture_opportunities": 0,
            "capture_successes": 0,
            "danger_opportunities": 0,
            "danger_avoids": 0,
        }
        for name, _ in agents
    }
    matchup_rows = []
    for i, (left_name, left_agent) in enumerate(agents):
        for j, (right_name, right_agent) in enumerate(agents[i + 1 :], start=i + 1):
            pair_wins = {left_name: 0, right_name: 0}
            pair_turns = 0
            pair_captures = {left_name: 0, right_name: 0}
            for game in range(games):
                game_seed = seed + i * 100_000 + j * 10_000 + game
                if game % 2 == 0:
                    result = play_game(left_agent, left_name, right_agent, right_name, game_seed)
                    first_name, second_name = left_name, right_name
                else:
                    result = play_game(right_agent, right_name, left_agent, left_name, game_seed)
                    first_name, second_name = right_name, left_name
                pair_wins[result["winner"]] += 1
                pair_turns += result["turns"]
                for name in (left_name, right_name):
                    totals[name]["wins"] += int(result["winner"] == name)
                    totals[name]["games"] += 1
                    if name == first_name:
                        totals[name]["first_games"] += 1
                        totals[name]["first_wins"] += int(result["winner"] == name)
                    if name == second_name:
                        totals[name]["second_games"] += 1
                        totals[name]["second_wins"] += int(result["winner"] == name)
                    totals[name]["turns"] += result["turns"]
                    totals[name]["captures"] += result["captures"][name]
                    totals[name]["finished"] += result["finished"][name]
                    totals[name]["reward"] += result["rewards"][name]
                    totals[name]["capture_opportunities"] += result["capture_opportunities"][name]
                    totals[name]["capture_successes"] += result["capture_successes"][name]
                    totals[name]["danger_opportunities"] += result["danger_opportunities"][name]
                    totals[name]["danger_avoids"] += result["danger_avoids"][name]
                    pair_captures[name] += result["captures"][name]
            matchup_rows.append(
                {
                    "agent_a": left_name,
                    "agent_b": right_name,
                    "games": games,
                    "agent_a_win_rate": pair_wins[left_name] / games,
                    "agent_b_win_rate": pair_wins[right_name] / games,
                    "avg_turns": pair_turns / games,
                    "agent_a_avg_captures": pair_captures[left_name] / games,
                    "agent_b_avg_captures": pair_captures[right_name] / games,
                }
            )
            print(f"{left_name} vs {right_name}: {pair_wins[left_name] / games:.3f} / {pair_wins[right_name] / games:.3f}")

    summary_rows = []
    for name, stats in totals.items():
        games_count = max(1, stats["games"])
        capture_opps = max(1, stats["capture_opportunities"])
        danger_opps = max(1, stats["danger_opportunities"])
        summary_rows.append(
            {
                "agent": name,
                "games": stats["games"],
                "overall_win_rate": stats["wins"] / games_count,
                "first_player_win_rate": stats["first_wins"] / max(1, stats["first_games"]),
                "second_player_win_rate": stats["second_wins"] / max(1, stats["second_games"]),
                "avg_turns": stats["turns"] / games_count,
                "avg_captures": stats["captures"] / games_count,
                "avg_finished_pieces": stats["finished"] / games_count,
                "avg_reward": stats["reward"] / games_count,
                "capture_opportunity_rate": stats["capture_opportunities"] / games_count,
                "capture_success_rate": stats["capture_successes"] / capture_opps,
                "avoid_capture_rate": stats["danger_avoids"] / danger_opps,
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("overall_win_rate", ascending=False)
    matchups = pd.DataFrame(matchup_rows)
    summary.to_csv(out_dir / "ppo_tournament_summary.csv", index=False)
    matchups.to_csv(out_dir / "ppo_tournament_matchups.csv", index=False)
    return summary, matchups


def plot_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ordered = summary.sort_values("overall_win_rate", ascending=True)
    axes[0, 0].barh(ordered["agent"], ordered["overall_win_rate"], color="#4c78a8")
    axes[0, 0].set_title("Win rate")
    axes[0, 0].set_xlim(0, 1)
    axes[0, 1].barh(ordered["agent"], ordered["avg_turns"], color="#72b7b2")
    axes[0, 1].set_title("Avg turns")
    axes[1, 0].barh(ordered["agent"], ordered["avg_captures"], color="#f58518")
    axes[1, 0].set_title("Avg captures")
    axes[1, 1].barh(ordered["agent"], ordered["avg_finished_pieces"], color="#54a24b")
    axes[1, 1].set_title("Avg finished pieces")
    fig.tight_layout()
    fig.savefig(out_dir / "ppo_tournament.png", dpi=160)
    plt.close(fig)

    for column, title, filename, color in [
        ("overall_win_rate", "Win rate", "ppo_win_rate.png", "#4c78a8"),
        ("avg_captures", "Average captures", "ppo_avg_captures.png", "#f58518"),
        ("capture_success_rate", "Capture success rate", "ppo_capture_success_rate.png", "#e45756"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        ordered = summary.sort_values(column, ascending=True)
        ax.barh(ordered["agent"], ordered[column], color=color)
        ax.set_title(title)
        if "rate" in column:
            ax.set_xlim(0, 1)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=160)
        plt.close(fig)


def plot_before_after(summary: pd.DataFrame, out_dir: Path) -> None:
    names = ["ppo_imitation", "ppo_capture_imitation", "ppo_tactical"]
    subset = summary[summary["agent"].isin(names)].copy()
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    present = set(subset["agent"])
    subset = subset.set_index("agent").reindex([name for name in names if name in present])
    ax.bar(subset.index, subset["overall_win_rate"], color=["#72b7b2", "#f58518", "#e45756"][: len(subset)])
    ax.set_ylim(0, 1)
    ax.set_title("PPO fine-tuning before/after")
    fig.tight_layout()
    fig.savefig(out_dir / "ppo_before_after_win_rate.png", dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="results/ppo_training")
    parser.add_argument("--out-dir", default="results/ppo_eval")
    parser.add_argument("--value-model", default="checkpoints/value.pt")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=91)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)

    agents = [
        ("strategic_value", make_strategic_value(args.seed, args.value_model)),
        ("strategic_rule", StrategicRuleBasedAgent(seed=args.seed + 1)),
    ]
    for idx, name in enumerate(["ppo_baseline", "ppo_masked", "ppo_curriculum", "ppo_imitation", "ppo_capture_imitation", "ppo_tactical"], start=10):
        path = model_dir / f"{name}.pt"
        if path.exists():
            agents.append((name, load_ppo(path, args.seed + idx, tactical=name in {"ppo_capture_imitation", "ppo_tactical"})))
        else:
            print(f"skip missing checkpoint: {path}")

    summary, _ = run_tournament(agents, args.games, args.seed + 10_000, out_dir)
    plot_summary(summary, out_dir)
    plot_before_after(summary, out_dir)

    ppo_rows = summary[summary["agent"].str.startswith("ppo_")]
    best_ppo = ppo_rows.sort_values("overall_win_rate", ascending=False).iloc[0].to_dict() if not ppo_rows.empty else {}
    params_path = model_dir / "ppo_params.json"
    params = json.loads(params_path.read_text(encoding="utf-8")) if params_path.exists() else {}
    result = {"best_ppo": best_ppo, "params": params}
    (out_dir / "best_ppo.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("best_ppo", result)


if __name__ == "__main__":
    main()
