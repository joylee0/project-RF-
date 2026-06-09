from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MPL_CONFIG_DIR = ROOT / "results" / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import pandas as pd

from common_rule_based_env import CommonRuleBasedAgent, CommonYutEnv, FINISH, START
from experiments.common_rule_based_evaluation import (
    CommonPPOAdapter,
    CommonTacticalPPOAdapter,
    EvalResult,
    call_agent,
    count_captured,
)


class RandomCommonAgent:
    model_type = "Random baseline"

    def __init__(self, seed: int = 0):
        import random

        self.rng = random.Random(seed)

    def select_action(self, observation: list[float], legal_actions: list[int], env: CommonYutEnv | None = None) -> int:
        return self.rng.choice(legal_actions)


def play_paired_game(agent0, agent1, seed: int) -> EvalResult:
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


def make_agent(name: str, *, checkpoint: str | None, seed: int, tactical_weight: float):
    key = name.lower()
    if key in {"random", "random_agent"}:
        return RandomCommonAgent(seed=seed), "Random"
    if key in {"rule_based", "common_rule", "common_rule_based"}:
        return CommonRuleBasedAgent(), "Rule-based"
    if key in {"ppo_imitation", "ppo_capture_imitation", "ppo_tactical", "friend_ppo", "ppo"}:
        path = checkpoint or default_checkpoint_for(key)
        if path is None:
            raise ValueError(f"{name} requires --checkpoint")
        if not Path(path).exists():
            raise FileNotFoundError(f"checkpoint not found for {name}: {path}")
        if key == "ppo_tactical":
            return CommonTacticalPPOAdapter(path, seed=seed, tactical_weight=tactical_weight), "PPO + tactical prior"
        return CommonPPOAdapter(path, seed=seed), "PPO policy"
    raise ValueError(f"unknown agent: {name}")


def default_checkpoint_for(key: str) -> str | None:
    defaults = {
        "ppo_imitation": "results/ppo_training/ppo_imitation.pt",
        "ppo_capture_imitation": "results/ppo_training/ppo_capture_imitation.pt",
        "ppo_tactical": "results/ppo_training/ppo_tactical.pt",
        "friend_ppo": "results/ppo_training/ppo_tactical.pt",
        "ppo": "results/ppo_training/ppo_tactical.pt",
    }
    return defaults.get(key)


def run_paired_evaluation(my_agent, friend_agent, *, num_paired_seeds: int, seed: int) -> pd.DataFrame:
    rows = []
    for idx in range(num_paired_seeds):
        base_seed = seed + idx
        games = [
            {
                "pair_game": "A",
                "env_seed": base_seed,
                "first_agent": "my_agent",
                "second_agent": "friend_ppo",
                "agents": [my_agent, friend_agent],
            },
            {
                "pair_game": "B",
                "env_seed": base_seed,
                "first_agent": "friend_ppo",
                "second_agent": "my_agent",
                "agents": [friend_agent, my_agent],
            },
        ]
        for game in games:
            result = play_paired_game(game["agents"][0], game["agents"][1], game["env_seed"])
            my_player = 0 if game["first_agent"] == "my_agent" else 1
            friend_player = 1 - my_player
            rows.append(
                {
                    "base_seed": base_seed,
                    "pair_game": game["pair_game"],
                    "first_agent": game["first_agent"],
                    "second_agent": game["second_agent"],
                    "winner_player": result.winner,
                    "winner_agent": _winner_agent(result.winner, my_player),
                    "my_agent_player": my_player,
                    "friend_player": friend_player,
                    "my_agent_win": int(result.winner == my_player),
                    "friend_win": int(result.winner == friend_player),
                    "first_player_win": int(result.winner == 0),
                    "second_player_win": int(result.winner == 1),
                    "my_agent_as_first": int(my_player == 0),
                    "my_agent_as_second": int(my_player == 1),
                    "friend_as_first": int(friend_player == 0),
                    "friend_as_second": int(friend_player == 1),
                    "turns": result.decisions,
                    "captures_my_agent": result.captures[my_player],
                    "captures_friend": result.captures[friend_player],
                    "finished_pieces_my_agent": result.finished[my_player],
                    "finished_pieces_friend": result.finished[friend_player],
                    "illegal_my_agent": int(result.illegal_player == my_player),
                    "illegal_friend": int(result.illegal_player == friend_player),
                    "evaluation_error": result.evaluation_error or "",
                }
            )
    return pd.DataFrame(rows)


def _winner_agent(winner: int | None, my_player: int) -> str:
    if winner is None:
        return ""
    return "my_agent" if winner == my_player else "friend_ppo"


def summarize(games: pd.DataFrame, elapsed: float, paired_seeds: int) -> pd.DataFrame:
    pair_rows = []
    for base_seed, group in games.groupby("base_seed"):
        pair_rows.append(
            {
                "base_seed": base_seed,
                "paired_win_diff": float(group["my_agent_win"].sum() - group["friend_win"].sum()) / max(1, len(group)),
                "my_pair_win_rate": float(group["my_agent_win"].mean()),
            }
        )
    pair_df = pd.DataFrame(pair_rows)
    total_games = len(games)
    my_first = games[games["my_agent_as_first"] == 1]
    my_second = games[games["my_agent_as_second"] == 1]
    friend_first = games[games["friend_as_first"] == 1]
    friend_second = games[games["friend_as_second"] == 1]
    summary = {
        "total_games": total_games,
        "paired_seeds": paired_seeds,
        "my_agent_win_rate": _mean(games["my_agent_win"]),
        "friend_ppo_win_rate": _mean(games["friend_win"]),
        "first_player_win_rate": _mean(games["first_player_win"]),
        "second_player_win_rate": _mean(games["second_player_win"]),
        "my_agent_as_first_win_rate": _mean(my_first["my_agent_win"]),
        "my_agent_as_second_win_rate": _mean(my_second["my_agent_win"]),
        "friend_as_first_win_rate": _mean(friend_first["friend_win"]),
        "friend_as_second_win_rate": _mean(friend_second["friend_win"]),
        "avg_turns": _mean(games["turns"]),
        "avg_captures_my_agent": _mean(games["captures_my_agent"]),
        "avg_captures_friend": _mean(games["captures_friend"]),
        "avg_finished_pieces_my_agent": _mean(games["finished_pieces_my_agent"]),
        "avg_finished_pieces_friend": _mean(games["finished_pieces_friend"]),
        "paired_win_diff": _mean(pair_df["paired_win_diff"]),
        "seed_mean": _mean(pair_df["my_pair_win_rate"]),
        "seed_std": float(pair_df["my_pair_win_rate"].std(ddof=0)) if len(pair_df) > 1 else 0.0,
        "my_win_ci95": confidence_interval_95(int(games["my_agent_win"].sum()), total_games),
        "illegal_my_agent": int(games["illegal_my_agent"].sum()),
        "illegal_friend": int(games["illegal_friend"].sum()),
        "evaluation_error_count": int((games["evaluation_error"] != "").sum()),
        "total_eval_seconds": elapsed,
        "avg_eval_seconds_per_game": elapsed / max(1, total_games),
    }
    return pd.DataFrame([summary])


def _mean(series) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.mean())


def confidence_interval_95(wins: int, games: int) -> float:
    if games <= 0:
        return 0.0
    p = wins / games
    return 1.96 * math.sqrt(p * (1.0 - p) / games)


def save_plots(games: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"skipped plots: matplotlib unavailable ({exc})")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    row = summary.iloc[0]

    _bar(
        ["MyAgent", "FriendPPO"],
        [row["my_agent_win_rate"], row["friend_ppo_win_rate"]],
        "Win rate",
        output_dir / "win_rate_bar.png",
        ylim=(0, 1),
        ylabel="win_rate",
    )
    _bar(
        ["First player", "Second player", "My first", "My second", "Friend first", "Friend second"],
        [
            row["first_player_win_rate"],
            row["second_player_win_rate"],
            row["my_agent_as_first_win_rate"],
            row["my_agent_as_second_win_rate"],
            row["friend_as_first_win_rate"],
            row["friend_as_second_win_rate"],
        ],
        "First/second player win rate",
        output_dir / "first_second_win_rate.png",
        ylim=(0, 1),
        ylabel="win_rate",
    )
    _bar(
        ["MyAgent", "FriendPPO"],
        [row["avg_captures_my_agent"], row["avg_captures_friend"]],
        "Average captures",
        output_dir / "capture_comparison.png",
        ylabel="captures",
    )
    _bar(
        ["MyAgent", "FriendPPO"],
        [row["avg_finished_pieces_my_agent"], row["avg_finished_pieces_friend"]],
        "Average finished pieces",
        output_dir / "finished_pieces_comparison.png",
        ylabel="finished_pieces",
    )

    pair_diff = pd.Series(
        [
            float(group["my_agent_win"].sum() - group["friend_win"].sum()) / max(1, len(group))
            for _, group in games.groupby("base_seed")
        ]
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pair_diff, bins=[-1.0, -0.5, 0.0, 0.5, 1.0], color="#4C78A8", edgecolor="white")
    ax.set_title("Paired seed win diff")
    ax.set_xlabel("my_agent_win_minus_friend_win per pair")
    ax.set_ylabel("paired_seed_count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "paired_seed_win_diff.png", dpi=160)
    plt.close(fig)


def _bar(labels: list[str], values: list[float], title: str, path: Path, ylim=None, ylabel: str = "") -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2"][: len(labels)])
    ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_report(summary: pd.DataFrame, output_dir: Path, my_name: str, friend_name: str, my_type: str, friend_type: str) -> None:
    row = summary.iloc[0].to_dict()
    coverage_note = ""
    if int(row["total_games"]) < 5000 or int(row["paired_seeds"]) < 2500:
        coverage_note = (
            "\n> Note: this report was generated from a smoke/test run. "
            "Run 2,500 paired seeds / 5,000 games before treating the numbers as final.\n"
        )
    report = f"""# Common Paired Evaluation Report
{coverage_note}

## Project Overview

This report separates historical tournament results from the final common paired evaluation. The goal is to compare `{my_name}` and `{friend_name}` under the same common env, same paired seeds, and balanced first/second-player conditions.

## Historical Results

Earlier project results included PPO tournament tables, PPO improvement stages, StrategicValue direct matchups, and tactical PPO variants. Those results are useful as historical development records only. They are not mixed with the common paired evaluation below.

## Existing Agents

- RandomAgent: uniform legal-action baseline.
- RuleBasedAgent/CommonRuleBasedAgent: fixed heuristic baseline.
- PPO variants: masked PPO, imitation PPO, capture-aware PPO, tactical PPO.
- StrategicValueNetworkAgent: strong heuristic/value hybrid baseline, not treated as pure RL in this report.

## Why Common Env

Direct comparison was difficult when env rules, action encoding, reward shaping, opponent behavior, and tie-breaking differed. This evaluation fixes the board rules, yut probabilities, legal action mask, terminal condition, and paired seed schedule.

## Common Env Design

- Board rule: two-player Yutnori, four pieces per player, capture and stacking enabled.
- Yut probability: do 0.1536, gae 0.3456, geol 0.3456, yut 0.1296, mo 0.0256.
- State representation: common env observation from the active player's perspective.
- Action encoding: 20 discrete actions, piece x positive yut result.
- Reward function: terminal/common-eval reward only for evaluation; training reward is not used for final comparison.
- Terminal condition: all four pieces finished.
- Legal action mask: illegal actions are counted and lose the game in evaluation.

## Evaluation Method

- Common paired evaluation.
- Each base seed creates two games.
- Game A: MyAgent first, FriendPPO second.
- Game B: FriendPPO first, MyAgent second.
- Paired seeds: {int(row['paired_seeds'])}
- Total games: {int(row['total_games'])}

## Results

| Metric | Value |
| --- | ---: |
| MyAgent | {my_name} ({my_type}) |
| FriendPPO | {friend_name} ({friend_type}) |
| MyAgent win rate | {row['my_agent_win_rate']:.4f} |
| FriendPPO win rate | {row['friend_ppo_win_rate']:.4f} |
| MyAgent 95% CI half-width | {row['my_win_ci95']:.4f} |
| First player win rate | {row['first_player_win_rate']:.4f} |
| Second player win rate | {row['second_player_win_rate']:.4f} |
| MyAgent as first | {row['my_agent_as_first_win_rate']:.4f} |
| MyAgent as second | {row['my_agent_as_second_win_rate']:.4f} |
| Friend as first | {row['friend_as_first_win_rate']:.4f} |
| Friend as second | {row['friend_as_second_win_rate']:.4f} |
| Avg turns | {row['avg_turns']:.2f} |
| Avg captures MyAgent | {row['avg_captures_my_agent']:.2f} |
| Avg captures FriendPPO | {row['avg_captures_friend']:.2f} |
| Avg finished pieces MyAgent | {row['avg_finished_pieces_my_agent']:.2f} |
| Avg finished pieces FriendPPO | {row['avg_finished_pieces_friend']:.2f} |
| Paired win diff | {row['paired_win_diff']:.4f} |
| Seed mean | {row['seed_mean']:.4f} |
| Seed std | {row['seed_std']:.4f} |
| Illegal MyAgent | {int(row['illegal_my_agent'])} |
| Illegal FriendPPO | {int(row['illegal_friend'])} |
| Evaluation errors | {int(row['evaluation_error_count'])} |

## Interpretation

This is not an agent tournament. It is a fair common-env comparison with paired first/second-player games. First-player bias is estimated separately and softened by swapping sides for every base seed. The more stable agent is the one with higher paired mean and lower seed-level variance.

Capture and finished-piece differences help explain whether the win-rate gap comes from tactical captures, race-to-finish behavior, or unstable seed outcomes.

## Limitations

- If the friend's pretrained PPO model is unavailable, this report uses the supplied compatible checkpoint or a locally trained replacement.
- Different state/reward designs can still create learning-performance differences before evaluation.
- Even 5,000 games are stochastic, so seed effects can remain.

## Future Work

- Increase paired seeds.
- Add more confidence intervals.
- Compare under identical training budget.
- Extend state/reward ablations under the same common paired evaluator.

## Run Command

```bash
python experiments/common_paired_evaluation.py \\
  --my-agent ppo_capture_imitation \\
  --friend-agent friend_ppo \\
  --num-paired-seeds 2500 \\
  --total-games 5000 \\
  --seed 42 \\
  --output-dir results/common_paired_eval
```
"""
    (output_dir / "common_paired_report.md").write_text(report, encoding="utf-8")
    docs_dir = ROOT / "docs"
    docs_dir.mkdir(exist_ok=True)
    (docs_dir / "common_paired_evaluation_report.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Common paired evaluation for MyAgent vs FriendPPO.")
    parser.add_argument("--my-agent", default="ppo_capture_imitation")
    parser.add_argument("--friend-agent", default="friend_ppo")
    parser.add_argument("--my-checkpoint", default=None)
    parser.add_argument("--friend-checkpoint", default=None)
    parser.add_argument("--num-paired-seeds", type=int, default=2500)
    parser.add_argument("--total-games", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/common_paired_eval")
    parser.add_argument("--my-tactical-weight", type=float, default=4.0)
    parser.add_argument("--friend-tactical-weight", type=float, default=4.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.total_games != args.num_paired_seeds * 2:
        raise ValueError("--total-games must equal --num-paired-seeds * 2 for paired evaluation")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    my_agent, my_type = make_agent(
        args.my_agent,
        checkpoint=args.my_checkpoint,
        seed=args.seed,
        tactical_weight=args.my_tactical_weight,
    )
    friend_agent, friend_type = make_agent(
        args.friend_agent,
        checkpoint=args.friend_checkpoint,
        seed=args.seed + 10_000,
        tactical_weight=args.friend_tactical_weight,
    )

    start = time.perf_counter()
    games = run_paired_evaluation(my_agent, friend_agent, num_paired_seeds=args.num_paired_seeds, seed=args.seed)
    elapsed = time.perf_counter() - start
    summary = summarize(games, elapsed=elapsed, paired_seeds=args.num_paired_seeds)

    games.to_csv(output_dir / "common_paired_results.csv", index=False)
    summary.to_csv(output_dir / "common_paired_summary.csv", index=False)
    save_plots(games, summary, output_dir)
    write_report(summary, output_dir, args.my_agent, args.friend_agent, my_type, friend_type)
    print(summary.to_string(index=False))
    print(f"saved: {output_dir}")


if __name__ == "__main__":
    main()
