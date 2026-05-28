from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .agents import RuleBasedAgent, StrategicRuleBasedAgent, StrategicValueNetworkAgent, ValueNetworkAgent
from .compare import play_measured_game
from .env import YutEnv


def matchup_win_rate(baseline_name, baseline_agent, improved_name, improved_agent, games: int, seed: int) -> dict:
    wins = {baseline_name: 0, improved_name: 0}
    turns = {baseline_name: 0, improved_name: 0}
    captures = {baseline_name: 0, improved_name: 0}
    finished = {baseline_name: 0, improved_name: 0}
    rewards = {baseline_name: 0.0, improved_name: 0.0}

    for game in range(games):
        if game % 2 == 0:
            metrics = play_measured_game(baseline_agent, baseline_name, improved_agent, improved_name, seed + game)
        else:
            metrics = play_measured_game(improved_agent, improved_name, baseline_agent, baseline_name, seed + game)
        for name in (baseline_name, improved_name):
            wins[name] += int(metrics.winner_name == name)
            turns[name] += metrics.turns
            captures[name] += metrics.captures[name]
            finished[name] += metrics.finished[name]
            rewards[name] += metrics.rewards[name]

    return {
        "win_rate": wins[improved_name] / games,
        "baseline_win_rate": wins[baseline_name] / games,
        "avg_turns": turns[improved_name] / games,
        "avg_captures": captures[improved_name] / games,
        "avg_finished": finished[improved_name] / games,
        "avg_reward": rewards[improved_name] / games,
    }


def tune_rule(games: int, seed: int, confirm_games: int, top_k: int) -> pd.DataFrame:
    rows = []
    candidates = [
        (130.0, 55.0, 3.0, 7.0, 28.0, 8.0, 18.0),
        (160.0, 45.0, 4.5, 7.0, 36.0, 12.0, 18.0),
        (170.0, 45.0, 5.0, 7.0, 36.0, 12.0, 18.0),
        (180.0, 40.0, 5.0, 7.0, 36.0, 14.0, 18.0),
        (180.0, 45.0, 5.5, 7.0, 44.0, 14.0, 18.0),
        (190.0, 40.0, 5.5, 7.0, 44.0, 16.0, 18.0),
        (160.0, 50.0, 4.5, 9.0, 36.0, 12.0, 18.0),
        (170.0, 50.0, 5.0, 9.0, 44.0, 14.0, 18.0),
        (180.0, 50.0, 5.5, 9.0, 44.0, 16.0, 18.0),
    ]
    for idx, (finish, capture, progress, stack, danger, shortcut, counterplay) in enumerate(candidates):
        agent = StrategicRuleBasedAgent(
            finish_weight=finish,
            capture_weight=capture,
            progress_weight=progress,
            stack_weight=stack,
            danger_weight=danger,
            shortcut_weight=shortcut,
            counterplay_weight=counterplay,
            seed=seed + idx,
        )
        result = matchup_win_rate("rule_based", RuleBasedAgent(), "strategic_rule_based", agent, games, seed + idx * 10_000)
        rows.append(
            {
                "kind": "rule",
                "finish_weight": finish,
                "capture_weight": capture,
                "progress_weight": progress,
                "stack_weight": stack,
                "danger_weight": danger,
                "shortcut_weight": shortcut,
                "counterplay_weight": counterplay,
                "heuristic_weight": None,
                **result,
            }
        )

    quick = pd.DataFrame(rows).sort_values(["win_rate", "avg_reward"], ascending=False)
    confirmed = []
    for rank, row in quick.head(top_k).iterrows():
        agent = StrategicRuleBasedAgent(
            finish_weight=float(row["finish_weight"]),
            capture_weight=float(row["capture_weight"]),
            progress_weight=float(row["progress_weight"]),
            stack_weight=float(row["stack_weight"]),
            danger_weight=float(row["danger_weight"]),
            shortcut_weight=float(row["shortcut_weight"]),
            counterplay_weight=float(row["counterplay_weight"]),
            seed=seed + int(rank),
        )
        result = matchup_win_rate("rule_based", RuleBasedAgent(), "strategic_rule_based", agent, confirm_games, seed + 900_000 + int(rank))
        confirmed.append({**row.to_dict(), **{f"confirm_{k}": v for k, v in result.items()}})
    return pd.DataFrame(confirmed).sort_values(["confirm_win_rate", "confirm_avg_reward"], ascending=False)


def tune_value(games: int, seed: int, confirm_games: int, top_k: int, value_model: str | None) -> pd.DataFrame:
    env = YutEnv(seed=seed)
    state_dim = len(env.reset())
    rows = []
    for idx, heuristic_weight in enumerate([0.15, 0.25, 0.35, 0.45, 0.55, 0.65]):
        value = ValueNetworkAgent(state_dim=state_dim, seed=seed)
        strategic = StrategicValueNetworkAgent(state_dim=state_dim, heuristic_weight=heuristic_weight, seed=seed + idx)
        if value_model:
            value.load(value_model)
            strategic.load(value_model)
        result = matchup_win_rate("value", value, "strategic_value", strategic, games, seed + idx * 10_000)
        rows.append(
            {
                "kind": "value",
                "finish_weight": None,
                "capture_weight": None,
                "progress_weight": None,
                "stack_weight": None,
                "danger_weight": None,
                "shortcut_weight": None,
                "counterplay_weight": None,
                "heuristic_weight": heuristic_weight,
                **result,
            }
        )

    quick = pd.DataFrame(rows).sort_values(["win_rate", "avg_reward"], ascending=False)
    confirmed = []
    for rank, row in quick.head(top_k).iterrows():
        value = ValueNetworkAgent(state_dim=state_dim, seed=seed + int(rank))
        strategic = StrategicValueNetworkAgent(
            state_dim=state_dim,
            heuristic_weight=float(row["heuristic_weight"]),
            seed=seed + int(rank),
        )
        if value_model:
            value.load(value_model)
            strategic.load(value_model)
        result = matchup_win_rate("value", value, "strategic_value", strategic, confirm_games, seed + 950_000 + int(rank))
        confirmed.append({**row.to_dict(), **{f"confirm_{k}": v for k, v in result.items()}})
    return pd.DataFrame(confirmed).sort_values(["confirm_win_rate", "confirm_avg_reward"], ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--confirm-games", type=int, default=1_000)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=220_000)
    parser.add_argument("--value-model", type=str, default="checkpoints/value.pt")
    parser.add_argument("--output", type=str, default="results/tuning_results.csv")
    args = parser.parse_args()

    rule = tune_rule(args.games, args.seed, args.confirm_games, args.top_k)
    value = tune_value(args.games, args.seed + 50_000, args.confirm_games, args.top_k, args.value_model)
    df = pd.concat([rule, value], ignore_index=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(df.to_string(index=False))
    print()
    print(f"[saved] {output}")


if __name__ == "__main__":
    main()
