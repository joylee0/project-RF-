from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = ROOT / "results" / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired common-env evaluation: MyAgent(ProjectRF rule) vs TeamPPO.")
    parser.add_argument("--team-repo", default="/private/tmp/RL-yutnori-team-model")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--training-seed", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=100_000)
    parser.add_argument("--num-paired-seeds", type=int, default=2500)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--observation-mode", default="tactical")
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--max-decisions", type=int, default=10_000)
    parser.add_argument("--output-dir", default="results/team_ppo_vs_my_agent_common_eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    team_repo = Path(args.team_repo)
    if str(team_repo) not in sys.path:
        sys.path.insert(0, str(team_repo))

    import numpy as np
    from sb3_contrib import MaskablePPO
    from yutnori.agents import ProjectRFRuleBasedAgent
    from yutnori.core import ACTION_SIZE, GameState, YutSampler
    from yutnori.env import encode_observation

    model = MaskablePPO.load(args.model_path, device=args.device)
    if hasattr(model.policy, "set_training_mode"):
        model.policy.set_training_mode(False)

    base_seeds = list(range(args.seed_start, args.seed_start + args.num_paired_seeds))
    rows = []
    started = time.perf_counter()

    def team_action(state: GameState, legal_actions: list[int]) -> int:
        observation = encode_observation(state, state.current_player, observation_mode=args.observation_mode)
        action_mask = np.zeros(ACTION_SIZE, dtype=np.bool_)
        action_mask[legal_actions] = True
        action, _ = model.predict(observation, deterministic=args.deterministic, action_masks=action_mask)
        return int(np.asarray(action).item())

    for base_seed in base_seeds:
        rows.append(
            play_game(
                base_seed=base_seed,
                team_starts=True,
                team_action=team_action,
                my_agent=ProjectRFRuleBasedAgent(),
                max_decisions=args.max_decisions,
            )
        )
        rows.append(
            play_game(
                base_seed=base_seed,
                team_starts=False,
                team_action=team_action,
                my_agent=ProjectRFRuleBasedAgent(),
                max_decisions=args.max_decisions,
            )
        )

    elapsed = time.perf_counter() - started
    summary = summarize(rows, base_seeds, elapsed, args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "team_ppo_vs_my_agent_games.csv").write_text(to_csv(rows), encoding="utf-8")
    (out_dir / "team_ppo_vs_my_agent_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "team_ppo_vs_my_agent_report.md").write_text(report_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def play_game(*, base_seed: int, team_starts: bool, team_action, my_agent, max_decisions: int) -> dict:
    from yutnori.core import GameState, YutSampler

    team_player = 0
    my_player = 1
    starting_player = team_player if team_starts else my_player
    state = GameState(starting_player=starting_player, yut_sampler=YutSampler(seed=base_seed))
    state.start_turn()
    error = ""
    illegal_player = None

    while state.winner is None:
        legal_actions = state.get_legal_actions()
        if state.current_player == team_player:
            action = int(team_action(state, legal_actions))
        else:
            action = int(my_agent.select_action(state, legal_actions))

        if action not in legal_actions:
            illegal_player = state.current_player
            state.winner = 1 - state.current_player
            break
        state.apply_action(action)
        if state.decision_count > max_decisions:
            error = "max_decisions_exceeded"
            break

    team_win = int(state.winner == team_player)
    my_win = int(state.winner == my_player)
    return {
        "base_seed": base_seed,
        "pair_game": "A" if team_starts else "B",
        "team_starts": int(team_starts),
        "my_agent_starts": int(not team_starts),
        "winner": "" if state.winner is None else state.winner,
        "team_ppo_win": team_win,
        "my_agent_win": my_win,
        "first_player_win": int(state.winner == starting_player),
        "second_player_win": int(state.winner == 1 - starting_player),
        "turns": state.turn_count,
        "decisions": state.decision_count,
        "illegal_team_ppo": int(illegal_player == team_player),
        "illegal_my_agent": int(illegal_player == my_player),
        "evaluation_error": error,
    }


def summarize(rows: list[dict], base_seeds: list[int], elapsed: float, args: argparse.Namespace) -> dict:
    team_wins = sum(row["team_ppo_win"] for row in rows)
    my_wins = sum(row["my_agent_win"] for row in rows)
    total_games = len(rows)
    team_first = [row for row in rows if row["team_starts"]]
    team_second = [row for row in rows if not row["team_starts"]]
    pair_rates = []
    for seed in base_seeds:
        pair = [row for row in rows if row["base_seed"] == seed]
        pair_rates.append(sum(row["team_ppo_win"] for row in pair) / max(1, len(pair)))
    return {
        "protocol": "project_rf_rule_paired_v1",
        "model_path": args.model_path,
        "training_seed": args.training_seed,
        "opponent": "project_rf_rule",
        "opponent_interpretation": "MyAgent ported in the team member repository",
        "observation_mode": args.observation_mode,
        "deterministic": args.deterministic,
        "seed_source": f"range:{args.seed_start}:{args.num_paired_seeds}",
        "base_seed_count": len(base_seeds),
        "base_seed_sha256": seed_sha256(base_seeds),
        "total_games": total_games,
        "team_ppo_wins": team_wins,
        "my_agent_wins": my_wins,
        "team_ppo_win_rate": team_wins / total_games,
        "my_agent_win_rate": my_wins / total_games,
        "team_ppo_as_first_win_rate": mean(row["team_ppo_win"] for row in team_first),
        "team_ppo_as_second_win_rate": mean(row["team_ppo_win"] for row in team_second),
        "my_agent_as_first_win_rate": mean(row["my_agent_win"] for row in team_second),
        "my_agent_as_second_win_rate": mean(row["my_agent_win"] for row in team_first),
        "first_player_win_rate": mean(row["first_player_win"] for row in rows),
        "second_player_win_rate": mean(row["second_player_win"] for row in rows),
        "avg_turns": mean(row["turns"] for row in rows),
        "avg_decisions": mean(row["decisions"] for row in rows),
        "paired_seed_team_ppo_mean": mean(pair_rates),
        "paired_seed_team_ppo_std": pstdev(pair_rates),
        "illegal_team_ppo": sum(row["illegal_team_ppo"] for row in rows),
        "illegal_my_agent": sum(row["illegal_my_agent"] for row in rows),
        "evaluation_error_count": sum(1 for row in rows if row["evaluation_error"]),
        "elapsed_seconds": elapsed,
    }


def mean(values) -> float:
    items = list(values)
    return 0.0 if not items else sum(items) / len(items)


def pstdev(values) -> float:
    items = list(values)
    if len(items) <= 1:
        return 0.0
    avg = mean(items)
    return (sum((item - avg) ** 2 for item in items) / len(items)) ** 0.5


def seed_sha256(base_seeds: list[int]) -> str:
    payload = json.dumps(list(base_seeds), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def to_csv(rows: list[dict]) -> str:
    if not rows:
        return ""
    fields = list(rows[0].keys())
    lines = [",".join(fields)]
    for row in rows:
        lines.append(",".join(str(row[field]) for field in fields))
    return "\n".join(lines) + "\n"


def report_markdown(summary: dict) -> str:
    return f"""# TeamPPO vs MyAgent Common Paired Evaluation

This is a direct paired evaluation between TeamPPO and MyAgent, where MyAgent is the `project_rf_rule` agent ported in the team member repository.

| Metric | Value |
| --- | ---: |
| Total games | {summary['total_games']} |
| Paired seeds | {summary['base_seed_count']} |
| TeamPPO win rate | {summary['team_ppo_win_rate']:.4f} |
| MyAgent win rate | {summary['my_agent_win_rate']:.4f} |
| TeamPPO as first | {summary['team_ppo_as_first_win_rate']:.4f} |
| TeamPPO as second | {summary['team_ppo_as_second_win_rate']:.4f} |
| MyAgent as first | {summary['my_agent_as_first_win_rate']:.4f} |
| MyAgent as second | {summary['my_agent_as_second_win_rate']:.4f} |
| First-player win rate | {summary['first_player_win_rate']:.4f} |
| Second-player win rate | {summary['second_player_win_rate']:.4f} |
| Avg turns | {summary['avg_turns']:.2f} |
| Avg decisions | {summary['avg_decisions']:.2f} |
| TeamPPO paired seed std | {summary['paired_seed_team_ppo_std']:.4f} |
| Illegal TeamPPO | {summary['illegal_team_ppo']} |
| Illegal MyAgent | {summary['illegal_my_agent']} |
| Evaluation errors | {summary['evaluation_error_count']} |

This result should be interpreted separately from TeamPPO vs `common_rule_based`.
"""


if __name__ == "__main__":
    main()
