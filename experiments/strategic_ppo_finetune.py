from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
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
import torch
import torch.nn.functional as F

from agents.ppo_agent import (
    ACTION_DIM,
    CaptureAwarePPOAgent,
    build_state,
    can_capture,
    can_enter_shortcut,
    can_finish,
    capture_danger,
    masked_logits,
    state_dim,
)
from train.train_ppo import (
    CAPTURE_REWARD,
    make_strategic_value,
    shaped_reward,
    train_mixed,
)
from yut_rl.agents import StrategicRuleBasedAgent
from yut_rl.env import FINISH, START, YutEnv, advance, decode_action, distance_to_finish


def count_captured(before: list[int], after: list[int]) -> int:
    return sum(
        1
        for old, new in zip(before, after)
        if old not in (START, FINISH) and new == START
    )


def load_capture_ppo(path: Path, seed: int, tactical_weight: float = 2.5) -> CaptureAwarePPOAgent:
    agent = CaptureAwarePPOAgent(state_dim=state_dim(seed), seed=seed, tactical_weight=tactical_weight)
    agent.load(path)
    agent.tactical_weight = tactical_weight
    return agent


def action_captures(env: YutEnv, action: int, player: int) -> bool:
    candidate = env.clone()
    before_opp = candidate.positions[1 - player][:]
    candidate.step(action)
    return count_captured(before_opp, candidate.positions[1 - player]) > 0


def action_finishes(env: YutEnv, action: int, player: int) -> bool:
    piece, steps = decode_action(action)
    return advance(env.positions[player][piece], steps) == FINISH


def action_shortcut(env: YutEnv, action: int, player: int) -> bool:
    piece, steps = decode_action(action)
    old_pos = env.positions[player][piece]
    new_pos = advance(old_pos, steps)
    return old_pos not in (START, FINISH) and new_pos in {4, 9, 24}


def has_near_goal_piece(env: YutEnv, player: int, max_distance: int = 5) -> bool:
    return any(
        pos not in (START, FINISH) and distance_to_finish(pos) <= max_distance
        for pos in env.positions[player]
    )


def opponent_near_goal(env: YutEnv, player: int, max_distance: int = 5) -> bool:
    return any(
        pos not in (START, FINISH) and distance_to_finish(pos) <= max_distance
        for pos in env.positions[1 - player]
    )


def action_starts_new_piece(env: YutEnv, action: int, player: int) -> bool:
    piece, _ = decode_action(action)
    return env.positions[player][piece] == START


def endgame_bad_choice(env: YutEnv, action: int, player: int) -> bool:
    late = (
        env.positions[player].count(FINISH) >= 2
        or env.positions[1 - player].count(FINISH) >= 2
        or has_near_goal_piece(env, player)
        or opponent_near_goal(env, player)
    )
    if not late:
        return False
    if can_finish(env, player) and not action_finishes(env, action, player):
        return True
    if opponent_near_goal(env, player) and can_capture(env, player) and not action_captures(env, action, player):
        return True
    if has_near_goal_piece(env, player) and action_starts_new_piece(env, action, player):
        return True
    return False


def mistake_tags(env: YutEnv, action: int, player: int) -> list[str]:
    tags = []
    before_danger = bool(capture_danger(env, player))
    if can_capture(env, player) and not action_captures(env, action, player):
        tags.append("missed_capture")
    if can_finish(env, player) and not action_finishes(env, action, player):
        tags.append("missed_finish")
    if can_enter_shortcut(env, player) and not action_shortcut(env, action, player):
        tags.append("missed_shortcut")
    candidate = env.clone()
    candidate.step(action)
    if not candidate.winner() and capture_danger(candidate, player) and not before_danger:
        tags.append("moved_into_danger")
    if endgame_bad_choice(env, action, player):
        tags.append("endgame_bad_choice")
    return tags


def teacher_distribution(teacher, env: YutEnv, temperature: float = 0.30) -> list[float]:
    legal = env.legal_actions()
    player = env.current_player
    scores = []
    for action in legal:
        scores.append(teacher.score_components(env, action, player)["score"])
    score_t = torch.tensor(scores, dtype=torch.float32)
    if len(scores) > 1:
        score_t = (score_t - score_t.mean()) / (score_t.std() + 1e-6)
    probs = torch.softmax(score_t / temperature, dim=0)
    out = [0.0] * ACTION_DIM
    for action, prob in zip(legal, probs):
        out[action] = float(prob.item())
    return out


def play_logged_game(agent, opponent, ppo_player: int, seed: int, teacher=None) -> dict:
    env = YutEnv(seed=seed)
    env.reset()
    rows = []
    critical_samples = []
    rewards = []
    ppo_captures = 0
    while True:
        player = env.current_player
        before = env.clone()
        if player == ppo_player:
            legal = env.legal_actions()
            action = agent.act(env, deterministic=True)
            tags = mistake_tags(env, action, player)
            if tags and teacher is not None:
                weight = 1 + 3 * len(tags)
                critical_samples.append(
                    {
                        "state": build_state(env),
                        "action": action,
                        "legal_actions": legal,
                        "teacher_probs": teacher_distribution(teacher, env),
                        "weight": weight,
                        "tags": tags,
                    }
                )
            result = env.step(action)
            ppo_captures += count_captured(before.positions[1 - ppo_player], env.positions[1 - ppo_player])
            rewards.append(shaped_reward(before, env, result, ppo_player, CAPTURE_REWARD))
            rows.append(
                {
                    "seed": seed,
                    "ppo_player": ppo_player,
                    "action": action,
                    "tags": ",".join(tags),
                    "missed_capture": int("missed_capture" in tags),
                    "missed_finish": int("missed_finish" in tags),
                    "moved_into_danger": int("moved_into_danger" in tags),
                    "missed_shortcut": int("missed_shortcut" in tags),
                    "endgame_bad_choice": int("endgame_bad_choice" in tags),
                    "captures_after_action": ppo_captures,
                }
            )
        else:
            action = opponent.act(env)
            result = env.step(action)
            if rewards:
                lost = count_captured(before.positions[ppo_player], env.positions[ppo_player])
                rewards[-1] += CAPTURE_REWARD.captured * lost
        if result.done:
            winner = env.winner()
            return {
                "winner": winner,
                "ppo_win": int(winner == ppo_player),
                "turns": len(rows),
                "ppo_captures": ppo_captures,
                "ppo_finished": env.positions[ppo_player].count(FINISH),
                "reward": sum(rewards),
                "rows": rows,
                "critical_samples": critical_samples,
            }


def analyze_losses(agent, games: int, seed: int, value_model: str | None, out_dir: Path):
    all_rows = []
    game_rows = []
    samples = []
    for game in range(games):
        ppo_player = game % 2
        teacher = make_strategic_value(seed + game, value_model)
        result = play_logged_game(agent, teacher, ppo_player, seed + game, teacher=teacher)
        game_rows.append(
            {
                "game": game,
                "ppo_player": ppo_player,
                "ppo_win": result["ppo_win"],
                "turns": result["turns"],
                "ppo_captures": result["ppo_captures"],
                "ppo_finished": result["ppo_finished"],
                "reward": result["reward"],
            }
        )
        if not result["ppo_win"]:
            all_rows.extend(result["rows"])
            samples.extend(result["critical_samples"])
    detail = pd.DataFrame(all_rows)
    games_df = pd.DataFrame(game_rows)
    summary = (
        detail[["missed_capture", "missed_finish", "moved_into_danger", "missed_shortcut", "endgame_bad_choice"]]
        .sum()
        .reset_index()
        if not detail.empty
        else pd.DataFrame(columns=["index", 0])
    )
    summary.columns = ["metric", "count"]
    detail.to_csv(out_dir / "strategic_loss_decision_log.csv", index=False)
    games_df.to_csv(out_dir / "strategic_loss_game_log.csv", index=False)
    summary.to_csv(out_dir / "strategic_loss_summary.csv", index=False)
    return samples, games_df, summary


def distill_critical_samples(agent, samples: list[dict], epochs: int, batch_size: int, out_dir: Path) -> list[dict]:
    if not samples:
        return []
    expanded = []
    for sample in samples:
        expanded.extend([sample] * int(sample["weight"]))
    states = torch.tensor([sample["state"] for sample in expanded], dtype=torch.float32)
    targets = torch.tensor([sample["teacher_probs"] for sample in expanded], dtype=torch.float32)
    legal_actions = [sample["legal_actions"] for sample in expanded]
    indices = list(range(len(expanded)))
    logs = []
    for epoch in range(epochs):
        random.shuffle(indices)
        total = 0.0
        batches = 0
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            logits, _ = agent.forward(states[batch_idx])
            row_losses = []
            for row, idx in enumerate(batch_idx):
                log_probs = F.log_softmax(masked_logits(logits[row], legal_actions[idx]), dim=0)
                row_losses.append(F.kl_div(log_probs, targets[idx], reduction="sum"))
            loss = torch.stack(row_losses).mean()
            agent.optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), agent.grad_clip)
            agent.optim.step()
            total += float(loss.item())
            batches += 1
        logs.append({"epoch": epoch + 1, "loss": total / max(1, batches), "samples": len(expanded)})
    pd.DataFrame(logs).to_csv(out_dir / "critical_distillation_log.csv", index=False)
    return logs


def evaluate_direct(agent, opponent_factory, games: int, seed: int) -> dict:
    wins = 0
    first_wins = 0
    second_wins = 0
    captures = 0
    turns = 0
    finished = 0
    for game in range(games):
        ppo_player = game % 2
        opponent = opponent_factory(seed + game)
        result = play_logged_game(agent, opponent, ppo_player, seed + game)
        wins += result["ppo_win"]
        if ppo_player == 0:
            first_wins += result["ppo_win"]
        else:
            second_wins += result["ppo_win"]
        captures += result["ppo_captures"]
        turns += result["turns"]
        finished += result["ppo_finished"]
    half = max(1, games // 2)
    return {
        "games": games,
        "win_rate": wins / games,
        "first_player_win_rate": first_wins / half,
        "second_player_win_rate": second_wins / half,
        "avg_turns": turns / games,
        "avg_captures": captures / games,
        "avg_finished_pieces": finished / games,
    }


def search_tactical_weight(agent_path: Path, weights: list[float], games: int, seed: int, value_model: str | None, out_dir: Path):
    rows = []
    best = None
    for idx, weight in enumerate(weights):
        agent = load_capture_ppo(agent_path, seed + idx, tactical_weight=weight)
        row = evaluate_direct(
            agent,
            lambda s: make_strategic_value(s, value_model),
            games=games,
            seed=seed + idx * 20_000,
        )
        row["tactical_weight"] = weight
        rows.append(row)
        print(f"tactical_weight={weight}: strategic_value win_rate={row['win_rate']:.3f}")
        if best is None or row["win_rate"] > best[0]["win_rate"]:
            best = (row, agent)
    df = pd.DataFrame(rows).sort_values("win_rate", ascending=False)
    df.to_csv(out_dir / "tactical_weight_search.csv", index=False)
    return best[0], best[1], df


def compare_three(base_agent, tuned_agent, games: int, seed: int, value_model: str | None, out_dir: Path):
    agents = [
        ("strategic_value", make_strategic_value(seed, value_model)),
        ("ppo_capture_imitation", base_agent),
        ("ppo_vs_strategic_finetuned", tuned_agent),
    ]
    totals = {name: {"wins": 0, "games": 0, "captures": 0, "turns": 0, "finished": 0} for name, _ in agents}
    matchups = []
    for i, (left_name, left_agent) in enumerate(agents):
        for j, (right_name, right_agent) in enumerate(agents[i + 1 :], start=i + 1):
            wins = {left_name: 0, right_name: 0}
            caps = {left_name: 0, right_name: 0}
            turns = 0
            for game in range(games):
                game_seed = seed + i * 100_000 + j * 10_000 + game
                if game % 2 == 0:
                    result = play_pair(left_agent, left_name, right_agent, right_name, game_seed)
                else:
                    result = play_pair(right_agent, right_name, left_agent, left_name, game_seed)
                wins[result["winner"]] += 1
                turns += result["turns"]
                for name in (left_name, right_name):
                    totals[name]["wins"] += int(result["winner"] == name)
                    totals[name]["games"] += 1
                    totals[name]["captures"] += result["captures"][name]
                    totals[name]["turns"] += result["turns"]
                    totals[name]["finished"] += result["finished"][name]
                    caps[name] += result["captures"][name]
            matchups.append(
                {
                    "agent_a": left_name,
                    "agent_b": right_name,
                    "games": games,
                    "agent_a_win_rate": wins[left_name] / games,
                    "agent_b_win_rate": wins[right_name] / games,
                    "avg_turns": turns / games,
                    "agent_a_avg_captures": caps[left_name] / games,
                    "agent_b_avg_captures": caps[right_name] / games,
                }
            )
    summary = []
    for name, stats in totals.items():
        game_count = max(1, stats["games"])
        summary.append(
            {
                "agent": name,
                "games": stats["games"],
                "overall_win_rate": stats["wins"] / game_count,
                "avg_turns": stats["turns"] / game_count,
                "avg_captures": stats["captures"] / game_count,
                "avg_finished_pieces": stats["finished"] / game_count,
            }
        )
    summary_df = pd.DataFrame(summary).sort_values("overall_win_rate", ascending=False)
    matchups_df = pd.DataFrame(matchups)
    summary_df.to_csv(out_dir / "strategic_finetune_summary.csv", index=False)
    matchups_df.to_csv(out_dir / "strategic_finetune_matchups.csv", index=False)
    return summary_df, matchups_df


def play_pair(agent0, name0: str, agent1, name1: str, seed: int) -> dict:
    env = YutEnv(seed=seed)
    env.reset()
    agents = [agent0, agent1]
    names = [name0, name1]
    captures = {name0: 0, name1: 0}
    turns = 0
    while True:
        player = env.current_player
        before = [row[:] for row in env.positions]
        action = agents[player].act(env, deterministic=True) if isinstance(agents[player], CaptureAwarePPOAgent) else agents[player].act(env)
        result = env.step(action)
        actor = result.info.get("player", player)
        actor_name = names[actor]
        captures[actor_name] += count_captured(before[1 - actor], env.positions[1 - actor])
        turns += 1
        if result.done:
            winner = env.winner()
            return {
                "winner": names[winner],
                "turns": turns,
                "captures": captures,
                "finished": {name0: env.positions[0].count(FINISH), name1: env.positions[1].count(FINISH)},
            }


def plot_results(summary: pd.DataFrame, matchups: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ordered = summary.sort_values("overall_win_rate", ascending=True)
    ax.barh(ordered["agent"], ordered["overall_win_rate"], color="#4c78a8")
    ax.set_xlim(0, 1)
    ax.set_title("Strategic fine-tune tournament win rate")
    fig.tight_layout()
    fig.savefig(out_dir / "strategic_finetune_win_rate.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ordered = summary.sort_values("avg_captures", ascending=True)
    ax.barh(ordered["agent"], ordered["avg_captures"], color="#f58518")
    ax.set_title("Strategic fine-tune average captures")
    fig.tight_layout()
    fig.savefig(out_dir / "strategic_finetune_captures.png", dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="results/ppo_training")
    parser.add_argument("--out-dir", default="results/ppo_strategic_finetune")
    parser.add_argument("--value-model", default="checkpoints/value.pt")
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--analysis-games", type=int, default=1000)
    parser.add_argument("--eval-games", type=int, default=1000)
    parser.add_argument("--distill-epochs", type=int, default=4)
    parser.add_argument("--finetune-episodes", type=int, default=180)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--eval-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)
    base_path = model_dir / "ppo_capture_imitation.pt"
    base_agent = load_capture_ppo(base_path, args.seed, tactical_weight=2.5)

    if args.eval_only:
        existing_path = model_dir / "ppo_vs_strategic_finetuned.pt"
        tuned_source_path = existing_path if existing_path.exists() else base_path
        base_weight, base_weighted, base_weight_df = search_tactical_weight(
            base_path,
            weights=[2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0],
            games=args.eval_games,
            seed=args.seed + 70_000,
            value_model=args.value_model,
            out_dir=out_dir,
        )
        base_weight_df.to_csv(out_dir / "base_tactical_weight_search.csv", index=False)
        tuned_weight, tuned_weighted, tuned_weight_df = search_tactical_weight(
            tuned_source_path,
            weights=[2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0],
            games=args.eval_games,
            seed=args.seed + 80_000,
            value_model=args.value_model,
            out_dir=out_dir,
        )
        selected_source = "finetuned"
        if base_weight["win_rate"] > tuned_weight["win_rate"]:
            tuned_weight = base_weight
            tuned_weighted = base_weighted
            selected_source = "base_weight_tuned"
        final_path = model_dir / "ppo_vs_strategic_finetuned.pt"
        tuned_weighted.save(final_path)
        base_direct = evaluate_direct(
            base_agent,
            lambda seed: make_strategic_value(seed, args.value_model),
            games=args.eval_games,
            seed=args.seed + 120_000,
        )
        final_for_compare = load_capture_ppo(final_path, args.seed + 3, tactical_weight=tuned_weight["tactical_weight"])
        tuned_direct = evaluate_direct(
            final_for_compare,
            lambda seed: make_strategic_value(seed, args.value_model),
            games=args.eval_games,
            seed=args.seed + 140_000,
        )
        direct_df = pd.DataFrame(
            [
                {"agent": "ppo_capture_imitation", **base_direct},
                {"agent": "ppo_vs_strategic_finetuned", **tuned_direct, "tactical_weight": tuned_weight["tactical_weight"]},
            ]
        )
        direct_df.to_csv(out_dir / "direct_vs_strategic_value.csv", index=False)
        summary, matchups = compare_three(
            load_capture_ppo(base_path, args.seed + 2, tactical_weight=2.5),
            final_for_compare,
            games=args.eval_games,
            seed=args.seed + 160_000,
            value_model=args.value_model,
            out_dir=out_dir,
        )
        plot_results(summary, matchups, out_dir)
        result = {
            "best_tactical_weight": tuned_weight,
            "selected_source": selected_source,
            "base_direct": base_direct,
            "tuned_direct": tuned_direct,
            "strategic_value_direct_win_rate": 1.0 - tuned_direct["win_rate"],
            "strategic_value_vs_ppo_gap": (1.0 - tuned_direct["win_rate"]) - tuned_direct["win_rate"],
            "improvement_vs_base": tuned_direct["win_rate"] - base_direct["win_rate"],
            "capture_improvement": tuned_direct["avg_captures"] - base_direct["avg_captures"],
            "checkpoint": str(final_path),
        }
        (out_dir / "strategic_finetune_best.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print("strategic_finetune_result", result)
        return

    samples, games_df, loss_summary = analyze_losses(
        base_agent,
        games=args.analysis_games,
        seed=args.seed + 10_000,
        value_model=args.value_model,
        out_dir=out_dir,
    )
    print("loss_summary")
    print(loss_summary.to_string(index=False))

    tuned = load_capture_ppo(base_path, args.seed + 1, tactical_weight=2.5)
    distill_critical_samples(
        tuned,
        samples,
        epochs=args.distill_epochs,
        batch_size=args.batch_size,
        out_dir=out_dir,
    )
    train_logs = train_mixed(
        tuned,
        "strategic_value_finetune",
        [
            ("strategic_value", 0.80, lambda seed: make_strategic_value(seed, args.value_model)),
            ("strategic_rule", 0.10, lambda seed: StrategicRuleBasedAgent(seed=seed)),
            ("ppo_self", 0.10, lambda seed: tuned),
        ],
        episodes=args.finetune_episodes,
        seed=args.seed + 50_000,
        n_steps=args.n_steps,
        ppo_epochs=args.ppo_epochs,
        batch_size=args.batch_size,
        gae_lambda=args.gae_lambda,
        reward_profile=CAPTURE_REWARD,
    )
    pd.DataFrame(train_logs).to_csv(out_dir / "strategic_finetune_train_log.csv", index=False)
    raw_tuned_path = out_dir / "ppo_vs_strategic_finetuned_raw.pt"
    tuned.save(raw_tuned_path)

    base_weight, base_weighted, base_weight_df = search_tactical_weight(
        base_path,
        weights=[2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0],
        games=args.eval_games,
        seed=args.seed + 70_000,
        value_model=args.value_model,
        out_dir=out_dir,
    )
    base_weight_df.to_csv(out_dir / "base_tactical_weight_search.csv", index=False)

    best_weight, tuned_weighted, weight_df = search_tactical_weight(
        raw_tuned_path,
        weights=[2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0],
        games=args.eval_games,
        seed=args.seed + 80_000,
        value_model=args.value_model,
        out_dir=out_dir,
    )
    selected_source = "finetuned"
    if base_weight["win_rate"] > best_weight["win_rate"]:
        best_weight = base_weight
        tuned_weighted = base_weighted
        selected_source = "base_weight_tuned"
    final_path = model_dir / "ppo_vs_strategic_finetuned.pt"
    tuned_weighted.save(final_path)

    base_direct = evaluate_direct(
        base_agent,
        lambda seed: make_strategic_value(seed, args.value_model),
        games=args.eval_games,
        seed=args.seed + 120_000,
    )
    tuned_direct = evaluate_direct(
        tuned_weighted,
        lambda seed: make_strategic_value(seed, args.value_model),
        games=args.eval_games,
        seed=args.seed + 140_000,
    )
    direct_df = pd.DataFrame(
        [
            {"agent": "ppo_capture_imitation", **base_direct},
            {"agent": "ppo_vs_strategic_finetuned", **tuned_direct, "tactical_weight": best_weight["tactical_weight"]},
        ]
    )
    direct_df.to_csv(out_dir / "direct_vs_strategic_value.csv", index=False)

    base_for_compare = load_capture_ppo(base_path, args.seed + 2, tactical_weight=2.5)
    final_for_compare = load_capture_ppo(final_path, args.seed + 3, tactical_weight=best_weight["tactical_weight"])
    summary, matchups = compare_three(
        base_for_compare,
        final_for_compare,
        games=args.eval_games,
        seed=args.seed + 160_000,
        value_model=args.value_model,
        out_dir=out_dir,
    )
    plot_results(summary, matchups, out_dir)

    result = {
        "best_tactical_weight": best_weight,
        "selected_source": selected_source,
        "base_direct": base_direct,
        "tuned_direct": tuned_direct,
        "strategic_value_direct_win_rate": 1.0 - tuned_direct["win_rate"],
        "strategic_value_vs_ppo_gap": (1.0 - tuned_direct["win_rate"]) - tuned_direct["win_rate"],
        "improvement_vs_base": tuned_direct["win_rate"] - base_direct["win_rate"],
        "capture_improvement": tuned_direct["avg_captures"] - base_direct["avg_captures"],
        "checkpoint": str(final_path),
    }
    (out_dir / "strategic_finetune_best.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("strategic_finetune_result", result)


if __name__ == "__main__":
    main()
