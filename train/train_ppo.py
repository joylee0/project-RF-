from __future__ import annotations

import argparse
from dataclasses import dataclass
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

import pandas as pd
import torch
import torch.nn.functional as F

from agents.ppo_agent import (
    ACTION_DIM,
    MaskedPPOAgent,
    build_state,
    can_capture,
    can_enter_shortcut,
    can_finish,
    capture_danger,
    masked_logits,
    state_dim,
)
from yut_rl.agents import RandomAgent, StrategicRuleBasedAgent, StrategicValueNetworkAgent
from yut_rl.env import FINISH, START, YutEnv, distance_to_finish


SHORTCUT_ENTRY_POSITIONS = {4, 9, 24}


@dataclass(frozen=True)
class RewardProfile:
    win: float = 100.0
    loss: float = -100.0
    finish: float = 30.0
    capture: float = 15.0
    captured: float = -15.0
    missed_capture: float = 0.0
    shortcut: float = 8.0
    progress_scale: float = 0.5
    danger: float = -5.0
    safe_move: float = 0.0
    turn_penalty: float = -0.1


STANDARD_REWARD = RewardProfile()
CAPTURE_REWARD = RewardProfile(
    capture=35.0,
    captured=-35.0,
    missed_capture=-10.0,
    danger=-15.0,
    safe_move=5.0,
)


def count_captured(before: list[int], after: list[int]) -> int:
    return sum(
        1
        for old, new in zip(before, after)
        if old not in (START, FINISH) and new == START
    )


def choose_action(agent, env: YutEnv) -> int:
    return agent.act(env)


def action_captures(env: YutEnv, action: int, player: int) -> bool:
    candidate = env.clone()
    before_opp = candidate.positions[1 - player][:]
    candidate.step(action)
    return count_captured(before_opp, candidate.positions[1 - player]) > 0


def action_enters_shortcut(env: YutEnv, action: int, player: int) -> bool:
    piece, _ = env_action_piece_steps(action)
    old_pos = env.positions[player][piece]
    candidate = env.clone()
    candidate.step(action)
    new_pos = candidate.positions[player][piece]
    return old_pos not in (START, FINISH) and new_pos in SHORTCUT_ENTRY_POSITIONS


def env_action_piece_steps(action: int) -> tuple[int, int]:
    piece = action % 4
    steps = action // 4 + 1
    return piece, steps


def make_strategic_value(seed: int, value_model: str | None) -> StrategicValueNetworkAgent:
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


def shaped_reward(before: YutEnv, after: YutEnv, result, player: int, profile: RewardProfile = STANDARD_REWARD) -> float:
    before_me = before.positions[player]
    before_opp = before.positions[1 - player]
    after_me = after.positions[player]
    after_opp = after.positions[1 - player]
    old_pos = result.info.get("from")
    new_pos = result.info.get("to")

    reward = profile.turn_penalty
    if result.done:
        reward += profile.win if after.winner() == player else profile.loss

    finished_gain = after_me.count(FINISH) - before_me.count(FINISH)
    reward += profile.finish * max(0, finished_gain)
    captured_count = count_captured(before_opp, after_opp)
    reward += profile.capture * captured_count
    if profile.missed_capture and can_capture(before, player) and captured_count == 0:
        reward += profile.missed_capture

    if old_pos is not None and new_pos is not None:
        progress = distance_to_finish(old_pos) - distance_to_finish(new_pos)
        reward += profile.progress_scale * max(0, progress)
        if old_pos not in (START, FINISH) and new_pos in SHORTCUT_ENTRY_POSITIONS:
            reward += profile.shortcut

    if not result.done:
        before_danger = bool(capture_danger(before, player))
        after_danger = bool(capture_danger(after, player))
        if after_danger:
            reward += profile.danger
        elif before_danger and not after_danger:
            reward += profile.safe_move
    return reward


def play_ppo_episode(
    agent: MaskedPPOAgent,
    opponent,
    seed: int,
    ppo_player: int,
    deterministic: bool = False,
    reward_profile: RewardProfile = STANDARD_REWARD,
) -> dict:
    env = YutEnv(seed=seed)
    env.reset()
    players = [None, None]
    players[ppo_player] = agent
    players[1 - ppo_player] = opponent

    states = []
    actions = []
    legal_actions = []
    old_log_probs = []
    values = []
    rewards = []
    turns = 0
    ppo_captures = 0
    last_ppo_reward_idx = None

    while True:
        player = env.current_player
        before = env.clone()
        if player == ppo_player:
            legal = env.legal_actions()
            state = build_state(env)
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, value = agent.forward(state_tensor)
                logits = masked_logits(logits.squeeze(0), legal)
                if deterministic:
                    action = int(torch.argmax(logits).item())
                else:
                    dist = torch.distributions.Categorical(logits=logits)
                    action = int(dist.sample().item())
                    old_log_prob = float(dist.log_prob(torch.tensor(action)).item())
            if deterministic:
                dist = torch.distributions.Categorical(logits=logits)
                old_log_prob = float(dist.log_prob(torch.tensor(action)).item())

            result = env.step(action)
            ppo_captures += count_captured(before.positions[1 - ppo_player], env.positions[1 - ppo_player])
            states.append(state)
            actions.append(action)
            legal_actions.append(legal)
            old_log_probs.append(old_log_prob)
            values.append(float(value.item()))
            rewards.append(shaped_reward(before, env, result, ppo_player, reward_profile))
            last_ppo_reward_idx = len(rewards) - 1
        else:
            result = env.step(choose_action(opponent, env))
            if last_ppo_reward_idx is not None:
                lost_count = count_captured(before.positions[ppo_player], env.positions[ppo_player])
                if lost_count:
                    rewards[last_ppo_reward_idx] += reward_profile.captured * lost_count
                if result.done and env.winner() != ppo_player:
                    rewards[last_ppo_reward_idx] -= 100.0

        turns += 1
        if result.done:
            break

    return {
        "states": states,
        "actions": actions,
        "legal_actions": legal_actions,
        "old_log_probs": old_log_probs,
        "values": values,
        "rewards": rewards,
        "winner": env.winner(),
        "turns": turns,
        "ppo_player": ppo_player,
        "captures": ppo_captures,
        "finished": env.positions[ppo_player].count(FINISH),
        "total_reward": sum(rewards),
    }


def compute_returns(rewards: list[float], gamma: float) -> list[float]:
    out = []
    value = 0.0
    for reward in reversed(rewards):
        value = reward + gamma * value
        out.append(value)
    out.reverse()
    return out


def compute_gae(rewards: list[float], values: list[float], gamma: float, gae_lambda: float) -> tuple[list[float], list[float]]:
    advantages = []
    gae = 0.0
    next_value = 0.0
    for idx in reversed(range(len(rewards))):
        delta = rewards[idx] + gamma * next_value - values[idx]
        gae = delta + gamma * gae_lambda * gae
        advantages.append(gae)
        next_value = values[idx]
    advantages.reverse()
    returns = [adv + value for adv, value in zip(advantages, values)]
    return returns, advantages


def update_ppo(agent: MaskedPPOAgent, episodes: list[dict], epochs: int, batch_size: int, gae_lambda: float = 0.95) -> float:
    states, actions, old_log_probs, returns, legal_actions, values = [], [], [], [], [], []
    advantages_raw = []
    for episode in episodes:
        if not episode["states"]:
            continue
        episode_returns, episode_advantages = compute_gae(
            episode["rewards"],
            episode["values"],
            agent.gamma,
            gae_lambda,
        )
        states.extend(episode["states"])
        actions.extend(episode["actions"])
        old_log_probs.extend(episode["old_log_probs"])
        returns.extend(episode_returns)
        legal_actions.extend(episode["legal_actions"])
        values.extend(episode["values"])
        advantages_raw.extend(episode_advantages)

    if not states:
        return 0.0

    states_t = torch.tensor(states, dtype=torch.float32)
    actions_t = torch.tensor(actions, dtype=torch.long)
    old_log_probs_t = torch.tensor(old_log_probs, dtype=torch.float32)
    returns_t = torch.tensor(returns, dtype=torch.float32)
    advantages = torch.tensor(advantages_raw, dtype=torch.float32)
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    indices = list(range(len(states)))
    last_loss = 0.0
    for _ in range(epochs):
        random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            batch_states = states_t[batch_idx]
            batch_actions = actions_t[batch_idx]
            batch_old_log_probs = old_log_probs_t[batch_idx]
            batch_returns = returns_t[batch_idx]
            batch_advantages = advantages[batch_idx]
            batch_legal = [legal_actions[i] for i in batch_idx]

            log_probs, value_pred, entropy = agent.evaluate_actions(batch_states, batch_actions, batch_legal)
            ratio = torch.exp(log_probs - batch_old_log_probs)
            clipped = torch.clamp(ratio, 1.0 - agent.clip, 1.0 + agent.clip) * batch_advantages
            policy_loss = -torch.min(ratio * batch_advantages, clipped).mean()
            value_loss = F.smooth_l1_loss(value_pred, batch_returns)
            entropy_bonus = entropy.mean()
            loss = policy_loss + agent.value_coef * value_loss - agent.entropy_coef * entropy_bonus

            agent.optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), agent.grad_clip)
            agent.optim.step()
            last_loss = float(loss.item())
    return last_loss


def train_against(
    agent: MaskedPPOAgent,
    opponent_factory,
    episodes: int,
    seed: int,
    update_every: int,
    ppo_epochs: int,
    batch_size: int,
    gae_lambda: float = 0.95,
    reward_profile: RewardProfile = STANDARD_REWARD,
) -> list[dict]:
    logs = []
    rollout = []
    wins = 0
    for episode in range(episodes):
        opponent = opponent_factory(seed + episode)
        ppo_player = episode % 2
        result = play_ppo_episode(agent, opponent, seed + episode, ppo_player, reward_profile=reward_profile)
        wins += int(result["winner"] == ppo_player)
        rollout.append(result)
        if len(rollout) >= update_every:
            loss = update_ppo(agent, rollout, ppo_epochs, batch_size, gae_lambda=gae_lambda)
            logs.append(
                {
                    "episode": episode + 1,
                    "loss": loss,
                    "recent_win_rate": wins / (episode + 1),
                    "avg_reward": sum(item["total_reward"] for item in rollout) / len(rollout),
                }
            )
            rollout = []
    if rollout:
        loss = update_ppo(agent, rollout, ppo_epochs, batch_size, gae_lambda=gae_lambda)
        logs.append({"episode": episodes, "loss": loss, "recent_win_rate": wins / max(1, episodes), "avg_reward": sum(item["total_reward"] for item in rollout) / len(rollout)})
    return logs


def train_mixed(
    agent: MaskedPPOAgent,
    stage_name: str,
    opponent_mix: list[tuple[str, float, object]],
    episodes: int,
    seed: int,
    n_steps: int,
    ppo_epochs: int,
    batch_size: int,
    gae_lambda: float,
    reward_profile: RewardProfile,
) -> list[dict]:
    logs = []
    rollout = []
    wins = 0
    steps_collected = 0
    rng = random.Random(seed)
    labels = [item[0] for item in opponent_mix]
    weights = [item[1] for item in opponent_mix]
    factories = [item[2] for item in opponent_mix]
    for episode in range(episodes):
        idx = rng.choices(range(len(factories)), weights=weights, k=1)[0]
        opponent = factories[idx](seed + episode)
        ppo_player = episode % 2
        result = play_ppo_episode(
            agent,
            opponent,
            seed + episode,
            ppo_player,
            reward_profile=reward_profile,
        )
        wins += int(result["winner"] == ppo_player)
        rollout.append(result)
        steps_collected += len(result["states"])
        if steps_collected >= n_steps:
            loss = update_ppo(agent, rollout, ppo_epochs, batch_size, gae_lambda=gae_lambda)
            logs.append(
                {
                    "stage": stage_name,
                    "episode": episode + 1,
                    "loss": loss,
                    "recent_win_rate": wins / (episode + 1),
                    "avg_reward": sum(item["total_reward"] for item in rollout) / len(rollout),
                    "steps": steps_collected,
                    "opponents": ",".join(labels),
                }
            )
            rollout = []
            steps_collected = 0
    if rollout:
        loss = update_ppo(agent, rollout, ppo_epochs, batch_size, gae_lambda=gae_lambda)
        logs.append(
            {
                "stage": stage_name,
                "episode": episodes,
                "loss": loss,
                "recent_win_rate": wins / max(1, episodes),
                "avg_reward": sum(item["total_reward"] for item in rollout) / len(rollout),
                "steps": steps_collected,
                "opponents": ",".join(labels),
            }
        )
    return logs


def evaluate_vs(agent: MaskedPPOAgent, opponent_factory, games: int, seed: int, reward_profile: RewardProfile = STANDARD_REWARD) -> dict:
    wins = 0
    turns = 0
    captures = 0
    finished = 0
    rewards = 0.0
    for game in range(games):
        ppo_player = game % 2
        opponent = opponent_factory(seed + game)
        result = play_ppo_episode(agent, opponent, seed + game, ppo_player, deterministic=True, reward_profile=reward_profile)
        wins += int(result["winner"] == ppo_player)
        turns += result["turns"]
        captures += result["captures"]
        finished += result["finished"]
        rewards += result["total_reward"]
    return {
        "games": games,
        "win_rate": wins / games,
        "avg_turns": turns / games,
        "avg_captures": captures / games,
        "avg_finished": finished / games,
        "avg_reward": rewards / games,
    }


def collect_teacher_samples(
    teacher,
    samples: int,
    seed: int,
) -> tuple[list[list[float]], list[int], list[list[int]]]:
    states = []
    actions = []
    legal_actions = []
    game = 0
    while len(states) < samples:
        env = YutEnv(seed=seed + game)
        env.reset()
        while True:
            if len(states) < samples:
                states.append(build_state(env))
                legal_actions.append(env.legal_actions())
                actions.append(teacher.act(env))
                result = env.step(actions[-1])
            else:
                result = env.step(teacher.act(env))
            if result.done:
                break
        game += 1
    return states, actions, legal_actions


def oversample_weight(env: YutEnv, player: int) -> tuple[int, list[str]]:
    weight = 1
    tags = []
    if can_capture(env, player):
        weight = max(weight, 5)
        tags.append("capture_possible")
    if capture_danger(env, player):
        weight = max(weight, 5)
        tags.append("danger_state")
    if can_finish(env, player):
        weight = max(weight, 3)
        tags.append("finish_possible")
    if can_enter_shortcut(env, player):
        weight = max(weight, 2)
        tags.append("shortcut_possible")
    return weight, tags


def teacher_action_distribution(teacher, env: YutEnv, temperature: float = 0.35) -> list[float]:
    legal = env.legal_actions()
    player = env.current_player
    scores = []
    for action in legal:
        if hasattr(teacher, "score_components"):
            score = teacher.score_components(env, action, player)["score"]
        elif hasattr(teacher, "score_action"):
            score = teacher.score_action(env, action)
        else:
            score = 1.0 if action == teacher.act(env) else 0.0
        scores.append(score)
    score_t = torch.tensor(scores, dtype=torch.float32)
    score_t = (score_t - score_t.mean()) / (score_t.std() + 1e-6) if len(scores) > 1 else score_t
    probs = torch.softmax(score_t / temperature, dim=0)
    out = [0.0] * ACTION_DIM
    for action, prob in zip(legal, probs):
        out[action] = float(prob.item())
    return out


def collect_tactical_teacher_samples(
    teacher,
    samples: int,
    seed: int,
    temperature: float = 0.35,
) -> tuple[list[list[float]], list[int], list[list[int]], list[list[float]], pd.DataFrame]:
    states = []
    actions = []
    legal_actions = []
    target_probs = []
    log_rows = []
    game = 0
    while len(states) < samples:
        env = YutEnv(seed=seed + game)
        env.reset()
        while True:
            player = env.current_player
            legal = env.legal_actions()
            action = teacher.act(env)
            probs = teacher_action_distribution(teacher, env, temperature=temperature)
            weight, tags = oversample_weight(env, player)
            for _ in range(weight):
                if len(states) >= samples:
                    break
                states.append(build_state(env))
                actions.append(action)
                legal_actions.append(legal)
                target_probs.append(probs)
                log_rows.append(
                    {
                        "game": game,
                        "player": player,
                        "action": action,
                        "weight": weight,
                        "tags": ",".join(tags) if tags else "normal",
                        "capture_possible": int("capture_possible" in tags),
                        "danger_state": int("danger_state" in tags),
                        "finish_possible": int("finish_possible" in tags),
                        "shortcut_possible": int("shortcut_possible" in tags),
                    }
                )
            result = env.step(action)
            if result.done or len(states) >= samples:
                break
        game += 1
    return states, actions, legal_actions, target_probs, pd.DataFrame(log_rows)


def imitation_pretrain(
    agent: MaskedPPOAgent,
    teacher,
    samples: int,
    epochs: int,
    batch_size: int,
    seed: int,
    tactical: bool = False,
    temperature: float = 0.35,
) -> list[dict]:
    target_probs = None
    sample_log = None
    if tactical:
        states, actions, legal_actions, target_probs, sample_log = collect_tactical_teacher_samples(
            teacher,
            samples,
            seed,
            temperature=temperature,
        )
    else:
        states, actions, legal_actions = collect_teacher_samples(teacher, samples, seed)
    states_t = torch.tensor(states, dtype=torch.float32)
    actions_t = torch.tensor(actions, dtype=torch.long)
    targets_t = torch.tensor(target_probs, dtype=torch.float32) if target_probs is not None else None
    indices = list(range(len(states)))
    logs = []
    for epoch in range(epochs):
        random.shuffle(indices)
        total_loss = 0.0
        batches = 0
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            logits, _ = agent.forward(states_t[batch_idx])
            row_losses = []
            for row, idx in enumerate(batch_idx):
                masked = masked_logits(logits[row], legal_actions[idx])
                if targets_t is None:
                    row_losses.append(F.cross_entropy(masked.unsqueeze(0), actions_t[idx].unsqueeze(0)))
                else:
                    log_probs = F.log_softmax(masked, dim=0)
                    row_losses.append(F.kl_div(log_probs, targets_t[idx], reduction="sum"))
            loss = torch.stack(row_losses).mean()
            agent.optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), agent.grad_clip)
            agent.optim.step()
            total_loss += float(loss.item())
            batches += 1
        logs.append({"epoch": epoch + 1, "loss": total_loss / max(1, batches), "samples": samples, "tactical": tactical})
    if sample_log is not None:
        logs.append({"epoch": 0, "loss": 0.0, "samples": samples, "tactical": True, "dataset_rows": len(sample_log)})
        agent._last_sample_log = sample_log
    return logs


def write_state_analysis(out_dir: Path) -> None:
    dim = state_dim()
    text = [
        "# PPO state representation",
        "",
        f"- 기존 `YutEnv.observe()` 차원: {len(YutEnv(seed=0).reset())}",
        f"- PPO 개선 state 차원: {dim}",
        "- 포함 feature: 내 말 4개 one-hot, 상대 말 4개 one-hot, 현재 pending 윷 결과, 내/상대 완주 말 수, 잡기/완주/지름길/위험 플래그, goal까지 남은 거리.",
        "- 기존 state는 위치와 윷 결과 중심이라 즉시 잡기, 잡힘 위험, 완주 가능성 같은 전략 feature를 PPO가 직접 추론해야 했다.",
        "- 개선 state는 policy/value network가 전술적 선택을 더 빨리 학습하도록 게임 규칙 기반 feature를 함께 제공한다.",
    ]
    (out_dir / "ppo_state_analysis.md").write_text("\n".join(text), encoding="utf-8")


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_variant(
    name: str,
    agent: MaskedPPOAgent,
    stages: list[tuple[str, object]],
    args,
    out_dir: Path,
    seed_offset: int,
) -> list[dict]:
    rows = []
    train_logs = []
    for idx, (stage_name, opponent_factory) in enumerate(stages, start=1):
        logs = train_against(
            agent,
            opponent_factory,
            episodes=args.stage_episodes,
            seed=args.seed + seed_offset + idx * 10_000,
            update_every=args.update_every,
            ppo_epochs=args.ppo_epochs,
            batch_size=args.batch_size,
            gae_lambda=args.gae_lambda,
        )
        for log in logs:
            train_logs.append({"variant": name, "stage": stage_name, **log})
        eval_row = evaluate_vs(
            agent,
            opponent_factory,
            games=args.eval_games,
            seed=args.seed + seed_offset + idx * 20_000,
        )
        rows.append({"variant": name, "stage": stage_name, **eval_row})
        print(f"{name} / {stage_name}: win_rate={eval_row['win_rate']:.3f}")
    pd.DataFrame(train_logs).to_csv(out_dir / f"{name}_train_log.csv", index=False)
    agent.save(out_dir / f"{name}.pt")
    return rows


def capture_hyperparameter_candidates(limit: int) -> list[dict]:
    grid = [
        {"lr": 1e-4, "gamma": 0.99, "gae_lambda": 0.95, "clip": 0.15, "entropy_coef": 0.03, "n_steps": 1024, "batch_size": 128},
        {"lr": 5e-5, "gamma": 0.995, "gae_lambda": 0.98, "clip": 0.10, "entropy_coef": 0.02, "n_steps": 1024, "batch_size": 128},
        {"lr": 1e-4, "gamma": 0.995, "gae_lambda": 0.95, "clip": 0.20, "entropy_coef": 0.05, "n_steps": 2048, "batch_size": 256},
        {"lr": 5e-5, "gamma": 0.99, "gae_lambda": 0.98, "clip": 0.15, "entropy_coef": 0.03, "n_steps": 2048, "batch_size": 256},
        {"lr": 1e-4, "gamma": 0.99, "gae_lambda": 0.98, "clip": 0.10, "entropy_coef": 0.05, "n_steps": 1024, "batch_size": 256},
        {"lr": 5e-5, "gamma": 0.995, "gae_lambda": 0.95, "clip": 0.20, "entropy_coef": 0.02, "n_steps": 2048, "batch_size": 128},
    ]
    return grid[: max(1, min(limit, len(grid)))]


def run_capture_agents(args, out_dir: Path, dim: int) -> None:
    teacher = make_strategic_value(args.seed + 40, args.value_model)
    search_rows = []
    stage_rows = []
    all_train_logs = []
    best = None

    for idx, hp in enumerate(capture_hyperparameter_candidates(args.capture_search_limit)):
        print(f"capture candidate {idx}: {hp}")
        agent = MaskedPPOAgent(
            dim,
            hidden_dim=args.hidden_dim,
            lr=hp["lr"],
            gamma=hp["gamma"],
            clip=hp["clip"],
            entropy_coef=hp["entropy_coef"],
            value_coef=args.value_coef,
            seed=args.seed + 900 + idx,
        )
        logs = imitation_pretrain(
            agent,
            teacher,
            samples=args.capture_samples,
            epochs=args.capture_imitation_epochs,
            batch_size=hp["batch_size"],
            seed=args.seed + 700_000 + idx * 10_000,
            tactical=True,
        )
        pd.DataFrame(logs).to_csv(out_dir / f"ppo_capture_candidate_{idx}_distill_log.csv", index=False)
        sample_log = getattr(agent, "_last_sample_log", None)
        if sample_log is not None:
            sample_log.to_csv(out_dir / f"ppo_capture_candidate_{idx}_samples.csv", index=False)

        capture_eval = evaluate_vs(
            agent,
            lambda seed: make_strategic_value(seed, args.value_model),
            games=args.eval_games,
            seed=args.seed + 710_000 + idx * 10_000,
            reward_profile=CAPTURE_REWARD,
        )
        search_row = {"candidate": idx, "phase": "capture_imitation", **hp, **capture_eval}
        search_rows.append(search_row)
        print(f"candidate {idx} imitation vs strategic_value: {capture_eval['win_rate']:.3f}")
        if best is None or capture_eval["win_rate"] > best[0]["win_rate"]:
            best = (search_row, agent, hp)

    pd.DataFrame(search_rows).to_csv(out_dir / "ppo_capture_hparam_search.csv", index=False)
    best_row, capture_agent, best_hp = best
    capture_agent.save(out_dir / "ppo_capture_imitation.pt")

    tactical = MaskedPPOAgent(
        dim,
        hidden_dim=args.hidden_dim,
        lr=best_hp["lr"],
        gamma=best_hp["gamma"],
        clip=best_hp["clip"],
        entropy_coef=best_hp["entropy_coef"],
        value_coef=args.value_coef,
        seed=args.seed + 990,
    )
    tactical.body.load_state_dict(capture_agent.body.state_dict())
    tactical.policy.load_state_dict(capture_agent.policy.state_dict())
    tactical.value.load_state_dict(capture_agent.value.state_dict())

    stage_mixes = [
        (
            "mixed_stage1",
            [
                ("random", 0.50, lambda seed: RandomAgent(seed=seed)),
                ("strategic_rule", 0.50, lambda seed: StrategicRuleBasedAgent(seed=seed)),
            ],
        ),
        (
            "mixed_stage2",
            [
                ("random", 0.20, lambda seed: RandomAgent(seed=seed)),
                ("strategic_rule", 0.40, lambda seed: StrategicRuleBasedAgent(seed=seed)),
                ("strategic_value", 0.40, lambda seed: make_strategic_value(seed, args.value_model)),
            ],
        ),
        (
            "mixed_stage3",
            [
                ("strategic_rule", 0.20, lambda seed: StrategicRuleBasedAgent(seed=seed)),
                ("strategic_value", 0.70, lambda seed: make_strategic_value(seed, args.value_model)),
                ("ppo_self", 0.10, lambda seed: tactical),
            ],
        ),
    ]
    for stage_idx, (stage_name, mix) in enumerate(stage_mixes):
        logs = train_mixed(
            tactical,
            stage_name,
            mix,
            episodes=args.capture_stage_episodes,
            seed=args.seed + 800_000 + stage_idx * 20_000,
            n_steps=best_hp["n_steps"],
            ppo_epochs=args.ppo_epochs,
            batch_size=best_hp["batch_size"],
            gae_lambda=best_hp["gae_lambda"],
            reward_profile=CAPTURE_REWARD,
        )
        all_train_logs.extend(logs)
        eval_row = evaluate_vs(
            tactical,
            lambda seed: make_strategic_value(seed, args.value_model),
            games=args.eval_games,
            seed=args.seed + 850_000 + stage_idx * 20_000,
            reward_profile=CAPTURE_REWARD,
        )
        stage_rows.append({"variant": "ppo_tactical", "stage": stage_name, **best_hp, **eval_row})
        print(f"ppo_tactical / {stage_name} vs strategic_value: {eval_row['win_rate']:.3f}")

    pd.DataFrame(all_train_logs).to_csv(out_dir / "ppo_tactical_train_log.csv", index=False)
    pd.DataFrame(stage_rows).to_csv(out_dir / "ppo_tactical_stage_eval.csv", index=False)
    tactical.save(out_dir / "ppo_tactical.pt")
    save_json(
        out_dir / "ppo_capture_best_params.json",
        {
            "best_capture_imitation": best_row,
            "best_hyperparameter": best_hp,
            "reward_profile": CAPTURE_REWARD.__dict__,
            "search_space": {
                "learning_rate": [1e-4, 5e-5],
                "gamma": [0.99, 0.995],
                "gae_lambda": [0.95, 0.98],
                "clip_range": [0.1, 0.15, 0.2],
                "entropy_coef": [0.02, 0.03, 0.05],
                "n_steps": [1024, 2048],
                "batch_size": [128, 256],
            },
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/ppo_training")
    parser.add_argument("--value-model", default="checkpoints/value.pt")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--stage-episodes", type=int, default=160)
    parser.add_argument("--eval-games", type=int, default=1000)
    parser.add_argument("--update-every", type=int, default=8)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--imitation-samples", type=int, default=1200)
    parser.add_argument("--imitation-epochs", type=int, default=4)
    parser.add_argument("--refresh-imitation-only", action="store_true")
    parser.add_argument("--train-capture-agents", action="store_true")
    parser.add_argument("--capture-samples", type=int, default=8000)
    parser.add_argument("--capture-imitation-epochs", type=int, default=8)
    parser.add_argument("--capture-stage-episodes", type=int, default=120)
    parser.add_argument("--capture-search-limit", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    write_state_analysis(out_dir)

    dim = state_dim(args.seed)
    params = {
        "state_dim": dim,
        "hidden_dim": args.hidden_dim,
        "lr": args.lr,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip": args.clip,
        "entropy_coef": args.entropy_coef,
        "value_coef": args.value_coef,
        "stage_episodes": args.stage_episodes,
        "eval_games": args.eval_games,
        "reward_shaping": {
            "win": 100,
            "loss": -100,
            "finish": 30,
            "capture": 15,
            "captured": -15,
            "shortcut": 8,
            "progress": "distance * 0.5",
            "danger": -5,
            "turn_penalty": -0.1,
        },
        "capture_reward_shaping": CAPTURE_REWARD.__dict__,
    }
    save_json(out_dir / "ppo_params.json", params)

    random_stage = [("stage1_random", lambda seed: RandomAgent(seed=seed))]
    masked_stage = [("masked_vs_strategic_rule", lambda seed: StrategicRuleBasedAgent(seed=seed))]
    curriculum_stages = [
        ("stage1_random", lambda seed: RandomAgent(seed=seed)),
        ("stage2_strategic_rule", lambda seed: StrategicRuleBasedAgent(seed=seed)),
        ("stage3_strategic_value", lambda seed: make_strategic_value(seed, args.value_model)),
    ]

    if args.train_capture_agents:
        run_capture_agents(args, out_dir, dim)
        return

    if args.refresh_imitation_only:
        imitation = MaskedPPOAgent(dim, hidden_dim=args.hidden_dim, lr=args.lr, gamma=args.gamma, clip=args.clip, value_coef=args.value_coef, entropy_coef=args.entropy_coef, seed=args.seed + 3)
        teacher = make_strategic_value(args.seed + 4, args.value_model)
        imitation_logs = imitation_pretrain(
            imitation,
            teacher,
            samples=args.imitation_samples,
            epochs=args.imitation_epochs,
            batch_size=args.batch_size,
            seed=args.seed + 400_000,
        )
        pd.DataFrame(imitation_logs).to_csv(out_dir / "ppo_imitation_pretrain_log.csv", index=False)
        imitation.save(out_dir / "ppo_imitation_pretrained.pt")
        imitation.save(out_dir / "ppo_imitation.pt")
        eval_rows = []
        for stage_name, opponent_factory in curriculum_stages:
            row = evaluate_vs(imitation, opponent_factory, games=args.eval_games, seed=args.seed + 500_000)
            eval_rows.append({"variant": "ppo_imitation", "stage": f"pretrained_{stage_name}", **row})
            print(f"ppo_imitation / pretrained_{stage_name}: win_rate={row['win_rate']:.3f}")
        pd.DataFrame(eval_rows).to_csv(out_dir / "stage_eval_imitation_refresh.csv", index=False)
        return

    all_eval_rows = []

    baseline = MaskedPPOAgent(dim, hidden_dim=args.hidden_dim, lr=args.lr, gamma=args.gamma, clip=args.clip, value_coef=args.value_coef, entropy_coef=args.entropy_coef, seed=args.seed)
    all_eval_rows.extend(run_variant("ppo_baseline", baseline, random_stage, args, out_dir, 100_000))

    masked = MaskedPPOAgent(dim, hidden_dim=args.hidden_dim, lr=args.lr, gamma=args.gamma, clip=args.clip, value_coef=args.value_coef, entropy_coef=args.entropy_coef, seed=args.seed + 1)
    all_eval_rows.extend(run_variant("ppo_masked", masked, masked_stage, args, out_dir, 200_000))

    curriculum = MaskedPPOAgent(dim, hidden_dim=args.hidden_dim, lr=args.lr, gamma=args.gamma, clip=args.clip, value_coef=args.value_coef, entropy_coef=args.entropy_coef, seed=args.seed + 2)
    all_eval_rows.extend(run_variant("ppo_curriculum", curriculum, curriculum_stages, args, out_dir, 300_000))

    imitation = MaskedPPOAgent(dim, hidden_dim=args.hidden_dim, lr=args.lr, gamma=args.gamma, clip=args.clip, value_coef=args.value_coef, entropy_coef=args.entropy_coef, seed=args.seed + 3)
    teacher = make_strategic_value(args.seed + 4, args.value_model)
    imitation_logs = imitation_pretrain(
        imitation,
        teacher,
        samples=args.imitation_samples,
        epochs=args.imitation_epochs,
        batch_size=args.batch_size,
        seed=args.seed + 400_000,
    )
    pd.DataFrame(imitation_logs).to_csv(out_dir / "ppo_imitation_pretrain_log.csv", index=False)
    imitation.save(out_dir / "ppo_imitation_pretrained.pt")
    all_eval_rows.extend(run_variant("ppo_imitation", imitation, curriculum_stages, args, out_dir, 400_000))

    stage_eval = pd.DataFrame(all_eval_rows)
    stage_eval.to_csv(out_dir / "stage_eval.csv", index=False)
    best = stage_eval.sort_values("win_rate", ascending=False).iloc[0].to_dict()
    save_json(out_dir / "best_stage.json", best)
    print("best_stage", best)


if __name__ == "__main__":
    main()
