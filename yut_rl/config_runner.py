from __future__ import annotations

import json
import random
from pathlib import Path
from statistics import mean, pstdev

from yut_rl.agents import A2CAgent, DQNAgent, DoubleDQNAgent, DuelingDQNAgent, PPOAgent, RandomAgent, ReplayBuffer, RuleBasedAgent, Transition
from yut_rl.env import FINISH, START, YutEnv
from yut_rl.reward_functions import _danger


def load_simple_yaml(path: str | Path) -> dict:
    data: dict[str, object] = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = _parse_scalar(value.strip())
    return data


def _parse_scalar(value: str):
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


def make_env_from_config(config: dict, seed: int | None = None) -> YutEnv:
    return YutEnv(
        seed=seed,
        state_encoder=str(config.get("state_encoder", "raw")),
        reward_function=str(config.get("reward_function", "sparse")),
        action_encoding=str(config.get("action_encoding", "step")),
        enable_action_mask=bool(config.get("action_masking", True)),
    )


class ConfigRuleBasedAgent:
    def act(self, env: YutEnv) -> int:
        player = env.current_player
        opponent = 1 - player
        legal = env.legal_actions()
        scored = []
        for action in legal:
            piece, steps = env.decode_action(action)
            old_pos = env.positions[player][piece]
            from yut_rl.env import advance, distance_to_finish

            new_pos = advance(old_pos, steps)
            captured = new_pos not in (START, FINISH) and new_pos in set(env.positions[opponent])
            score = 100.0 * (new_pos == FINISH)
            score += 50.0 * captured
            score += 4.0 * max(0, env.positions[player].count(old_pos) - 1)
            score -= 0.5 * distance_to_finish(new_pos)
            scored.append((score, -action, action))
        return max(scored)[2]


def make_opponent(config: dict, seed: int):
    kind = str(config.get("opponent_type", "rule_based")).lower()
    if kind in {"random", "random_agent"}:
        return RandomAgent(seed=seed)
    if kind in {"rule", "rule_based"}:
        return ConfigRuleBasedAgent()
    if kind == "legacy_rule_based":
        return RuleBasedAgent()
    return ConfigRuleBasedAgent()


def make_learning_agent(config: dict, state_dim: int, seed: int):
    algorithm = str(config.get("algorithm", "dqn")).lower()
    if algorithm == "dqn":
        return DQNAgent(state_dim=state_dim, seed=seed)
    if algorithm == "double_dqn":
        return DoubleDQNAgent(state_dim=state_dim, seed=seed)
    if algorithm == "dueling_dqn":
        return DuelingDQNAgent(state_dim=state_dim, seed=seed)
    if algorithm == "a2c":
        return A2CAgent(state_dim=state_dim, seed=seed)
    if algorithm in {"ppo", "masked_ppo", "masked-ppo"}:
        return PPOAgent(state_dim=state_dim, seed=seed)
    raise ValueError(f"unsupported algorithm for config runner: {algorithm}")


def train_from_config_dict(config: dict, out_dir: str | Path) -> dict:
    seed = int(config.get("seed", 0))
    random.seed(seed)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    probe = make_env_from_config(config, seed=seed)
    state_dim = len(probe.reset(seed=seed))
    agent = make_learning_agent(config, state_dim, seed)
    opponent = make_opponent(config, seed + 1000)
    algorithm = str(config.get("algorithm", "dqn")).lower()
    episodes = int(config.get("train_episodes", 100))
    replay = ReplayBuffer(capacity=20_000)
    learning_curve = []

    for episode in range(episodes):
        result = _run_training_episode(agent, opponent, config, seed + episode, replay, algorithm)
        if hasattr(agent, "sync_target") and episode % 20 == 0:
            agent.sync_target()
        learning_curve.append({"episode": episode + 1, **result})

    model_path = out_path / "model.pt"
    if hasattr(agent, "save"):
        agent.save(model_path)
    (out_path / "learning_curve.json").write_text(json.dumps(learning_curve, indent=2), encoding="utf-8")
    summary = {
        "model_path": str(model_path),
        "episodes": episodes,
        "final_train_win_rate": mean(row["learner_win"] for row in learning_curve[-max(1, min(50, len(learning_curve))) :]),
        "learning_curve": learning_curve,
    }
    (out_path / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _run_training_episode(agent, opponent, config: dict, seed: int, replay: ReplayBuffer, algorithm: str) -> dict:
    env = make_env_from_config(config, seed=seed)
    env.reset(seed=seed)
    captures = 0
    danger_moves = 0
    total_reward = 0.0
    steps = 0
    policy_episode = []

    while True:
        player = env.current_player
        actor = agent if player == 0 else opponent
        state = env.observe_for(player)
        legal = env.legal_actions()
        before_danger = _danger(env, player)
        action = actor.act(env, epsilon=max(0.02, 0.25 * (0.995 ** steps))) if actor is agent else actor.act(env)
        result = env.step(action)
        after_state = env.observe_for(player)
        reward = result.reward if player == 0 else -result.reward
        if player == 0:
            total_reward += reward
            captures += int(result.info.get("captured", False))
            danger_moves += int(not before_danger and _danger(env, player))
            if algorithm in {"dqn", "double_dqn", "dueling_dqn"}:
                replay.push(Transition(state, action, reward, after_state, result.done, [] if result.done else env.legal_actions()))
                agent.train_batch(replay, batch_size=64)
            else:
                policy_episode.append((state, action, reward, legal))
        steps += 1
        if result.done:
            break

    if policy_episode and algorithm in {"ppo", "masked_ppo", "masked-ppo", "a2c"}:
        _update_policy_agent(agent, policy_episode)

    return {
        "learner_win": int(env.winner() == 0),
        "episode_return": total_reward,
        "turns": steps,
        "captures": captures,
        "danger_moves": danger_moves,
        "finished_pieces": env.positions[0].count(FINISH),
    }


def _update_policy_agent(agent, episode) -> None:
    torch = agent.torch
    states = torch.tensor([item[0] for item in episode], dtype=torch.float32)
    actions = torch.tensor([item[1] for item in episode], dtype=torch.long)
    rewards = [item[2] for item in episode]
    legal_actions = [item[3] for item in episode]
    returns = []
    value = 0.0
    for reward in reversed(rewards):
        value = reward + agent.gamma * value
        returns.append(value)
    returns.reverse()
    returns_t = torch.tensor(returns, dtype=torch.float32)
    log_probs, values, entropies = agent.evaluate_actions(states, actions, legal_actions)
    advantages = returns_t - values.detach()
    loss = -(log_probs * advantages).mean() + 0.5 * torch.nn.functional.smooth_l1_loss(values, returns_t) - 0.01 * entropies.mean()
    agent.optim.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(agent.net.parameters()) + list(agent.policy.parameters()) + list(agent.value.parameters()), agent.grad_clip)
    agent.optim.step()


def evaluate_config_agent(config: dict, model_path: str | Path | None = None, games: int | None = None) -> dict:
    seed = int(config.get("seed", 0))
    eval_games = int(games or config.get("eval_games", 100))
    probe = make_env_from_config(config, seed=seed)
    state_dim = len(probe.reset(seed=seed))
    agent = make_learning_agent(config, state_dim, seed)
    if model_path and Path(model_path).exists():
        agent.load(model_path)
    opponent = make_opponent(config, seed + 5000)
    rows = []
    for idx in range(eval_games):
        rows.append(_run_eval_game(agent, opponent, config, seed + 10_000 + idx))
    return _summarize_eval(rows)


def run_learning_curve_experiment(
    config: dict,
    config_name: str,
    seed: int,
    train_episodes: int | None = None,
    eval_interval: int | None = None,
    eval_games: int | None = None,
) -> list[dict]:
    """Train one config and emit comparable evaluation checkpoints.

    The returned rows are intentionally flat so experiment scripts can save
    them directly as CSV without mixing them with tournament outputs.
    """

    run_config = {**config, "seed": seed}
    random.seed(seed)
    probe = make_env_from_config(run_config, seed=seed)
    state_dim = len(probe.reset(seed=seed))
    agent = make_learning_agent(run_config, state_dim, seed)
    opponent = make_opponent(run_config, seed + 1000)
    algorithm = str(run_config.get("algorithm", "dqn")).lower()
    episodes = int(train_episodes or run_config.get("train_episodes", 200))
    interval = max(1, int(eval_interval or run_config.get("eval_interval", 25)))
    games = int(eval_games or run_config.get("eval_games", 50))
    replay = ReplayBuffer(capacity=int(run_config.get("replay_capacity", 20_000)))
    rows: list[dict] = []
    train_returns: list[float] = []
    eval_win_rates: list[float] = []

    for episode in range(1, episodes + 1):
        result = _run_training_episode(agent, opponent, run_config, seed + episode, replay, algorithm)
        train_returns.append(float(result["episode_return"]))
        if hasattr(agent, "sync_target") and episode % int(run_config.get("target_sync_interval", 20)) == 0:
            agent.sync_target()

        should_eval = episode == 1 or episode % interval == 0 or episode == episodes
        if not should_eval:
            continue

        eval_summary = evaluate_agent_instance(agent, run_config, games=games, seed=seed + 100_000 + episode)
        eval_win_rates.append(float(eval_summary["win_rate"]))
        row = {
            "train_episode": episode,
            "average_episode_return": mean(_tail(train_returns, interval)),
            "moving_average_return": mean(_tail(train_returns, max(5, interval))),
            "evaluation_win_rate": eval_summary["win_rate"],
            "moving_average_win_rate": mean(_tail(eval_win_rates, min(5, len(eval_win_rates)))),
            "avg_turns": eval_summary["avg_turns"],
            "avg_captures": eval_summary["avg_captures"],
            "avg_finished_pieces": eval_summary["avg_finished_pieces"],
            "invalid_action_rate": eval_summary["invalid_action_rate"],
            "danger_move_rate": eval_summary["danger_move_rate"],
            "capture_success_rate": eval_summary["capture_success_rate"],
            "seed": seed,
            "config_name": config_name,
        }
        rows.append(row)

    return rows


def evaluate_agent_instance(agent, config: dict, games: int, seed: int) -> dict:
    opponent = make_opponent(config, seed + 5000)
    rows = []
    for idx in range(games):
        rows.append(_run_eval_game(agent, opponent, config, seed + idx))
    return _summarize_eval(rows)


def _tail(values: list[float], n: int) -> list[float]:
    return values[-max(1, n) :]


def _run_eval_game(agent, opponent, config: dict, seed: int) -> dict:
    env = make_env_from_config(config, seed=seed)
    env.reset(seed=seed)
    total_return = 0.0
    captures = 0
    capture_opportunities = 0
    capture_successes = 0
    danger_moves = 0
    invalid_actions = 0
    steps = 0
    while True:
        player = env.current_player
        actor = agent if player == 0 else opponent
        legal = env.legal_actions()
        before_danger = _danger(env, player)
        if player == 0 and _capture_possible(env, player):
            capture_opportunities += 1
        action = actor.act(env, epsilon=0.0) if actor is agent else actor.act(env)
        if action not in legal:
            invalid_actions += 1
            action = legal[0]
        result = env.step(action)
        if player == 0:
            total_return += result.reward
            captures += int(result.info.get("captured", False))
            capture_successes += int(result.info.get("captured", False))
            danger_moves += int(not before_danger and _danger(env, player))
        steps += 1
        if result.done:
            return {
                "win": int(env.winner() == 0),
                "episode_return": total_return,
                "turns": steps,
                "captures": captures,
                "finished_pieces": env.positions[0].count(FINISH),
                "invalid_actions": invalid_actions,
                "capture_opportunities": capture_opportunities,
                "capture_successes": capture_successes,
                "danger_moves": danger_moves,
            }


def _capture_possible(env: YutEnv, player: int) -> bool:
    from yut_rl.env import advance

    targets = {pos for pos in env.positions[1 - player] if pos not in (START, FINISH)}
    for action in env.legal_actions():
        piece, steps = env.decode_action(action)
        if advance(env.positions[player][piece], steps) in targets:
            return True
    return False


def _summarize_eval(rows: list[dict]) -> dict:
    games = len(rows)
    capture_opportunities = sum(row["capture_opportunities"] for row in rows)
    return {
        "games": games,
        "win_rate": mean(row["win"] for row in rows) if rows else 0.0,
        "average_episode_return": mean(row["episode_return"] for row in rows) if rows else 0.0,
        "avg_turns": mean(row["turns"] for row in rows) if rows else 0.0,
        "avg_captures": mean(row["captures"] for row in rows) if rows else 0.0,
        "avg_finished_pieces": mean(row["finished_pieces"] for row in rows) if rows else 0.0,
        "invalid_action_rate": sum(row["invalid_actions"] for row in rows) / max(1, sum(row["turns"] for row in rows)),
        "capture_success_rate": sum(row["capture_successes"] for row in rows) / max(1, capture_opportunities),
        "danger_move_rate": sum(row["danger_moves"] for row in rows) / max(1, sum(row["turns"] for row in rows)),
        "seed_mean": mean(row["win"] for row in rows) if rows else 0.0,
        "seed_std": pstdev(row["win"] for row in rows) if len(rows) > 1 else 0.0,
        "learning_curve": [],
    }
