from __future__ import annotations

import argparse
from pathlib import Path
import random
from statistics import mean

from .agents import (
    A2CAgent,
    DQNAgent,
    DoubleDQNAgent,
    DuelingDQNAgent,
    MCTSValueAgent,
    PPOAgent,
    RandomAgent,
    ReplayBuffer,
    ReinforceAgent,
    RuleBasedAgent,
    StrategicRuleBasedAgent,
    StrategicValueNetworkAgent,
    TabularQAgent,
    Transition,
    ValueNetworkAgent,
    ValueTransition,
)
from .env import FINISH, YutEnv


def shape_reward(args, raw_reward: float, done: bool, winner: int | None, acting_player: int) -> float:
    if done and winner is not None:
        return 1.0 if winner == acting_player else -1.0
    if args.reward_mode == "terminal":
        return 0.0
    if args.reward_mode == "hybrid":
        return raw_reward * args.dense_reward_scale
    return raw_reward


def play_game(
    agent0,
    agent1,
    seed: int | None = None,
    train_agent: DQNAgent | ValueNetworkAgent | None = None,
    replay: ReplayBuffer | None = None,
    epsilon: float = 0.0,
    batch_size: int = 64,
):
    env = YutEnv(seed=seed)
    env.reset()
    turns = 0
    pending_dqn: dict[int, dict] = {}

    def push_dqn_transition(player: int, done: bool) -> None:
        pending = pending_dqn.pop(player, None)
        if pending is None or replay is None:
            return
        replay.push(
            Transition(
                state=pending["state"],
                action=pending["action"],
                reward=pending["reward"],
                next_state=env.observe_for(player),
                done=done,
                next_legal=[] if done else env.legal_actions(),
                discount_power=pending["discount_power"],
            )
        )

    while True:
        if env.winner() is not None:
            return env.winner(), turns
        agent = agent0 if env.current_player == 0 else agent1
        if (
            replay is not None
            and isinstance(train_agent, DQNAgent)
            and agent is train_agent
            and env.current_player in pending_dqn
        ):
            push_dqn_transition(env.current_player, done=False)
            train_agent.train_batch(replay, batch_size=batch_size)
        if isinstance(agent, (DQNAgent, ValueNetworkAgent, TabularQAgent, MCTSValueAgent, ReinforceAgent, A2CAgent, PPOAgent)):
            action = agent.act(env, epsilon=epsilon)
        else:
            action = agent.act(env)

        acting_player = env.current_player
        state = env.observe_for(acting_player)
        result = env.step(action)
        turns += 1

        if replay is not None and agent is train_agent:
            winner = env.winner() if result.done else None
            reward = shape_reward(args=play_game.args, raw_reward=result.reward, done=result.done, winner=winner, acting_player=acting_player)
            if isinstance(train_agent, DQNAgent):
                if result.done:
                    replay.push(
                        Transition(
                            state=state,
                            action=action,
                            reward=reward,
                            next_state=env.observe_for(acting_player),
                            done=True,
                            next_legal=[],
                        )
                    )
                elif env.current_player == acting_player:
                    replay.push(
                        Transition(
                            state=state,
                            action=action,
                            reward=reward,
                            next_state=env.observe_for(acting_player),
                            done=False,
                            next_legal=env.legal_actions(),
                        )
                    )
                else:
                    pending_dqn[acting_player] = {
                        "state": state,
                        "action": action,
                        "reward": reward,
                        "discount_power": 1,
                    }
            else:
                replay.push(
                    ValueTransition(
                        state=state,
                        reward=reward,
                        next_state=env.observe_for(acting_player),
                        done=result.done,
                    )
                )
        elif replay is not None and isinstance(train_agent, DQNAgent) and pending_dqn:
            winner = env.winner() if result.done else None
            opponent_reward = shape_reward(
                args=play_game.args,
                raw_reward=result.reward,
                done=result.done,
                winner=winner,
                acting_player=acting_player,
            )
            for player, pending in pending_dqn.items():
                pending["reward"] += (play_game.args.gamma ** pending["discount_power"]) * (-opponent_reward)
                pending["discount_power"] += 1
        if train_agent is not None and replay is not None:
            train_agent.train_batch(replay, batch_size=batch_size)

        if result.done:
            if replay is not None and isinstance(train_agent, DQNAgent):
                for player in list(pending_dqn):
                    push_dqn_transition(player, done=True)
            winner = env.winner()
            if (
                replay is not None
                and isinstance(train_agent, ValueNetworkAgent)
                and winner is not None
            ):
                loser = 1 - winner
                loser_is_trained = (
                    (loser == 0 and agent0 is train_agent)
                    or (loser == 1 and agent1 is train_agent)
                )
                if loser_is_trained:
                    replay.push(
                        ValueTransition(
                            state=env.observe_for(loser),
                            reward=-1.0,
                            next_state=env.observe_for(loser),
                            done=True,
                        )
                    )
            return env.winner(), turns


play_game.args = argparse.Namespace(reward_mode="dense", dense_reward_scale=0.5)


def evaluate(agent0, agent1, games: int = 200, seed_offset: int = 10_000):
    wins = [0, 0]
    lengths = []
    captures = [0, 0]
    finished = [0, 0]
    for i in range(games):
        winner, turns, metrics = play_game_with_metrics(agent0, agent1, seed=seed_offset + i)
        if winner is not None:
            wins[winner] += 1
        lengths.append(turns)
        captures[0] += metrics["captures"][0]
        captures[1] += metrics["captures"][1]
        finished[0] += metrics["finished"][0]
        finished[1] += metrics["finished"][1]
    return {
        "p0_win_rate": wins[0] / games,
        "p1_win_rate": wins[1] / games,
        "avg_turns": mean(lengths),
        "p0_avg_captures": captures[0] / games,
        "p1_avg_captures": captures[1] / games,
        "p0_avg_finished": finished[0] / games,
        "p1_avg_finished": finished[1] / games,
    }


def play_game_with_metrics(agent0, agent1, seed: int | None = None):
    env = YutEnv(seed=seed)
    env.reset()
    turns = 0
    captures = [0, 0]
    while True:
        if env.winner() is not None:
            return env.winner(), turns, {
                "captures": captures,
                "finished": [env.positions[0].count(FINISH), env.positions[1].count(FINISH)],
            }
        agent = agent0 if env.current_player == 0 else agent1
        action = agent.act(env, epsilon=0.0) if isinstance(agent, (DQNAgent, ValueNetworkAgent, MCTSValueAgent, ReinforceAgent, A2CAgent, PPOAgent)) else agent.act(env)
        result = env.step(action)
        turns += 1
        player = result.info.get("player")
        if player is not None and result.info.get("captured"):
            captures[player] += 1
        if result.done:
            return env.winner(), turns, {
                "captures": captures,
                "finished": [env.positions[0].count(FINISH), env.positions[1].count(FINISH)],
            }


def train_tabular(args):
    play_game.args = args
    agent = TabularQAgent(alpha=args.alpha, gamma=args.gamma, seed=args.seed)
    if args.load_model:
        agent.load(args.load_model)

    opponent = RuleBasedAgent()
    best_rule = -1.0

    for episode in range(1, args.episodes + 1):
        progress = episode / max(1, args.episodes)
        epsilon = max(args.epsilon_end, args.epsilon_start - progress * (args.epsilon_start - args.epsilon_end))
        env = YutEnv(seed=args.seed + episode)
        env.reset()

        while True:
            if env.current_player == 0:
                state_key = agent.state_key(env)
                action = agent.act(env, epsilon=epsilon)
                result = env.step(action)
                if result.done or env.current_player != 0:
                    next_key = None
                    next_legal = []
                else:
                    next_key = agent.state_key(env, player=0)
                    next_legal = env.legal_actions()
                reward = shape_reward(args, result.reward, result.done, env.winner() if result.done else None, 0)
                agent.update(state_key, action, reward, next_key, next_legal, result.done)
            else:
                action = opponent.act(env)
                result = env.step(action)

            if result.done:
                break

        if args.eval_interval and episode % args.eval_interval == 0:
            random_result = evaluate(agent, RandomAgent(seed=args.seed + episode), games=args.eval_games, seed_offset=60_000 + episode)
            rule_result = evaluate(agent, opponent, games=args.eval_games, seed_offset=70_000 + episode)
            if rule_result["p0_win_rate"] > best_rule:
                best_rule = rule_result["p0_win_rate"]
                agent.save(args.best_model)
            print(
                f"[episode {episode:>6}] "
                f"epsilon={epsilon:.3f} "
                f"vs_random={random_result['p0_win_rate']:.3f} "
                f"vs_rule={rule_result['p0_win_rate']:.3f} "
                f"best_rule={best_rule:.3f}"
            )
    return agent


def train_dqn(args):
    play_game.args = args
    probe_env = YutEnv(seed=args.seed)
    state_dim = len(probe_env.reset())
    dqn_cls = {
        "dqn": DQNAgent,
        "double-dqn": DoubleDQNAgent,
        "dueling-dqn": DuelingDQNAgent,
    }[args.agent]
    dqn = dqn_cls(
        state_dim=state_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        grad_clip=args.grad_clip,
        seed=args.seed,
    )
    if args.load_model:
        dqn.load(args.load_model)

    opponent = RuleBasedAgent()
    replay = ReplayBuffer(capacity=args.replay_capacity)
    best_rule = -1.0

    for episode in range(1, args.episodes + 1):
        progress = episode / max(1, args.episodes)
        epsilon = max(args.epsilon_end, args.epsilon_start - progress * (args.epsilon_start - args.epsilon_end))
        if episode % 2:
            play_game(
                dqn,
                opponent,
                seed=args.seed + episode,
                train_agent=dqn,
                replay=replay,
                epsilon=epsilon,
                batch_size=args.batch_size,
            )
        else:
            play_game(
                opponent,
                dqn,
                seed=args.seed + episode,
                train_agent=dqn,
                replay=replay,
                epsilon=epsilon,
                batch_size=args.batch_size,
            )
        if episode % args.target_sync == 0:
            dqn.sync_target()
        if args.eval_interval and episode % args.eval_interval == 0:
            random_result = evaluate(dqn, RandomAgent(seed=args.seed + episode), games=args.eval_games, seed_offset=20_000 + episode)
            rule_result = evaluate(dqn, opponent, games=args.eval_games, seed_offset=30_000 + episode)
            if rule_result["p0_win_rate"] > best_rule:
                best_rule = rule_result["p0_win_rate"]
                dqn.save(args.best_model)
            print(
                f"[episode {episode:>6}] "
                f"epsilon={epsilon:.3f} "
                f"vs_random={random_result['p0_win_rate']:.3f} "
                f"vs_rule={rule_result['p0_win_rate']:.3f} "
                f"best_rule={best_rule:.3f}"
            )
    return dqn


def train_value(args):
    play_game.args = args
    probe_env = YutEnv(seed=args.seed)
    state_dim = len(probe_env.reset())
    value_cls = StrategicValueNetworkAgent if args.agent == "strategic-value" else ValueNetworkAgent
    value_kwargs = {"heuristic_weight": args.heuristic_weight} if value_cls is StrategicValueNetworkAgent else {}
    value_agent = value_cls(
        state_dim=state_dim,
        hidden_dim=args.hidden_dim,
        lookahead_depth=args.lookahead_depth,
        lr=args.lr,
        gamma=args.gamma,
        grad_clip=args.grad_clip,
        seed=args.seed,
        **value_kwargs,
    )
    if args.load_model:
        value_agent.load(args.load_model)

    rng = random.Random(args.seed)
    replay = ReplayBuffer(capacity=args.replay_capacity)
    fixed_opponents = [RandomAgent(seed=args.seed + 100), RuleBasedAgent()]
    snapshot_opponents: list[ValueNetworkAgent] = []
    best_rule = -1.0

    for episode in range(1, args.episodes + 1):
        progress = episode / max(1, args.episodes)
        epsilon = max(args.epsilon_end, args.epsilon_start - progress * (args.epsilon_start - args.epsilon_end))
        opponents = [value_agent, fixed_opponents[0], fixed_opponents[1]] + snapshot_opponents
        weights = [
            args.self_opponent_weight,
            args.random_opponent_weight,
            args.rule_opponent_weight,
        ] + [args.snapshot_opponent_weight] * len(snapshot_opponents)
        opponent = rng.choices(opponents, weights=weights, k=1)[0]
        if rng.random() < 0.5:
            agent0, agent1 = value_agent, opponent
        else:
            agent0, agent1 = opponent, value_agent
        play_game(
            agent0,
            agent1,
            seed=args.seed + episode,
            train_agent=value_agent,
            replay=replay,
            epsilon=epsilon,
            batch_size=args.batch_size,
        )
        if episode % args.target_sync == 0:
            value_agent.sync_target()
        if args.snapshot_interval and episode % args.snapshot_interval == 0:
            snapshot_opponents.append(value_agent.clone_frozen())
            snapshot_opponents = snapshot_opponents[-args.opponent_pool_size :]
        if args.eval_interval and episode % args.eval_interval == 0:
            random_result = evaluate(value_agent, RandomAgent(seed=args.seed + episode), games=args.eval_games, seed_offset=40_000 + episode)
            rule_result = evaluate(value_agent, RuleBasedAgent(), games=args.eval_games, seed_offset=50_000 + episode)
            if rule_result["p0_win_rate"] > best_rule:
                best_rule = rule_result["p0_win_rate"]
                value_agent.save(args.best_model)
            print(
                f"[episode {episode:>6}] "
                f"epsilon={epsilon:.3f} "
                f"snapshots={len(snapshot_opponents)} "
                f"rule_weight={args.rule_opponent_weight:.2f} "
                f"vs_random={random_result['p0_win_rate']:.3f} "
                f"vs_rule={rule_result['p0_win_rate']:.3f} "
                f"best_rule={best_rule:.3f}"
            )
    return value_agent


def collect_policy_episode(agent, opponent, seed: int, epsilon: float = 0.0):
    env = YutEnv(seed=seed)
    env.reset()
    trajectory = []
    turns = 0
    while True:
        if env.winner() is not None:
            return trajectory, env.winner(), turns
        current = agent if env.current_player == 0 else opponent
        if current is agent:
            state = env.observe_for(0)
            legal = env.legal_actions()
            action = agent.act(env, epsilon=epsilon)
            result = env.step(action)
            reward = result.reward
            winner = env.winner() if result.done else None
            reward = shape_reward(collect_policy_episode.args, reward, result.done, winner, 0)
            trajectory.append(
                {
                    "state": state,
                    "legal": legal,
                    "action": action,
                    "reward": reward,
                    "done": result.done,
                }
            )
        else:
            result = env.step(current.act(env))
            if result.done and trajectory and env.winner() != 0:
                trajectory[-1]["reward"] = -1.0
        turns += 1
        if result.done:
            return trajectory, env.winner(), turns


def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    out = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        out.append(running)
    return list(reversed(out))


collect_policy_episode.args = argparse.Namespace(reward_mode="dense", dense_reward_scale=0.5)


def normalize_tensor(torch, values):
    if values.numel() < 2:
        return values
    std = values.std(unbiased=False)
    if float(std.item()) < 1e-8:
        return values - values.mean()
    return (values - values.mean()) / (std + 1e-8)


def train_policy(args):
    collect_policy_episode.args = args
    probe_env = YutEnv(seed=args.seed)
    state_dim = len(probe_env.reset())
    agent_cls = {
        "reinforce": ReinforceAgent,
        "a2c": A2CAgent,
        "ppo": PPOAgent,
    }[args.agent]
    agent = agent_cls(
        state_dim=state_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        grad_clip=args.grad_clip,
        seed=args.seed,
    )
    if args.load_model:
        agent.load(args.load_model)

    torch = agent.torch
    opponent = RuleBasedAgent()
    best_rule = -1.0

    for episode in range(1, args.episodes + 1):
        progress = episode / max(1, args.episodes)
        epsilon = max(args.epsilon_end, args.epsilon_start - progress * (args.epsilon_start - args.epsilon_end))
        trajectories = [
            collect_policy_episode(agent, opponent, args.seed + episode * max(1, args.rollout_episodes) + idx, epsilon=epsilon)[0]
            for idx in range(args.rollout_episodes)
        ]
        trajectory = [step for item in trajectories for step in item]
        if trajectory:
            states = torch.tensor([t["state"] for t in trajectory], dtype=torch.float32)
            actions = torch.tensor([t["action"] for t in trajectory], dtype=torch.long)
            returns_list = []
            for item in trajectories:
                returns_list.extend(discounted_returns([t["reward"] for t in item], args.gamma))
            returns = torch.tensor(returns_list, dtype=torch.float32)
            if args.normalize_returns:
                returns = normalize_tensor(torch, returns)
            legal_actions = [t["legal"] for t in trajectory]
            log_probs, values, entropies = agent.evaluate_actions(states, actions, legal_actions)
            advantages = returns - values.detach()
            if args.normalize_advantages:
                advantages = normalize_tensor(torch, advantages)

            if isinstance(agent, ReinforceAgent):
                loss = -(log_probs * returns).mean() - args.entropy_coef * entropies.mean()
                agent.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(agent.net.parameters()) + list(agent.policy.parameters()) + list(agent.value.parameters()), args.grad_clip)
                agent.optim.step()
            elif isinstance(agent, A2CAgent):
                policy_loss = -(log_probs * advantages).mean()
                value_loss = 0.5 * (values - returns).pow(2).mean()
                entropy_bonus = entropies.mean()
                loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy_bonus
                agent.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(agent.net.parameters()) + list(agent.policy.parameters()) + list(agent.value.parameters()), args.grad_clip)
                agent.optim.step()
            else:
                with torch.no_grad():
                    old_log_probs = log_probs.detach()
                    old_advantages = advantages.detach()
                for _ in range(args.ppo_epochs):
                    new_log_probs, new_values, new_entropies = agent.evaluate_actions(states, actions, legal_actions)
                    ratio = (new_log_probs - old_log_probs).exp()
                    clipped = torch.clamp(ratio, 1 - args.ppo_clip, 1 + args.ppo_clip) * old_advantages
                    policy_loss = -torch.min(ratio * old_advantages, clipped).mean()
                    value_loss = 0.5 * (new_values - returns).pow(2).mean()
                    entropy_bonus = new_entropies.mean()
                    loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy_bonus
                    agent.optim.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(agent.net.parameters()) + list(agent.policy.parameters()) + list(agent.value.parameters()), args.grad_clip)
                    agent.optim.step()

        if args.eval_interval and episode % args.eval_interval == 0:
            random_result = evaluate(agent, RandomAgent(seed=args.seed + episode), games=args.eval_games, seed_offset=80_000 + episode)
            rule_result = evaluate(agent, opponent, games=args.eval_games, seed_offset=90_000 + episode)
            if rule_result["p0_win_rate"] > best_rule:
                best_rule = rule_result["p0_win_rate"]
                agent.save(args.best_model)
            print(
                f"[episode {episode:>6}] "
                f"epsilon={epsilon:.3f} "
                f"vs_random={random_result['p0_win_rate']:.3f} "
                f"vs_rule={rule_result['p0_win_rate']:.3f} "
                f"best_rule={best_rule:.3f}"
            )
    return agent


def train_mcts_value(args):
    value_agent = train_value(args)
    return MCTSValueAgent(value_agent, simulations=args.mcts_simulations, rollout_depth=args.mcts_rollout_depth, seed=args.seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        choices=(
            "tabular",
            "strategic",
            "value",
            "strategic-value",
            "dqn",
            "double-dqn",
            "dueling-dqn",
            "reinforce",
            "a2c",
            "ppo",
            "mcts-value",
        ),
        default="value",
    )
    parser.add_argument("--episodes", type=int, default=1_000)
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=50_000)
    parser.add_argument("--epsilon-start", type=float, default=0.8)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--target-sync", type=int, default=100)
    parser.add_argument("--lookahead-depth", type=int, default=2)
    parser.add_argument("--mcts-simulations", type=int, default=32)
    parser.add_argument("--mcts-rollout-depth", type=int, default=6)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--heuristic-weight", type=float, default=0.35)
    parser.add_argument("--reward-mode", choices=("dense", "terminal", "hybrid"), default="hybrid")
    parser.add_argument("--dense-reward-scale", type=float, default=0.5)
    parser.add_argument("--rollout-episodes", type=int, default=4)
    parser.add_argument("--snapshot-interval", type=int, default=1_000)
    parser.add_argument("--opponent-pool-size", type=int, default=5)
    parser.add_argument("--self-opponent-weight", type=float, default=1.0)
    parser.add_argument("--random-opponent-weight", type=float, default=0.5)
    parser.add_argument("--rule-opponent-weight", type=float, default=4.0)
    parser.add_argument("--snapshot-opponent-weight", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=5_000)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--ppo-clip", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--no-normalize-returns", action="store_false", dest="normalize_returns")
    parser.add_argument("--no-normalize-advantages", action="store_false", dest="normalize_advantages")
    parser.set_defaults(normalize_returns=True, normalize_advantages=True)
    parser.add_argument("--save-model", type=str, default=None)
    parser.add_argument("--best-model", type=str, default=None)
    parser.add_argument("--load-model", type=str, default=None)
    args = parser.parse_args()

    print("Yutnori RL basic experiment")
    print("Rules: 2 players, 4 pieces, capture, stacking, bonus rolls, no back-do")
    print()

    random_agent = RandomAgent(seed=args.seed)
    rule_agent = RuleBasedAgent()
    print("[baseline] RuleBased vs Random")
    print(evaluate(rule_agent, random_agent, games=args.eval_games, seed_offset=1_000))
    print()

    if args.save_model is None:
        suffix = "json" if args.agent == "tabular" else "pt"
        args.save_model = f"checkpoints/{args.agent}.{suffix}"
    if args.best_model is None:
        suffix = "json" if args.agent == "tabular" else "pt"
        args.best_model = f"checkpoints/best_{args.agent}_vs_rule.{suffix}"

    if args.agent == "strategic":
        print("[agent] strategic rule-based heuristic agent")
        trained_agent = StrategicRuleBasedAgent(seed=args.seed)
    elif args.agent == "tabular":
        print(
            f"[train] {args.agent} episodes={args.episodes} "
            f"alpha={args.alpha} gamma={args.gamma}"
        )
        trained_agent = train_tabular(args)
    elif args.agent in {"reinforce", "a2c", "ppo"}:
        print(
            f"[train] {args.agent} episodes={args.episodes} "
            f"lr={args.lr} gamma={args.gamma} hidden_dim={args.hidden_dim}"
        )
        trained_agent = train_policy(args)
    elif args.agent == "mcts-value":
        print(
            f"[train] {args.agent} episodes={args.episodes} "
            f"lr={args.lr} gamma={args.gamma} hidden_dim={args.hidden_dim} "
            f"lookahead_depth={args.lookahead_depth} simulations={args.mcts_simulations}"
        )
        trained_agent = train_mcts_value(args)
    else:
        print(
            f"[train] {args.agent} episodes={args.episodes} "
            f"lr={args.lr} gamma={args.gamma} hidden_dim={args.hidden_dim} batch_size={args.batch_size}"
        )
        trained_agent = train_value(args) if args.agent in {"value", "strategic-value"} else train_dqn(args)
    print()

    print(f"[eval] {args.agent} vs Random")
    print(evaluate(trained_agent, RandomAgent(seed=args.seed + 1), games=args.eval_games, seed_offset=2_000))
    print()

    print(f"[eval] {args.agent} vs RuleBased")
    print(evaluate(trained_agent, rule_agent, games=args.eval_games, seed_offset=3_000))

    if args.save_model and hasattr(trained_agent, "save"):
        save_path = Path(args.save_model)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        trained_agent.save(save_path)
        print()
        print(f"[saved] {save_path}")


if __name__ == "__main__":
    main()
