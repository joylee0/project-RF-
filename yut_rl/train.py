from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

from .agents import DQNAgent, RandomAgent, ReplayBuffer, RuleBasedAgent, Transition, ValueNetworkAgent, ValueTransition
from .env import YutEnv


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
    while True:
        agent = agent0 if env.current_player == 0 else agent1
        if isinstance(agent, (DQNAgent, ValueNetworkAgent)):
            action = agent.act(env, epsilon=epsilon)
        else:
            action = agent.act(env)

        acting_player = env.current_player
        state = env.observe_for(acting_player)
        result = env.step(action)
        turns += 1

        if replay is not None and agent is train_agent:
            if isinstance(train_agent, DQNAgent):
                replay.push(
                    Transition(
                        state=state,
                        action=action,
                        reward=result.reward,
                        next_state=result.observation,
                        done=result.done,
                        next_legal=[] if result.done else env.legal_actions(),
                    )
                )
            else:
                replay.push(
                    ValueTransition(
                        state=state,
                        reward=result.reward,
                        next_state=env.observe_for(acting_player),
                        done=result.done,
                    )
                )
        if train_agent is not None and replay is not None:
            train_agent.train_batch(replay, batch_size=batch_size)

        if result.done:
            return env.winner(), turns


def evaluate(agent0, agent1, games: int = 200, seed_offset: int = 10_000):
    wins = [0, 0]
    lengths = []
    for i in range(games):
        winner, turns = play_game(agent0, agent1, seed=seed_offset + i)
        if winner is not None:
            wins[winner] += 1
        lengths.append(turns)
    return {
        "p0_win_rate": wins[0] / games,
        "p1_win_rate": wins[1] / games,
        "avg_turns": mean(lengths),
    }


def train_dqn(args):
    probe_env = YutEnv(seed=args.seed)
    state_dim = len(probe_env.reset())
    dqn = DQNAgent(state_dim=state_dim, lr=args.lr, gamma=args.gamma, seed=args.seed)
    if args.load_model:
        dqn.load(args.load_model)

    opponent = RuleBasedAgent()
    replay = ReplayBuffer(capacity=args.replay_capacity)

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
            print(
                f"[episode {episode:>6}] "
                f"epsilon={epsilon:.3f} "
                f"vs_random={random_result['p0_win_rate']:.3f} "
                f"vs_rule={rule_result['p0_win_rate']:.3f}"
            )
    return dqn


def train_value(args):
    probe_env = YutEnv(seed=args.seed)
    state_dim = len(probe_env.reset())
    value_agent = ValueNetworkAgent(state_dim=state_dim, lr=args.lr, gamma=args.gamma, seed=args.seed)
    if args.load_model:
        value_agent.load(args.load_model)

    replay = ReplayBuffer(capacity=args.replay_capacity)

    for episode in range(1, args.episodes + 1):
        progress = episode / max(1, args.episodes)
        epsilon = max(args.epsilon_end, args.epsilon_start - progress * (args.epsilon_start - args.epsilon_end))
        play_game(
            value_agent,
            value_agent,
            seed=args.seed + episode,
            train_agent=value_agent,
            replay=replay,
            epsilon=epsilon,
            batch_size=args.batch_size,
        )
        if episode % args.target_sync == 0:
            value_agent.sync_target()
        if args.eval_interval and episode % args.eval_interval == 0:
            random_result = evaluate(value_agent, RandomAgent(seed=args.seed + episode), games=args.eval_games, seed_offset=40_000 + episode)
            rule_result = evaluate(value_agent, RuleBasedAgent(), games=args.eval_games, seed_offset=50_000 + episode)
            print(
                f"[episode {episode:>6}] "
                f"epsilon={epsilon:.3f} "
                f"vs_random={random_result['p0_win_rate']:.3f} "
                f"vs_rule={rule_result['p0_win_rate']:.3f}"
            )
    return value_agent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=("value", "dqn"), default="value")
    parser.add_argument("--episodes", type=int, default=50_000)
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=50_000)
    parser.add_argument("--epsilon-start", type=float, default=0.8)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--target-sync", type=int, default=25)
    parser.add_argument("--eval-interval", type=int, default=5_000)
    parser.add_argument("--save-model", type=str, default=None)
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
        args.save_model = f"checkpoints/{args.agent}.pt"

    print(
        f"[train] {args.agent} episodes={args.episodes} "
        f"lr={args.lr} gamma={args.gamma} batch_size={args.batch_size}"
    )
    trained_agent = train_value(args) if args.agent == "value" else train_dqn(args)
    print()

    print(f"[eval] {args.agent} vs Random")
    print(evaluate(trained_agent, RandomAgent(seed=args.seed + 1), games=args.eval_games, seed_offset=2_000))
    print()

    print(f"[eval] {args.agent} vs RuleBased")
    print(evaluate(trained_agent, rule_agent, games=args.eval_games, seed_offset=3_000))

    if args.save_model:
        save_path = Path(args.save_model)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        trained_agent.save(save_path)
        print()
        print(f"[saved] {save_path}")


if __name__ == "__main__":
    main()
