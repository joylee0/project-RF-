from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from yut_rl.env import ACTION_DIM, FINISH, MAX_STEPS, START, YutEnv, advance, distance_to_finish


def encode_piece_yut_action(piece: int, steps: int) -> int:
    """Agent-facing action: action = piece_id * 5 + yut_id."""
    return piece * MAX_STEPS + (steps - 1)


def decode_piece_yut_action(action: int) -> tuple[int, int]:
    return action // MAX_STEPS, action % MAX_STEPS + 1


def legal_agent_actions(env: YutEnv) -> list[int]:
    return env.legal_actions()


def action_mask_tensor(legal_actions: list[int], device: torch.device) -> torch.Tensor:
    mask = torch.zeros(ACTION_DIM, dtype=torch.bool, device=device)
    mask[legal_actions] = True
    return mask


class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = ACTION_DIM, hidden_dim: int = 256):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        features = self.feature(states)
        values = self.value(features)
        advantages = self.advantage(features)
        return values + advantages - advantages.mean(dim=1, keepdim=True)


@dataclass
class Transition:
    state: list[float]
    action: int
    reward: float
    next_state: list[float]
    next_legal_actions: list[int]
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int):
        self.items: deque[Transition] = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def add(self, item: Transition) -> None:
        self.items.append(item)

    def sample(self, batch_size: int) -> list[Transition]:
        return self.rng.sample(list(self.items), batch_size)

    def __len__(self) -> int:
        return len(self.items)


class DuelingDoubleDQNAgent:
    model_type = "Pure Dueling Double DQN"

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        lr: float = 1e-4,
        gamma: float = 0.99,
        seed: int = 0,
        device: str | None = None,
    ):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.rng = random.Random(seed)
        self.device = torch.device(device or default_device())
        torch.manual_seed(seed)
        self.online = DuelingQNetwork(state_dim, hidden_dim=hidden_dim).to(self.device)
        self.target = DuelingQNetwork(state_dim, hidden_dim=hidden_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optim = torch.optim.Adam(self.online.parameters(), lr=lr)

    def select_action(self, observation: list[float], legal_actions: list[int], epsilon: float = 0.0) -> int:
        if self.rng.random() < epsilon:
            return self.rng.choice(legal_actions)
        with torch.no_grad():
            state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.online(state).squeeze(0)
            mask = action_mask_tensor(legal_actions, self.device)
            q_values = q_values.masked_fill(~mask, -1e9)
            return int(torch.argmax(q_values).item())

    def update(self, batch: list[Transition]) -> float:
        states = torch.tensor([item.state for item in batch], dtype=torch.float32, device=self.device)
        actions = torch.tensor([item.action for item in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([item.reward for item in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor([item.next_state for item in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([item.done for item in batch], dtype=torch.float32, device=self.device)

        q_sa = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_online = self.online(next_states)
            for row, item in enumerate(batch):
                if item.done or not item.next_legal_actions:
                    next_online[row] = -1e9
                else:
                    mask = action_mask_tensor(item.next_legal_actions, self.device)
                    next_online[row] = next_online[row].masked_fill(~mask, -1e9)
            next_actions = torch.argmax(next_online, dim=1)
            next_target = self.target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards + self.gamma * (1.0 - dones) * next_target

        loss = F.smooth_l1_loss(q_sa, targets)
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optim.step()
        return float(loss.item())

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def soft_update_target(self, tau: float) -> None:
        with torch.no_grad():
            for target_param, online_param in zip(self.target.parameters(), self.online.parameters()):
                target_param.mul_(1.0 - tau).add_(online_param, alpha=tau)

    def save(self, path: Path, args: argparse.Namespace) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "state_dim": self.state_dim,
                "hidden_dim": self.hidden_dim,
                "gamma": self.gamma,
                "args": vars(args),
                "action_encoding": "piece_yut",
                "reward": args.reward_function,
                "state": args.state_encoder,
                "pure_rl_baseline": True,
            },
            path,
        )

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.online.load_state_dict(checkpoint["online"])
        self.target.load_state_dict(checkpoint.get("target", checkpoint["online"]))


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def configure_compute(args: argparse.Namespace) -> None:
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    if args.torch_interop_threads > 0:
        torch.set_num_interop_threads(args.torch_interop_threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested with --device cuda, but torch.cuda.is_available() is False.")
    if args.device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested with --device mps, but torch.backends.mps.is_available() is False.")


class FixedRuleBasedOpponent:
    """Fixed rule-based opponent compatible with piece-major action encoding."""

    def select_action(self, env: YutEnv) -> int:
        legal = env.legal_actions()
        return min(legal, key=lambda action: (-self.score_action(env, action), action))

    @staticmethod
    def score_action(env: YutEnv, action: int) -> float:
        player = env.current_player
        opponent = 1 - player
        piece, steps = env.decode_action(action)
        old_pos = env.positions[player][piece]
        new_pos = advance(old_pos, steps)
        opponent_positions = {
            pos for pos in env.positions[opponent]
            if pos not in (START, FINISH)
        }
        stack_size = 1 if old_pos in (START, FINISH) else env.positions[player].count(old_pos)
        score = 0.0
        if new_pos == FINISH:
            score += 100.0
        if new_pos in opponent_positions:
            score += 50.0
        if old_pos == START:
            score += 5.0
        score += 4.0 * max(0, stack_size - 1)
        score -= 0.5 * distance_to_finish(new_pos)
        return score


def make_env(args: argparse.Namespace, seed: int) -> YutEnv:
    env = YutEnv(
        seed=seed,
        max_turns=args.max_turns,
        state_encoder=args.state_encoder,
        reward_function=args.reward_function,
        action_encoding="piece_yut",
        enable_action_mask=True,
    )
    env.reset()
    return env


def reward_for_player(before: YutEnv, after: YutEnv, info: dict, done: bool, player: int) -> float:
    if after.reward_function is None:
        if done and after.winner() is not None:
            return 1.0 if after.winner() == player else -1.0
        return 0.0
    return after.reward_function(before, after, info, done, player)


def epsilon_by_step(args: argparse.Namespace, step: int) -> float:
    progress = min(1.0, step / max(1, args.epsilon_decay_steps))
    return args.epsilon_start + progress * (args.epsilon_end - args.epsilon_start)


def step_until_agent_turn(
    env: YutEnv,
    agent_action: int,
    dqn_player: int,
    opponent: FixedRuleBasedOpponent,
) -> tuple[list[float], float, bool, dict]:
    before = env.clone()
    result = env.step(agent_action)
    if result.info.get("illegal"):
        return env.observe_for(dqn_player), -1.0, True, result.info
    reward = reward_for_player(before, env, result.info, result.done, dqn_player)
    if result.done:
        return env.observe_for(dqn_player), reward, True, result.info

    while env.current_player != dqn_player:
        before = env.clone()
        opp_action = opponent.select_action(env)
        result = env.step(opp_action)
        reward += reward_for_player(before, env, result.info, result.done, dqn_player)
        if result.done:
            return env.observe_for(dqn_player), reward, True, result.info

    return env.observe_for(dqn_player), reward, False, result.info


def run_training_episode(
    agent: DuelingDoubleDQNAgent,
    replay: ReplayBuffer,
    args: argparse.Namespace,
    episode: int,
    global_step: int,
) -> dict:
    env = make_env(args, args.seed + episode)
    opponent = FixedRuleBasedOpponent()
    dqn_player = episode % 2

    while env.current_player != dqn_player:
        opp_action = opponent.select_action(env)
        result = env.step(opp_action)
        if result.done:
            return {"win": 0, "turns": env.turn_count, "epsilon": 0.0, "loss": 0.0, "agent_steps": 0}

    losses = []
    agent_steps = 0
    epsilon = epsilon_by_step(args, global_step)
    done = False
    while not done:
        epsilon = epsilon_by_step(args, global_step + agent_steps)
        state = env.observe_for(dqn_player)
        legal = legal_agent_actions(env)
        action = agent.select_action(state, legal, epsilon=epsilon)
        next_state, reward, done, info = step_until_agent_turn(env, action, dqn_player, opponent)
        next_legal = [] if done else legal_agent_actions(env)
        replay.add(Transition(state, action, reward, next_state, next_legal, done))
        agent_steps += 1

        if len(replay) >= args.batch_size and global_step + agent_steps >= args.learning_starts:
            losses.append(agent.update(replay.sample(args.batch_size)))

    winner = env.winner()
    return {
        "win": int(winner == dqn_player),
        "turns": env.turn_count,
        "epsilon": epsilon,
        "loss": float(np.mean(losses)) if losses else 0.0,
        "illegal": int(info.get("illegal", False)),
        "agent_steps": agent_steps,
    }


def evaluate(agent: DuelingDoubleDQNAgent, args: argparse.Namespace, games: int, seed: int) -> dict:
    opponent = FixedRuleBasedOpponent()
    wins = 0
    turns = 0
    first_wins = 0
    second_wins = 0
    for game in range(games):
        env = make_env(args, seed + game)
        dqn_player = game % 2
        done = False
        while not done:
            if env.current_player == dqn_player:
                action = agent.select_action(env.observe_for(dqn_player), legal_agent_actions(env), epsilon=0.0)
                result = env.step(action)
            else:
                action = opponent.select_action(env)
                result = env.step(action)
            done = result.done
        win = int(env.winner() == dqn_player)
        wins += win
        if dqn_player == 0:
            first_wins += win
        else:
            second_wins += win
        turns += env.turn_count

    first_games = (games + 1) // 2
    second_games = games // 2
    return {
        "games": games,
        "win_rate": wins / games,
        "first_player_win_rate": first_wins / max(1, first_games),
        "second_player_win_rate": second_wins / max(1, second_games),
        "avg_turns": turns / games,
    }


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a pure RL Dueling Double DQN baseline against a fixed Rule-based Agent.")
    parser.add_argument("--out-dir", default="results/pure_dueling_dqn")
    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--eval-every", type=int, default=1_000)
    parser.add_argument("--eval-games", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--target-update-every", type=int, default=1)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=200_000)
    parser.add_argument("--max-turns", type=int, default=1_000)
    parser.add_argument("--state-encoder", default="raw", choices=["raw"])
    parser.add_argument("--reward-function", default="sparse", choices=["sparse"])
    parser.add_argument("--device", default=None, choices=[None, "cpu", "cuda", "mps"], help="Training device. Defaults to torch.device('cuda' if available else 'cpu') style selection.")
    parser.add_argument("--torch-threads", type=int, default=min(32, os.cpu_count() or 1), help="CPU threads used by PyTorch tensor ops.")
    parser.add_argument("--torch-interop-threads", type=int, default=4, help="PyTorch inter-op parallelism threads.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_compute(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    probe_env = make_env(args, args.seed)
    state_dim = len(probe_env.observe())
    agent = DuelingDoubleDQNAgent(
        state_dim=state_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        seed=args.seed,
        device=args.device,
    )
    replay = ReplayBuffer(args.buffer_size, seed=args.seed)

    rows = []
    eval_rows = []
    global_steps = 0
    best_win_rate = -1.0
    best_path = out_dir / "dueling_double_dqn_best.pt"
    latest_path = out_dir / "dueling_double_dqn_latest.pt"

    for episode in range(1, args.episodes + 1):
        row = run_training_episode(agent, replay, args, episode, global_steps)
        global_steps += int(row.get("agent_steps", 0))
        rows.append({"episode": episode, **row})

        if episode % args.target_update_every == 0:
            agent.soft_update_target(args.tau)

        if episode % args.eval_every == 0:
            recent = pd.DataFrame(rows[-args.eval_every:])
            metrics = evaluate(agent, args, args.eval_games, seed=args.seed + 1_000_000 + episode)
            eval_row = {
                "episode": episode,
                "train_recent_win_rate": float(recent["win"].mean()),
                "train_recent_loss": float(recent["loss"].mean()),
                "train_recent_turns": float(recent["turns"].mean()),
                "agent_steps": global_steps,
                **metrics,
            }
            eval_rows.append(eval_row)
            print(
                f"episode={episode} "
                f"train_win={eval_row['train_recent_win_rate']:.3f} "
                f"eval_win={eval_row['win_rate']:.3f} "
                f"avg_turns={eval_row['avg_turns']:.1f}"
            )
            pd.DataFrame(rows).to_csv(out_dir / "dueling_double_dqn_train_log.csv", index=False)
            pd.DataFrame(eval_rows).to_csv(out_dir / "dueling_double_dqn_eval_log.csv", index=False)
            agent.save(latest_path, args)
            if metrics["win_rate"] > best_win_rate:
                best_win_rate = metrics["win_rate"]
                agent.save(best_path, args)

    summary = {
        "agent": "Dueling Double DQN",
        "algorithm": "Double Dueling DQN",
        "network": "MLP [256, 256]",
        "activation": "ReLU",
        "optimizer": "Adam",
        "learning_rate": args.lr,
        "gamma": args.gamma,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
        "learning_starts": args.learning_starts,
        "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end,
        "epsilon_decay_steps": args.epsilon_decay_steps,
        "target_update": f"soft update tau={args.tau}",
        "setting": "pure_rl_raw_sparse_dueling_double_dqn",
        "state": args.state_encoder,
        "action": "piece_id * 5 + yut_id",
        "action_masking": True,
        "reward": args.reward_function,
        "reward_detail": "win +1 / lose -1 / otherwise 0",
        "reward_shaping": False,
        "engineered_state": False,
        "imitation_learning": False,
        "tactical_prior": False,
        "opponent": "fixed Rule-based Agent",
        "state_dim": state_dim,
        "action_dim": ACTION_DIM,
        "episodes": args.episodes,
        "seed": args.seed,
        "device": str(agent.device),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "best_eval_win_rate": best_win_rate,
        "best_checkpoint": str(best_path),
        "latest_checkpoint": str(latest_path),
    }
    write_json(out_dir / "dueling_double_dqn_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
