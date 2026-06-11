from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
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

from common_rule_based_env import (
    ACTION_DIM,
    CommonRuleBasedAgent,
    CommonYutEnv,
    MAX_STEPS,
    PIECES_PER_PLAYER,
    decode_action as env_decode_action,
    encode_action as env_encode_action,
)


def encode_piece_yut_action(piece: int, steps: int) -> int:
    """Agent-facing action: action = piece_id * 5 + yut_id."""
    return piece * MAX_STEPS + (steps - 1)


def decode_piece_yut_action(action: int) -> tuple[int, int]:
    return action // MAX_STEPS, action % MAX_STEPS + 1


def env_action_to_agent_action(env_action: int) -> int:
    piece, steps = env_decode_action(env_action)
    return encode_piece_yut_action(piece, steps)


def agent_action_to_env_action(agent_action: int) -> int:
    piece, steps = decode_piece_yut_action(agent_action)
    return env_encode_action(piece, steps)


def legal_agent_actions(env: CommonYutEnv) -> list[int]:
    return [env_action_to_agent_action(action) for action in env.legal_actions()]


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
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
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
                "reward": "win/loss terminal only",
            },
            path,
        )

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.online.load_state_dict(checkpoint["online"])
        self.target.load_state_dict(checkpoint.get("target", checkpoint["online"]))


def step_until_agent_turn(
    env: CommonYutEnv,
    agent_action: int,
    dqn_player: int,
    opponent: CommonRuleBasedAgent,
) -> tuple[list[float], float, bool, dict]:
    result = env.step(agent_action_to_env_action(agent_action))
    if result.info.get("illegal"):
        return env.observe_for(dqn_player), -1.0, True, {"illegal": True}
    if result.done:
        winner = env.winner()
        return env.observe_for(dqn_player), 1.0 if winner == dqn_player else -1.0, True, result.info

    while env.current_player != dqn_player:
        legal = env.legal_actions()
        opp_action = opponent.select_action(env.observe(), legal, env=env)
        result = env.step(opp_action)
        if result.done:
            winner = env.winner()
            return env.observe_for(dqn_player), 1.0 if winner == dqn_player else -1.0, True, result.info

    return env.observe_for(dqn_player), 0.0, False, result.info


def run_training_episode(
    agent: DuelingDoubleDQNAgent,
    replay: ReplayBuffer,
    args: argparse.Namespace,
    episode: int,
) -> dict:
    env = CommonYutEnv(seed=args.seed + episode, max_decisions=args.max_decisions)
    env.reset()
    opponent = CommonRuleBasedAgent()
    dqn_player = episode % 2

    while env.current_player != dqn_player:
        opp_action = opponent.select_action(env.observe(), env.legal_actions(), env=env)
        result = env.step(opp_action)
        if result.done:
            return {"win": 0, "turns": env.decision_count, "epsilon": 0.0, "loss": 0.0}

    epsilon = max(args.epsilon_final, args.epsilon_start - episode / max(1, args.epsilon_decay_episodes) * (args.epsilon_start - args.epsilon_final))
    losses = []
    done = False
    while not done:
        state = env.observe_for(dqn_player)
        legal = legal_agent_actions(env)
        action = agent.select_action(state, legal, epsilon=epsilon)
        next_state, reward, done, info = step_until_agent_turn(env, action, dqn_player, opponent)
        next_legal = [] if done else legal_agent_actions(env)
        replay.add(Transition(state, action, reward, next_state, next_legal, done))

        if len(replay) >= args.batch_size and episode >= args.learning_starts:
            losses.append(agent.update(replay.sample(args.batch_size)))

    winner = env.winner()
    return {
        "win": int(winner == dqn_player),
        "turns": env.decision_count,
        "epsilon": epsilon,
        "loss": float(np.mean(losses)) if losses else 0.0,
        "illegal": int(info.get("illegal", False)),
    }


def evaluate(agent: DuelingDoubleDQNAgent, games: int, seed: int, max_decisions: int) -> dict:
    opponent = CommonRuleBasedAgent()
    wins = 0
    turns = 0
    first_wins = 0
    second_wins = 0
    for game in range(games):
        env = CommonYutEnv(seed=seed + game, max_decisions=max_decisions)
        env.reset()
        dqn_player = game % 2
        done = False
        while not done:
            if env.current_player == dqn_player:
                action = agent.select_action(env.observe_for(dqn_player), legal_agent_actions(env), epsilon=0.0)
                result = env.step(agent_action_to_env_action(action))
            else:
                action = opponent.select_action(env.observe(), env.legal_actions(), env=env)
                result = env.step(action)
            done = result.done
        win = int(env.winner() == dqn_player)
        wins += win
        if dqn_player == 0:
            first_wins += win
        else:
            second_wins += win
        turns += env.decision_count

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
    parser = argparse.ArgumentParser(description="Train a pure Dueling Double DQN baseline against the fixed common Rule-based Agent.")
    parser.add_argument("--out-dir", default="results/pure_dqn")
    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--eval-every", type=int, default=1_000)
    parser.add_argument("--eval-games", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-starts", type=int, default=200)
    parser.add_argument("--target-update-every", type=int, default=500)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-final", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-episodes", type=int, default=12_000)
    parser.add_argument("--max-decisions", type=int, default=10_000)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    probe_env = CommonYutEnv(seed=args.seed)
    state_dim = len(probe_env.reset())
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
    best_win_rate = -1.0
    best_path = out_dir / "dueling_double_dqn_best.pt"
    latest_path = out_dir / "dueling_double_dqn_latest.pt"

    for episode in range(1, args.episodes + 1):
        row = run_training_episode(agent, replay, args, episode)
        rows.append({"episode": episode, **row})

        if episode % args.target_update_every == 0:
            agent.sync_target()

        if episode % args.eval_every == 0:
            recent = pd.DataFrame(rows[-args.eval_every:])
            metrics = evaluate(agent, args.eval_games, seed=args.seed + 1_000_000 + episode, max_decisions=args.max_decisions)
            eval_row = {
                "episode": episode,
                "train_recent_win_rate": float(recent["win"].mean()),
                "train_recent_loss": float(recent["loss"].mean()),
                "train_recent_turns": float(recent["turns"].mean()),
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
        "setting": "pure_rl_raw_state_win_loss_reward",
        "state": "CommonYutEnv raw observation",
        "action": "piece_id * 5 + yut_id",
        "reward": "win +1 / loss -1 / otherwise 0",
        "opponent": "fixed CommonRuleBasedAgent",
        "state_dim": state_dim,
        "action_dim": ACTION_DIM,
        "episodes": args.episodes,
        "seed": args.seed,
        "best_eval_win_rate": best_win_rate,
        "best_checkpoint": str(best_path),
        "latest_checkpoint": str(latest_path),
    }
    write_json(out_dir / "dueling_double_dqn_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
