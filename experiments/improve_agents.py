from __future__ import annotations

import argparse
from dataclasses import dataclass
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

from yut_rl.agents import (
    DQNAgent,
    RandomAgent,
    ReplayBuffer,
    RuleBasedAgent,
    StrategicRuleBasedAgent,
    StrategicValueNetworkAgent,
    Transition,
    ValueNetworkAgent,
)
from yut_rl.env import FINISH, YutEnv, distance_to_finish


@dataclass
class NamedAgent:
    name: str
    agent: object


def get_state_dim(seed: int) -> int:
    env = YutEnv(seed=seed)
    return len(env.reset())


def load_if_possible(agent, path: str | None) -> None:
    if path and Path(path).exists() and hasattr(agent, "load"):
        agent.load(path)


def choose_action(agent, env: YutEnv, epsilon: float = 0.0) -> int:
    if isinstance(agent, (DQNAgent, ValueNetworkAgent)):
        return agent.act(env, epsilon=epsilon)
    return agent.act(env)


def play_eval_game(agent0, name0: str, agent1, name1: str, seed: int) -> dict:
    env = YutEnv(seed=seed)
    env.reset()
    names = [name0, name1]
    agents = [agent0, agent1]
    captures = {name0: 0, name1: 0}
    turns = 0
    while True:
        player = env.current_player
        result = env.step(choose_action(agents[player], env))
        turns += 1
        actor = result.info.get("player")
        if actor is not None and result.info.get("captured"):
            captures[names[actor]] += 1
        if result.done:
            winner = env.winner()
            return {
                "winner": names[winner] if winner is not None else None,
                "turns": turns,
                "captures": captures,
                "finished": {
                    name0: env.positions[0].count(FINISH),
                    name1: env.positions[1].count(FINISH),
                },
            }


def compare_agents(agents: list[NamedAgent], games: int, seed: int) -> pd.DataFrame:
    totals = {
        item.name: {"wins": 0, "games": 0, "turns": 0, "captures": 0, "finished": 0}
        for item in agents
    }
    for i, left in enumerate(agents):
        for j, right in enumerate(agents[i + 1 :], start=i + 1):
            for game in range(games):
                if game % 2 == 0:
                    result = play_eval_game(left.agent, left.name, right.agent, right.name, seed + i * 20_000 + j * 1_000 + game)
                else:
                    result = play_eval_game(right.agent, right.name, left.agent, left.name, seed + i * 20_000 + j * 1_000 + game)
                for name in (left.name, right.name):
                    totals[name]["wins"] += int(result["winner"] == name)
                    totals[name]["games"] += 1
                    totals[name]["turns"] += result["turns"]
                    totals[name]["captures"] += result["captures"][name]
                    totals[name]["finished"] += result["finished"][name]
    rows = []
    for name, stats in totals.items():
        rows.append(
            {
                "agent": name,
                "games": stats["games"],
                "overall_win_rate": stats["wins"] / stats["games"],
                "avg_turns": stats["turns"] / stats["games"],
                "avg_captures": stats["captures"] / stats["games"],
                "avg_finished": stats["finished"] / stats["games"],
            }
        )
    return pd.DataFrame(rows).sort_values("overall_win_rate", ascending=False)


def strategic_value_candidates(state_dim: int, seed: int, value_model: str | None) -> list[tuple[str, StrategicValueNetworkAgent, dict]]:
    configs = [
        {"heuristic_weight": 0.25, "heuristic_params": {"finish_weight": 130.0, "capture_weight": 55.0, "progress_weight": 3.0, "danger_weight": 28.0, "counterplay_weight": 18.0}},
        {"heuristic_weight": 0.35, "heuristic_params": {"finish_weight": 130.0, "capture_weight": 55.0, "progress_weight": 3.0, "danger_weight": 28.0, "counterplay_weight": 18.0}},
        {"heuristic_weight": 0.45, "heuristic_params": {"finish_weight": 150.0, "capture_weight": 50.0, "progress_weight": 4.0, "danger_weight": 34.0, "counterplay_weight": 20.0}},
        {"heuristic_weight": 0.35, "heuristic_params": {"finish_weight": 170.0, "capture_weight": 45.0, "progress_weight": 4.5, "danger_weight": 38.0, "counterplay_weight": 22.0}},
    ]
    out = []
    for idx, cfg in enumerate(configs):
        agent = StrategicValueNetworkAgent(state_dim=state_dim, seed=seed + idx, **cfg)
        load_if_possible(agent, value_model)
        out.append((f"strategic_value_cfg_{idx}", agent, cfg))
    return out


def tune_strategic_value(state_dim: int, value_model: str | None, games: int, seed: int, out_dir: Path):
    field = [
        NamedAgent("strategic_rule", StrategicRuleBasedAgent(seed=seed + 30)),
        NamedAgent("value", ValueNetworkAgent(state_dim=state_dim, seed=seed + 31)),
    ]
    load_if_possible(field[1].agent, value_model)
    rows = []
    best = None
    for name, agent, params in strategic_value_candidates(state_dim, seed, value_model):
        result = compare_agents([NamedAgent(name, agent), *field], games, seed + 50_000)
        row = result[result["agent"] == name].iloc[0].to_dict()
        rows.append({**row, "params": repr(params)})
        if best is None or row["overall_win_rate"] > best[0]["overall_win_rate"]:
            best = (row, name, agent, params)
    df = pd.DataFrame(rows).sort_values("overall_win_rate", ascending=False)
    df.to_csv(out_dir / "strategic_value_search.csv", index=False)
    return best[1], best[2], best[3], df


def log_strategic_value_contributions(agent: StrategicValueNetworkAgent, games: int, seed: int, out_dir: Path) -> None:
    rows = []
    for game in range(games):
        env = YutEnv(seed=seed + game)
        env.reset()
        opponent = StrategicRuleBasedAgent(seed=seed + 10_000 + game)
        while True:
            if env.current_player == 0:
                action = agent.act(env)
                parts = agent.score_components(env, action, env.current_player)
                rows.append({"game": game, "turn_player": 0, "action": action, **parts})
                result = env.step(action)
            else:
                result = env.step(opponent.act(env))
            if result.done:
                break
    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / "strategic_value_contributions.csv", index=False)
    summary = raw.drop(columns=["game", "turn_player", "action"]).mean(numeric_only=True).reset_index()
    summary.columns = ["component", "mean_value"]
    summary.to_csv(out_dir / "strategic_value_contribution_summary.csv", index=False)


def dqn_diagnostics(out_dir: Path) -> None:
    rows = [
        {"area": "state representation", "finding": "one-hot positions and pending rolls are present, but the state has no learned tactical features such as immediate threat count or route pressure.", "action": "Keep one-hot state, add reward/teacher signals so DQN can learn tactical features indirectly."},
        {"area": "reward shaping", "finding": "terminal rewards are sparse and dense rewards are small for DQN.", "action": "Add progress, capture, finish, and loss shaping in the DQN training loop."},
        {"area": "action masking", "finding": "legal action masking is applied in act() and target calculation.", "action": "Keep masking; also mask logits during imitation pretraining."},
        {"area": "epsilon decay", "finding": "linear decay can explore too little before it understands stronger opponents.", "action": "Use configurable exponential epsilon decay across curriculum phases."},
        {"area": "replay buffer", "finding": "uniform replay is simple and can be dominated by weak early games.", "action": "Use curriculum ordering and teacher samples before self-generated replay dominates."},
        {"area": "target network update", "finding": "fixed sync interval can lag behind curriculum difficulty changes.", "action": "Tune target_sync together with gamma and learning rate."},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "dqn_diagnostics.csv", index=False)
    (out_dir / "dqn_diagnostics.md").write_text(
        "\n".join(f"- **{row['area']}**: {row['finding']} -> {row['action']}" for row in rows),
        encoding="utf-8",
    )


def shaped_dqn_reward(result, acting_player: int, winner: int | None) -> float:
    if result.done and winner is not None:
        return 1.0 if winner == acting_player else -1.0
    reward = result.reward * 0.5
    if result.info.get("captured"):
        reward += 0.08
    old_pos = result.info.get("from")
    new_pos = result.info.get("to")
    if old_pos is not None and new_pos is not None:
        reward += 0.015 * max(0, distance_to_finish(old_pos) - distance_to_finish(new_pos))
        if new_pos == FINISH:
            reward += 0.12
    return reward


def play_dqn_training_game(agent: DQNAgent, opponent, replay: ReplayBuffer, epsilon: float, batch_size: int, seed: int) -> None:
    env = YutEnv(seed=seed)
    env.reset()
    pending = {}

    def push_pending(player: int, done: bool) -> None:
        item = pending.pop(player, None)
        if item is None:
            return
        replay.push(
            Transition(
                state=item["state"],
                action=item["action"],
                reward=item["reward"],
                next_state=env.observe_for(player),
                done=done,
                next_legal=[] if done else env.legal_actions(),
                discount_power=item["discount_power"],
            )
        )

    while True:
        current_agent = agent if env.current_player == 0 else opponent
        if current_agent is agent and 0 in pending:
            push_pending(0, done=False)
        acting_player = env.current_player
        state = env.observe_for(acting_player)
        action = choose_action(current_agent, env, epsilon=epsilon if current_agent is agent else 0.0)
        result = env.step(action)
        winner = env.winner() if result.done else None
        reward = shaped_dqn_reward(result, acting_player, winner)

        if current_agent is agent:
            if result.done:
                replay.push(Transition(state, action, reward, env.observe_for(0), True, []))
            elif env.current_player == acting_player:
                replay.push(Transition(state, action, reward, env.observe_for(0), False, env.legal_actions()))
            else:
                pending[0] = {"state": state, "action": action, "reward": reward, "discount_power": 1}
        elif pending:
            for item in pending.values():
                item["reward"] -= 0.97 ** item["discount_power"] * reward
                item["discount_power"] += 1

        agent.train_batch(replay, batch_size=batch_size)
        if result.done:
            push_pending(0, done=True)
            agent.train_batch(replay, batch_size=batch_size)
            return


def imitation_pretrain(agent: DQNAgent, teacher, samples: int, epochs: int, seed: int) -> None:
    torch = agent.torch
    states, actions, legal_actions = [], [], []
    opponent = RandomAgent(seed=seed + 99)
    game = 0
    while len(states) < samples:
        env = YutEnv(seed=seed + game)
        env.reset()
        while len(states) < samples:
            if env.current_player == 0:
                legal = env.legal_actions()
                action = teacher.act(env)
                states.append(env.observe_for(0))
                actions.append(action)
                legal_actions.append(legal)
                result = env.step(action)
            else:
                result = env.step(opponent.act(env))
            if result.done:
                break
        game += 1

    x = torch.tensor(states, dtype=torch.float32)
    y = torch.tensor(actions, dtype=torch.long)
    for _ in range(epochs):
        logits = agent.model(x)
        mask = torch.full_like(logits, -1e9)
        for row, legal in enumerate(legal_actions):
            mask[row, legal] = 0
        loss = torch.nn.functional.cross_entropy(logits + mask, y)
        agent.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.model.parameters(), agent.grad_clip)
        agent.optim.step()
    agent.sync_target()


def train_dqn_variant(name: str, state_dim: int, teacher, episodes: int, seed: int, curriculum: bool, imitation_samples: int) -> DQNAgent:
    agent = DQNAgent(state_dim=state_dim, lr=5e-4, gamma=0.97, seed=seed)
    if imitation_samples:
        imitation_pretrain(agent, teacher, imitation_samples, epochs=3, seed=seed + 10_000)
    replay = ReplayBuffer(capacity=60_000)
    opponents = [RandomAgent(seed=seed + 1), StrategicRuleBasedAgent(seed=seed + 2), teacher]
    for episode in range(1, episodes + 1):
        if curriculum:
            phase = min(2, (episode - 1) * 3 // max(1, episodes))
            opponent = opponents[phase]
        else:
            opponent = opponents[1]
        epsilon = max(0.04, 0.75 * (0.992 ** episode))
        play_dqn_training_game(agent, opponent, replay, epsilon, batch_size=64, seed=seed + episode)
        if episode % 75 == 0:
            agent.sync_target()
    agent.sync_target()
    return agent


def plot_results(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    df.plot(kind="bar", x="agent", y="overall_win_rate", ax=ax, legend=False, color="#4C78A8")
    ax.set_title("Final tournament win rate")
    ax.set_xlabel("")
    ax.set_ylabel("win rate")
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--search-games", type=int, default=120)
    parser.add_argument("--contribution-games", type=int, default=30)
    parser.add_argument("--dqn-episodes", type=int, default=300)
    parser.add_argument("--imitation-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=440_000)
    parser.add_argument("--value-model", type=str, default="checkpoints/value.pt")
    parser.add_argument("--out-dir", type=str, default="results/agent_improvement")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dim = get_state_dim(args.seed)

    base_value = StrategicValueNetworkAgent(state_dim=dim, seed=args.seed)
    load_if_possible(base_value, args.value_model)
    tuned_name, tuned_value, tuned_params, search_df = tune_strategic_value(dim, args.value_model, args.search_games, args.seed, out_dir)
    log_strategic_value_contributions(tuned_value, args.contribution_games, args.seed + 60_000, out_dir)
    dqn_diagnostics(out_dir)

    dqn_before = DQNAgent(state_dim=dim, seed=args.seed + 100)
    dqn_improved = train_dqn_variant("dqn_improved", dim, tuned_value, args.dqn_episodes, args.seed + 200, curriculum=False, imitation_samples=0)
    dqn_curriculum = train_dqn_variant("dqn_curriculum", dim, tuned_value, args.dqn_episodes, args.seed + 300, curriculum=True, imitation_samples=args.imitation_samples)

    before_after = compare_agents(
        [
            NamedAgent("dqn_before", dqn_before),
            NamedAgent("dqn_improved", dqn_improved),
            NamedAgent("dqn_curriculum", dqn_curriculum),
            NamedAgent("strategic_value_teacher", tuned_value),
        ],
        args.games,
        args.seed + 90_000,
    )
    before_after.to_csv(out_dir / "dqn_before_after.csv", index=False)

    final = compare_agents(
        [
            NamedAgent("strategic_value", base_value),
            NamedAgent("strategic_value_tuned", tuned_value),
            NamedAgent("dqn_improved", dqn_improved),
            NamedAgent("dqn_curriculum", dqn_curriculum),
        ],
        args.games,
        args.seed + 120_000,
    )
    final.to_csv(out_dir / "final_tournament.csv", index=False)
    plot_results(final, out_dir / "final_tournament.png")

    best = final.iloc[0].to_dict()
    pd.DataFrame(
        [
            {
                "best_agent": best["agent"],
                "best_win_rate": best["overall_win_rate"],
                "best_params": repr(tuned_params) if best["agent"] == "strategic_value_tuned" else "",
                "strategic_value_tuned_name": tuned_name,
            }
        ]
    ).to_csv(out_dir / "best_agent.csv", index=False)

    print("[strategic_value search]")
    print(search_df.to_string(index=False))
    print()
    print("[DQN before/after]")
    print(before_after.to_string(index=False))
    print()
    print("[final tournament]")
    print(final.to_string(index=False))
    print()
    print("[best]")
    print(best)
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()
