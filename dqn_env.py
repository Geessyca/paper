import json
import math
import os
import random
import time
from collections import deque, namedtuple
from datetime import datetime
from typing import Any, Dict, List, Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from dqn import DQNetwork

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def log_message(log_path: str, message: str) -> None:
    timestamp = datetime.utcnow().isoformat()
    line = f"[{timestamp}] {message}"
    print(line)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def set_global_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def moving_average(values: List[float], window: int) -> float:
    if not values:
        return 0.0
    if len(values) < window:
        return float(np.mean(values))
    return float(np.mean(values[-window:]))


class EpsilonGreedyPolicy:
    def __init__(self, start: float, end: float, decay: float):
        self.start = start
        self.end = end
        self.decay = decay
        self.steps_done = 0

    def select_action(self, state, policy_net, n_actions: int, device) -> Dict[str, Any]:
        sample = random.random()
        eps_threshold = self.end + (self.start - self.end) * math.exp(
            -1.0 * self.steps_done / self.decay
        )
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                action = policy_net(state).max(1)[1].view(1, 1)
        else:
            action = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
        return {"action": action, "epsilon": eps_threshold}


class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args) -> None:
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class OptimizeModel:
    def main(self, policy_net, target_net, memory, optimizer, batch_size, gamma, device) -> Optional[float]:
        if len(memory) < batch_size:
            return None
        transitions = memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool
        )
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(batch_size, device=device)
        next_state_values[non_final_mask] = (
            target_net(non_final_next_states).max(1)[0].detach()
        )
        expected_state_action_values = (next_state_values * gamma) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        optimizer.step()
        return float(loss.item())


class DQNAgent:
    def __init__(self, config: Dict[str, Any], run_dir: str, method_name: str, hyperparams: Dict[str, Any]):
        self.config = config
        self.run_dir = run_dir
        self.method_name = method_name
        ensure_dir(run_dir)
        ensure_dir(os.path.join(run_dir, "plots"))
        self.log_path = os.path.join(run_dir, "execution.log")

        seed = config["reproducibility"]["seed"]
        deterministic = config["reproducibility"]["deterministic"]
        set_global_seed(seed, deterministic)

        env_cfg = config["env"]
        self.env = gym.make(
            env_cfg["id"],
            render_mode=env_cfg["render_mode"],
            max_episode_steps=env_cfg["max_steps"],
        )
        self.env.action_space.seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        n_actions = self.env.action_space.n
        state_dim = self.env.observation_space.shape[0]
        train_cfg = config["train"]

        self.batch_size = hyperparams["batch_size"]
        self.learning_rate = hyperparams["learning_rate"]
        self.layers = hyperparams["layers"]
        self.activation = hyperparams["activation"]

        self.policy = EpsilonGreedyPolicy(
            train_cfg["eps_start"], train_cfg["eps_end"], train_cfg["eps_decay"]
        )
        self.policy_net = DQNetwork(state_dim, n_actions, self.layers, self.activation).to(self.device)
        self.target_net = DQNetwork(state_dim, n_actions, self.layers, self.activation).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayMemory(train_cfg["replay_buffer_size"])

    def _plot_series(self, values: List[float], title: str, y_label: str, file_name: str) -> None:
        plt.figure(figsize=(12, 4))
        plt.plot(range(len(values)), values, linestyle="-", color="b")
        plt.title(title)
        plt.xlabel("Episode")
        plt.ylabel(y_label)
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, "plots", file_name))
        plt.close()

    def train(self, num_episodes: Optional[int] = None) -> Dict[str, Any]:
        train_cfg = self.config["train"]
        convergence_cfg = train_cfg["convergence"]
        num_episodes = num_episodes or train_cfg["num_episodes"]
        gamma = train_cfg["gamma"]
        target_update = train_cfg["target_update"]
        reward_clip = train_cfg["reward_clipping"]

        log_message(
            self.log_path,
            f"Start training: method={self.method_name} episodes={num_episodes}",
        )

        episode_metrics: List[Dict[str, Any]] = []
        rewards: List[float] = []
        losses: List[float] = []
        epsilons: List[float] = []
        episode_times: List[float] = []

        start_train = time.time()
        convergence_episode = None
        time_to_convergence = None

        for i_episode in range(num_episodes):
            ep_start = time.time()
            state, _ = self.env.reset(seed=self.config["reproducibility"]["seed"] + i_episode)
            state = torch.tensor([state], device=self.device, dtype=torch.float32)

            total_reward = 0.0
            total_loss = 0.0
            loss_count = 0
            steps = 0
            done = False

            while not done:
                policy_out = self.policy.select_action(
                    state, self.policy_net, self.env.action_space.n, self.device
                )
                action = policy_out["action"]
                eps_value = policy_out["epsilon"]

                next_state, reward, terminated, truncated, _ = self.env.step(action.item())
                done = terminated or truncated
                if reward_clip is not None:
                    reward = float(np.clip(reward, reward_clip[0], reward_clip[1]))
                total_reward += reward
                steps += 1

                reward_tensor = torch.tensor([reward], device=self.device, dtype=torch.float32)
                if not done:
                    next_state_tensor = torch.tensor([next_state], device=self.device, dtype=torch.float32)
                else:
                    next_state_tensor = None

                self.memory.push(state, action, next_state_tensor, reward_tensor)
                state = next_state_tensor

                loss = OptimizeModel().main(
                    self.policy_net,
                    self.target_net,
                    self.memory,
                    self.optimizer,
                    self.batch_size,
                    gamma,
                    self.device,
                )
                if loss is not None:
                    total_loss += loss
                    loss_count += 1

            avg_loss = total_loss / loss_count if loss_count > 0 else 0.0
            rewards.append(total_reward)
            losses.append(avg_loss)
            epsilons.append(eps_value)
            episode_time = time.time() - ep_start
            episode_times.append(episode_time)

            moving_avg = moving_average(rewards, convergence_cfg["window"])
            episode_metrics.append(
                {
                    "episode": i_episode,
                    "reward": total_reward,
                    "moving_avg": moving_avg,
                    "loss": avg_loss,
                    "epsilon": eps_value,
                    "episode_time_sec": episode_time,
                }
            )

            if i_episode % target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            if (
                convergence_episode is None
                and len(rewards) >= convergence_cfg["window"]
                and moving_avg >= convergence_cfg["threshold"]
            ):
                convergence_episode = i_episode
                time_to_convergence = time.time() - start_train

        total_time = time.time() - start_train
        avg_episode_time = float(np.mean(episode_times)) if episode_times else 0.0
        fitness = float(np.mean(rewards)) if rewards else 0.0

        metrics_path = os.path.join(self.run_dir, "episode_metrics.csv")
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write("episode,reward,moving_avg,loss,epsilon,episode_time_sec\n")
            for row in episode_metrics:
                f.write(
                    f"{row['episode']},{row['reward']},{row['moving_avg']},{row['loss']},{row['epsilon']},{row['episode_time_sec']}\n"
                )

        summary = {
            "method": self.method_name,
            "timestamp": datetime.utcnow().isoformat(),
            "seed": self.config["reproducibility"]["seed"],
            "hyperparameters": {
                "batch_size": self.batch_size,
                "layers": self.layers,
                "activation": self.activation,
                "learning_rate": self.learning_rate,
            },
            "metrics": {
                "avg_episode_time_sec": avg_episode_time,
                "total_training_time_sec": total_time,
                "time_to_convergence_sec": time_to_convergence,
                "episodes_to_convergence": convergence_episode,
                "fitness": fitness,
            },
        }
        summary_path = os.path.join(self.run_dir, "run_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        self._plot_series(rewards, "Reward vs Episode", "Reward", "reward.png")
        self._plot_series(
            [m["moving_avg"] for m in episode_metrics],
            "Moving Average Reward",
            "Moving Avg",
            "moving_avg.png",
        )
        self._plot_series(losses, "Loss vs Episode", "Loss", "loss.png")
        self._plot_series(epsilons, "Epsilon Decay", "Epsilon", "epsilon.png")
        self._plot_series(episode_times, "Episode Time", "Time (sec)", "episode_time.png")

        self.env.close()

        summary["fitness"] = fitness
        log_message(
            self.log_path,
            f"End training: method={self.method_name} fitness={fitness:.4f} total_time_sec={total_time:.2f}",
        )
        return summary