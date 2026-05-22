import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
from config import load_config
from dqn_env import DQNAgent, ensure_dir, log_message
from genetic_algorithm import GeneticOptimizer


def plot_comparison(run_dirs: List[str], labels: List[str], output_path: str) -> None:
    plt.figure(figsize=(12, 4))
    for run_dir, label in zip(run_dirs, labels):
        metrics_path = os.path.join(run_dir, "episode_metrics.csv")
        if not os.path.exists(metrics_path):
            continue
        episodes = []
        rewards = []
        with open(metrics_path, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.strip().split(",")
                episodes.append(int(parts[0]))
                rewards.append(float(parts[1]))
        plt.plot(episodes, rewards, label=label)
    plt.title("Reward Comparison")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def build_hyperparams(config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    train_cfg = config["train"]
    params = {
        "batch_size": train_cfg["batch_size"],
        "layers": train_cfg["layers"],
        "activation": train_cfg["activation"],
        "learning_rate": train_cfg["learning_rate"],
    }
    params.update(overrides)
    return params


def main(config: Dict[str, Any] = None) -> None:
    config = config or load_config("config/default.yaml")
    output_dir = config["logging"]["output_dir"]
    ensure_dir(output_dir)
    log_path = os.path.join(output_dir, "execution.log")
    log_message(log_path, "Start experiment run")

    run_dirs: List[str] = []
    labels: List[str] = []

    for baseline in config["baselines"]:
        method = baseline["name"]
        run_dir = os.path.join(output_dir, method)
        ensure_dir(run_dir)
        hyperparams = build_hyperparams(config, baseline)
        log_message(log_path, f"Start baseline: {method}")
        agent = DQNAgent(config, run_dir, method, hyperparams)
        agent.train()
        log_message(log_path, f"End baseline: {method}")
        run_dirs.append(run_dir)
        labels.append(method)

    ga_optimizer = GeneticOptimizer(config, output_dir)
    log_message(log_path, "Start GA optimization")
    ga_params, ga_fitness = ga_optimizer.optimize()
    log_message(log_path, "End GA optimization")
    ga_run_dir = os.path.join(output_dir, "ga_best")
    log_message(log_path, "Start GA best training")
    ga_agent = DQNAgent(config, ga_run_dir, "ga", ga_params)
    ga_agent.train()
    log_message(log_path, "End GA best training")
    run_dirs.append(ga_run_dir)
    labels.append("ga")
    
    comparison_path = os.path.join(output_dir, "comparison.png")
    plot_comparison(run_dirs, labels, comparison_path)

    print(f"GA best fitness: {ga_fitness}")
    log_message(log_path, "End experiment run")


if __name__ == "__main__":
    main()