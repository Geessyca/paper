import json
import os
import time
from typing import Any, Dict, List, Tuple

from skopt import gp_minimize
from skopt.space import Categorical, Integer, Real

from hyperparameter_optimizer import HyperparameterOptimizer
from dqn_env import DQNAgent, ensure_dir


class BayesianOptimizer(HyperparameterOptimizer):
    def __init__(self, config: Dict[str, Any], output_dir: str):
        super().__init__(config, output_dir)

    def optimize(self) -> Tuple[Dict[str, Any], float]:
        search_space = self.config["search_space"]
        layer_cfg = search_space["layers"]
        lr_cfg = search_space["learning_rate"]
        space = [
            Categorical(search_space["batch_size"], name="batch_size"),
            Integer(layer_cfg["min_layers"], layer_cfg["max_layers"], name="layer_count"),
            Integer(layer_cfg["min_units"], layer_cfg["max_units"], name="layer_1"),
            Integer(layer_cfg["min_units"], layer_cfg["max_units"], name="layer_2"),
            Integer(layer_cfg["min_units"], layer_cfg["max_units"], name="layer_3"),
            Integer(layer_cfg["min_units"], layer_cfg["max_units"], name="layer_4"),
            Integer(layer_cfg["min_units"], layer_cfg["max_units"], name="layer_5"),
            Categorical(search_space["activation"], name="activation"),
            Real(lr_cfg["min"], lr_cfg["max"], prior="log-uniform", name="learning_rate"),
        ]

        evaluation_episodes = self.config["search"]["evaluation_episodes"]
        bayes_cfg = self.config["search"]["bayesian"]
        results: List[Dict[str, Any]] = []
        run_root = os.path.join(self.output_dir, "bayesian")
        ensure_dir(run_root)

        def objective(values):
            params = {
                "batch_size": values[0],
                "layers": values[2 : 2 + int(values[1])],
                "activation": values[7],
                "learning_rate": values[8],
            }
            run_dir = os.path.join(run_root, f"trial_{len(results):03d}")
            agent = DQNAgent(self.config, run_dir, "bayesian_trial", params)
            summary = agent.train(num_episodes=evaluation_episodes)
            fitness = summary["fitness"]
            results.append({"params": params, "fitness": fitness, "summary": summary})
            return -fitness

        start = time.time()
        res = gp_minimize(
            objective,
            space,
            n_calls=bayes_cfg["calls"],
            n_initial_points=bayes_cfg["random_starts"],
            acq_func=bayes_cfg["acq_func"],
            random_state=self.config["reproducibility"]["seed"],
        )
        total_time = time.time() - start

        best_params = {
            "batch_size": res.x[0],
            "layers": res.x[2 : 2 + int(res.x[1])],
            "activation": res.x[7],
            "learning_rate": res.x[8],
        }
        best_fitness = -res.fun

        summary_path = os.path.join(run_root, "bayesian_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_params": best_params,
                    "best_fitness": best_fitness,
                    "total_search_time_sec": total_time,
                    "trials": results,
                },
                f,
                indent=2,
            )

        return best_params, best_fitness
