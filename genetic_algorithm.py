import json
import os
import random
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from deap import algorithms, base, creator, tools

from dqn_env import DQNAgent, ensure_dir
from hyperparameter_optimizer import HyperparameterOptimizer


class GeneticOptimizer(HyperparameterOptimizer):
    def __init__(self, config: Dict[str, Any], output_dir: str):
        super().__init__(config, output_dir)

    def optimize(self) -> Tuple[Dict[str, Any], float]:
        search_space = self.config["search_space"]
        ga_cfg = self.config["search"]["ga"]
        evaluation_episodes = self.config["search"]["evaluation_episodes"]
        run_root = os.path.join(self.output_dir, "ga")
        ensure_dir(run_root)

        if "FitnessMax" not in creator.__dict__:
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if "Individual" not in creator.__dict__:
            creator.create("Individual", dict, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        def random_layers():
            layer_cfg = search_space["layers"]
            count = random.randint(layer_cfg["min_layers"], layer_cfg["max_layers"])
            return [
                random.randint(layer_cfg["min_units"], layer_cfg["max_units"])
                for _ in range(count)
            ]

        def random_activation():
            return random.choice(search_space["activation"])

        def random_batch_size():
            return random.choice(search_space["batch_size"])

        def random_learning_rate():
            lr_cfg = search_space["learning_rate"]
            return random.uniform(lr_cfg["min"], lr_cfg["max"])

        def init_network(container):
            return container(
                layers=random_layers(),
                activation=random_activation(),
                batch_size=random_batch_size(),
                learning_rate=random_learning_rate(),
            )

        toolbox.register("individual", init_network, creator.Individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        population = toolbox.population(n=ga_cfg["population_size"])

        results: List[Dict[str, Any]] = []

        def fitness_calculate(individual):
            params = {
                "batch_size": individual["batch_size"],
                "layers": individual["layers"],
                "activation": individual["activation"],
                "learning_rate": individual["learning_rate"],
            }
            run_dir = os.path.join(run_root, f"trial_{len(results):03d}")
            agent = DQNAgent(self.config, run_dir, "ga_trial", params)
            summary = agent.train(num_episodes=evaluation_episodes)
            fitness = summary["fitness"]
            results.append({"params": params, "fitness": fitness, "summary": summary})
            return (fitness,)

        toolbox.register("evaluate", fitness_calculate)

        def mutate(individual):
            mutation_choice = random.randint(0, 3)
            if mutation_choice == 0:
                individual["layers"] = random_layers()
            elif mutation_choice == 1:
                individual["activation"] = random_activation()
            elif mutation_choice == 2:
                individual["batch_size"] = random_batch_size()
            else:
                individual["learning_rate"] = random_learning_rate()
            return (individual,)

        toolbox.register("mutate", mutate)

        def cx_individual(ind1, ind2):
            for key in ["layers", "activation", "batch_size", "learning_rate"]:
                if random.random() < 0.5:
                    ind1[key], ind2[key] = ind2[key], ind1[key]
            return ind1, ind2

        toolbox.register("mate", cx_individual)
        toolbox.register("select", tools.selRoulette)

        statistic = tools.Statistics(lambda ind: ind.fitness.values)
        statistic.register("fitness_max", np.max)
        statistic.register("fitness_min", np.min)
        hof = tools.HallOfFame(1)

        start = time.time()
        algorithms.eaMuPlusLambda(
            population,
            toolbox,
            mu=ga_cfg["population_size"],
            lambda_=int(ga_cfg["population_size"]),
            cxpb=ga_cfg["cxpb"],
            mutpb=ga_cfg["mutpb"],
            ngen=ga_cfg["generations"],
            stats=statistic,
            halloffame=hof,
            verbose=True,
        )
        total_time = time.time() - start

        best_params = dict(hof[0])
        best_fitness = hof[0].fitness.values[0]

        summary_path = os.path.join(run_root, "ga_summary.json")
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

