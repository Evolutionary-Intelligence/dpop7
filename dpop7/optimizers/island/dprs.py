import time

import ray  # engine for distributed computing
import numpy as np  # engine for numerical computing

# abstract class of all distributed optimizers
from dpop7.optimizers.core.distributed_optimizer import DistributedOptimizer as DO
from pypop7.optimizers.rs.prs import PRS  # Pure Random Search


class DPRS(DO):
    """Distributed Pure Random Search (DPRS) based on the island model.
    """
    def __init__(self, problem, options):
        """Initialize the class with two inputs (problem arguments and optimizer options)."""
        DO.__init__(self, problem, options)
        self.n_islands = options.get('n_islands')  # number of islands
        assert self.n_islands > 1
        self.island_runtime = options.get('island_runtime')
        assert self.island_runtime > 0
        self.island_saving_fitness = options.get('island_saving_fitness', 100)
        assert self.island_saving_fitness > 0
        self._optimizer = PRS

    def optimize(self, fitness_function=None, args=None):  # for all iterations (generations)
        """For the entire optimization/evolution stage: initialization + iteration."""
        super(DO, self).optimize(fitness_function)
        fitness = []  # to store all fitness generated during search
        ray_problem = ray.put(self.problem)  # to be shared across all nodes
        ray_opt = ray.remote(num_cpus=1)(self._optimizer)  # to be shared across all nodes
        options = [None]*self.n_islands  # for each island
        while not self._check_terminations():
            ray_optimizer, ray_results = [], []
            for i in range(self.n_islands):  # to run each island in parallel (driven by engine of ray)
                options[i] = {'max_runtime': self.island_runtime,
                    'fitness_threshold': self.fitness_threshold,
                    'seed_rng': self.rng_optimization.integers(0, np.iinfo(np.int64).max),
                    'verbose': False,
                    'saving_fitness': self.island_saving_fitness}
                ray_optimizer.append(ray_opt.remote(ray_problem, options[i]))
                ray_results.append(ray_optimizer[i].optimize.remote(self.fitness_function, args))
            results = ray.get(ray_results)  # to synchronize (a time-consuming operation)
            for i, r in enumerate(results):  # to run serially (clearly which should be light-weighted)
                if self.best_so_far_y > r['best_so_far_y']:  # to update best-so-far solution and fitness
                    self.best_so_far_x, self.best_so_far_y = r['best_so_far_x'], r['best_so_far_y']
                fitness_start, fitness_end = np.copy(r['fitness'][0]), np.copy(r['fitness'][-1])
                fitness_start[0] += self.n_function_evaluations
                fitness_end[0] += self.n_function_evaluations
                self.n_function_evaluations += r['n_function_evaluations']
                self.time_function_evaluations += r['time_function_evaluations']
                fitness.extend([fitness_start, fitness_end])
        return self._collect(fitness)

    def _collect(self, fitness=None):
        """Collect final optimization states."""
        return {'best_so_far_x': self.best_so_far_x,
            'best_so_far_y': self.best_so_far_y,
            'n_function_evaluations': self.n_function_evaluations,
            'runtime': time.time() - self.start_time,
            'termination_signal': self.termination_signal,
            'time_function_evaluations': self.time_function_evaluations,
            'fitness': np.array(fitness)}
