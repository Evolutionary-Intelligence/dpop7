import time

import ray  # engine for distributed computing
import numpy as np  # engine for numerical computing
# Standard Particle Swarm Optimizer with a global topology (a serial implementation)
from pypop7.optimizers.pso.spso import SPSO

# abstract class of all distributed optimizers
from dpop7.optimizers.core.distributed_optimizer import DistributedOptimizer as DO


class DSPSO(DO):
    """Distributed Standard Particle Swarm Optimizer with a global topology (DSPSO)
        based on the island model.

    Parameters
    ----------
    problem : dict
              problem arguments with the following common settings (`keys`):
                * 'fitness_function' - objective/cost function to be **minimized** (`func`),
                * 'ndim_problem'     - number of dimensionality (`int`),
                * 'upper_boundary'   - upper boundary of search range (`array_like`),
                * 'lower_boundary'   - lower boundary of search range (`array_like`).
    options : dict
              optimizer options with the following common settings (`keys`):
                * 'max_function_evaluations' - maximum of function evaluations (`int`, default: `np.Inf`),
                * 'max_runtime'              - maximal runtime to be allowed (`float`, default: `np.Inf`),
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`).

    Examples
    --------
    Use the optimizer `DSPSO` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import ray  # engine for distributed computing
       >>> import numpy  # engine for numerical computing
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from dpop7.optimizers.dpso.dspso import DSPSO
       >>> @ray.remote
       ... def f(x):  # for parallel function evaluations
       ...     return rosenbrock(x)
       >>> problem = {'fitness_function': f,  # define problem arguments
       ...            'ndim_problem': 20,
       ...            'lower_boundary': -5.0*numpy.ones((20,)),
       ...            'upper_boundary': 5.0*numpy.ones((20,))}
       >>> options = {'max_runtime': 60*10,  # set optimizer options
       ...            'seed_rng': 2023,  # seed for random number generation
       ...            'n_islands': 4}  # number of parallel islands
       >>> dspso = DSPSO(problem, options)  # initialize the optimizer class
       >>> results = dspso.optimize()  # run the parallel optimization process
       >>> print(f"DSPSO: {results['n_function_evaluations']}, {results['best_so_far_y']}")
    """
    def __init__(self, problem, options):
        """Initialize the class with two inputs (problem arguments and optimizer options)."""
        DO.__init__(self, problem, options)
        self.n_islands = options.get('n_islands')  # number of parallel islands
        assert self.n_islands > 1, 'Please use *PyPop7* directly (without any parallelism costs).'
        # set maximal runtime (at the inner level) of each island at each round (at the outer level)
        self.island_min_rt = options.get('island_min_rt', 3)  # minimal runtime of each island (for stability)
        assert self.island_min_rt >= 0
        self.island_rt = options.get('island_rt', 60*3)  # maximal runtime of each island
        assert self.island_rt >= self.island_min_rt
        self.island_sf = options.get('island_sf', 100)
        assert self.island_sf >= 0
        self.island_max_fe = options.get('island_max_fe', np.Inf)
        assert self.island_max_fe > 0
        self._optimizer = SPSO  # class of Particle Swarm Optimizer as the base of each island

    def optimize(self, fitness_function=None, args=None):  # for all iterations (generations)
        """For the entire optimization/evolution stage: initialization + iteration."""
        fitness = DO.optimize(self, fitness_function)
        ray_problem = ray.put(self.problem)  # to be shared for all islands across all nodes
        ray_base_optimizer = ray.remote(num_cpus=1)(self._optimizer)  # to be shared across all nodes
        options = [None]*self.n_islands  # optimizer options for each island
        while not self._check_terminations():
            ray_optimizers, ray_results = [], []  # to store all optimizers and their optimization results
            for i in range(self.n_islands):  # to run each island in parallel (driven by engine of ray)
                max_runtime = min(self.max_runtime - (time.time() - self.start_time), self.island_rt)
                options[i] = {'max_runtime': max(max_runtime, self.island_min_rt),
                    'max_function_evaluations': self.island_max_fe,
                    'fitness_threshold': self.fitness_threshold,
                    'seed_rng': self.rng_optimization.integers(0, np.iinfo(np.int64).max),
                    'verbose': False,
                    'saving_fitness': self.island_sf}
                ray_optimizers.append(ray_base_optimizer.remote(ray_problem, options[i]))
                # run each optimizer *serially* inside each *parallel* island
                ray_results.append(ray_optimizers[i].optimize.remote(self.fitness_function, args))
            results = ray.get(ray_results)  # to synchronize (a time-consuming operation)
            for r in results:  # to run serially (clearly which should be light-weighted)
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
