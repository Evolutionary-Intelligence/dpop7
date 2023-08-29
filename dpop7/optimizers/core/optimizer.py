from abc import ABC  # abstract base classes
import time  # for timing
from enum import IntEnum  # for termination conditions

import numpy as np  # engine for numerical computing


class Terminations(IntEnum):  # for termination conditions
    """Helper class used by all optimizer classes."""
    NO_TERMINATION = 0  # as a flag to no termination
    MAX_FUNCTION_EVALUATIONS = 1  # maximum of function evaluations
    MAX_RUNTIME = 2  # maximal runtime to be allowed
    FITNESS_THRESHOLD = 3  # threshold of fitness
    #  (when the best-so-far fitness is below it, the optimizer will stop)


class Optimizer(ABC):
    """Base (abstract) class of all optimizers for continuous black-box **minimization**.

    References
    ----------
    Kochenderfer, M.J. and Wheeler, T.A., 2019.
    Algorithms for optimization.
    MIT Press.
    https://algorithmsbook.com/optimization/
    (See Chapter 7: Direct Methods for details.)

    Nesterov, Y., 2018.
    Lectures on convex optimization.
    Berlin: Springer International Publishing.
    https://link.springer.com/book/10.1007/978-3-319-91578-4

    Nesterov, Y. and Spokoiny, V., 2017.
    Random gradient-free minimization of convex functions.
    Foundations of Computational Mathematics, 17(2), pp.527-566.
    https://link.springer.com/article/10.1007/s10208-015-9296-2

    Audet, C. and Hare, W., 2017.
    Derivative-free and blackbox optimization.
    Berlin: Springer International Publishing.
    https://link.springer.com/book/10.1007/978-3-319-68913-5
    """
    def __init__(self, problem, options):
        """Initialize the class with two inputs (problem arguments and optimizer options).

        Parameters
        ----------
        problem : dict
                  problem arguments with the following common settings (`keys`):
                    * 'fitness_function'       - objective function to be **minimized** (`func`),
                    * 'ndim_problem'           - number of dimensionality (`int`),
                    * 'upper_boundary'         - upper boundary of search range (`array_like`),
                    * 'lower_boundary'         - lower boundary of search range (`array_like`),
                    * 'initial_upper_boundary' - initial upper boundary of search range (`array_like`),
                    * 'initial_lower_boundary' - initial lower boundary of search range (`array_like`),
                    * 'problem_name'           - problem name mainly for debugging purpose (`str`).
        options : dict
                  optimizer options with the following common settings (`keys`):
                    * 'max_function_evaluations' - maximum of function evaluations (`int`, default: `np.Inf`),
                    * 'max_runtime'              - maximal runtime to be allowed (`float`, default: `np.Inf`),
                    * 'seed_rng'                 - seed for random number generation (RNG) to be *explicitly* set (`int`),
                    * 'fitness_threshold'        - fitness threshold to stop the optimization process;
                  and with the following particular settings (`key`) [dependent on different optimizers]:
                    * 'x'                   - initial (starting) point (`array_like`),
                    * 'n_individuals'       - offspring population size (`int`),
                    * 'n_parents'           - parent population size (`int`),
                    * 'seed_initialization' - RNG seed only for the initialization process (`int`),
                    * 'seed_optimization'   - RNG seed only for the optimization process (`int`),
                    * 'saving_fitness'      - flag/frequency to save generated fitnesses (`int`),
                    * 'verbose'             - flag/frequency to print verbose information (`int`).
        """
        # problem-related settings (in `dict` form)
        self.fitness_function = problem.get('fitness_function')
        self.ndim_problem = problem['ndim_problem']
        self.upper_boundary = problem.get('upper_boundary')
        self.lower_boundary = problem.get('lower_boundary')
        self.initial_upper_boundary = problem.get('initial_upper_boundary', self.upper_boundary)
        self.initial_lower_boundary = problem.get('initial_lower_boundary', self.lower_boundary)
        self.problem_name = problem.get('problem_name')
        if (self.problem_name is None) and hasattr(self.fitness_function, '__name__'):
            self.problem_name = self.fitness_function.__name__

        # optimizer-related options (in `dict` form)
        self.max_function_evaluations = options.get('max_function_evaluations', np.Inf)
        self.max_runtime = options.get('max_runtime', np.Inf)
        self.fitness_threshold = options.get('fitness_threshold', -np.Inf)
        self.n_individuals = options.get('n_individuals')  # offspring population size
        self.n_parents = options.get('n_parents')  # parent population size
        self.seed_rng = options.get('seed_rng')
        if self.seed_rng is None:  # it is highly recommended to explicitly set *seed_rng*
            self.rng = np.random.default_rng()  # NOT use it, if possible
        else:
            self.rng = np.random.default_rng(self.seed_rng)
        self.seed_initialization = options.get('seed_initialization', self.rng.integers(np.iinfo(np.int64).max))
        self.rng_initialization = np.random.default_rng(self.seed_initialization)  # RNG only for the initialization process
        self.seed_optimization = options.get('seed_optimization', self.rng.integers(np.iinfo(np.int64).max))
        self.rng_optimization = np.random.default_rng(self.seed_optimization)  # RNG only for the optimization process
        self.saving_fitness = options.get('saving_fitness', 0)
        self.verbose = options.get('verbose', 10)
        self.is_restart = options.get('is_restart', True)

        # auxiliary members (mainly for state recordings)
        self.Terminations = Terminations
        self.n_function_evaluations = options.get('n_function_evaluations', 0)
        self.start_function_evaluations = None
        self.time_function_evaluations = options.get('time_function_evaluations', 0)
        self.runtime = options.get('runtime', 0)
        self.start_time = None
        self.best_so_far_y = options.get('best_so_far_y', np.Inf)
        self.best_so_far_x = None
        self.termination_signal = 0  # NO_TERMINATION
        self.fitness = None  # to save generated fitnesses

    def _evaluate_fitness(self, x, args=None):
        """Only for serial fitness evaluation."""
        self.start_function_evaluations = time.time()
        if args is None:
            y = self.fitness_function(x)
        else:
            y = self.fitness_function(x, args=args)
        self.time_function_evaluations += time.time() - self.start_function_evaluations
        self.n_function_evaluations += 1
        # update best-so-far solution (`x`) and fitness (`y`)
        if y < self.best_so_far_y:
            self.best_so_far_x, self.best_so_far_y = np.copy(x), y
        return float(y)  # for simplicity

    def _check_terminations(self):
        """Check termination conditions."""
        self.runtime = time.time() - self.start_time  # actually used runtime
        if self.n_function_evaluations >= self.max_function_evaluations:
            termination_signal = True, Terminations.MAX_FUNCTION_EVALUATIONS
        elif self.runtime >= self.max_runtime:
            termination_signal = True, Terminations.MAX_RUNTIME
        elif self.best_so_far_y <= self.fitness_threshold:
            termination_signal = True, Terminations.FITNESS_THRESHOLD
        else:
            termination_signal = False, Terminations.NO_TERMINATION
        self.termination_signal = termination_signal[1]
        return termination_signal[0]

    def _compress_fitness(self, fitness):
        """Compress fitness data in a predefined frequency `saving_fitness`."""
        fitness = np.array(fitness)
        for i in range(len(fitness) - 1):  # arrange in non-increasing order
            if fitness[i] < fitness[i + 1]:
                fitness[i + 1] = fitness[i]
        if self.saving_fitness == 1:  # to save all fitnesses generated during evolution
            self.fitness = np.stack((np.arange(len(fitness)) + 1, fitness), 1)
        elif self.saving_fitness > 1:  # to first sample then save
            # use 1-based index
            index = np.arange(1, len(fitness), self.saving_fitness)
            # recover 0-based index via - 1
            index = np.append(index, len(fitness)) - 1
            self.fitness = np.stack((index, fitness[index]), 1)
            # recover 1-based index for convergence data
            self.fitness[0, 0], self.fitness[-1, 0] = 1, len(fitness)

    def _check_success(self):
        """Check the final state (success/True or failure/False) of the best-so-far
            solution/fitness according to two conditions:
            1. Whether the best-so-far solution is out of the search range or not,
            2. Whether the best-so-far solution or fitness includes NaN or not.
        """
        if (self.upper_boundary is not None) and (self.lower_boundary is not None) and (
                np.any(self.lower_boundary > self.best_so_far_x) or
                np.any(self.best_so_far_x > self.upper_boundary)):
            return False
        elif np.isnan(self.best_so_far_y) or np.any(np.isnan(self.best_so_far_x)):
            return False
        return True

    def _collect(self, fitness):
        """Collect final states shared by all optimizer classes."""
        if self.saving_fitness:  # to reduce fitness data for faster postprocessing
            self._compress_fitness(fitness[:self.n_function_evaluations])
        return {'best_so_far_x': self.best_so_far_x,  # best-so-far solution found
                'best_so_far_y': self.best_so_far_y,  # best-so-far fitness found
                'n_function_evaluations': self.n_function_evaluations,
                'runtime': time.time() - self.start_time,  # actually used runtime
                'termination_signal': self.termination_signal,
                'time_function_evaluations': self.time_function_evaluations,
                'fitness': self.fitness,  # sampled fitnesses
                'success': self._check_success()}  # flag to show the success state

    def initialize(self):
        """Only for the initialization stage."""
        raise NotImplementedError

    def iterate(self):
        """Only for the iteration stage."""
        raise NotImplementedError

    def optimize(self, fitness_function=None):
        """For the entire optimization/evolution stage: initialization + iteration."""
        self.start_time = time.time()
        if fitness_function is not None:
            self.fitness_function = fitness_function
        fitness = []  # to store fitness generated during evolution/optimization
        return fitness
