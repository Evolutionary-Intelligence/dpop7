import time
import numpy as np  # engine for numerical computing

# base class for Distributed Differential Evolution and Ray-based parallel fitness evaluations
from dpop7.optimizers.dde.dde import DDE
from dpop7.optimizers.core.distributed_optimizer import parallelize_evaluations as p_e


class DCDE(DDE):
    """Distributed Classic Differential Evolution (DCDE).

    Parameters
    ----------
    problem : `dict`
              problem arguments with the following common settings (`keys`):
                * 'fitness_function' - objective/cost function to be **minimized** (`func`),
                * 'ndim_problem'     - number of dimensionality (`int`),
                * 'upper_boundary'   - upper boundary of search range (`array_like`),
                * 'lower_boundary'   - lower boundary of search range (`array_like`).
    options : `dict`
              optimizer options with the following common settings (`keys`):
                * 'n_individuals'            - number of parallel offspring (`int`, default: `100`),
                * 'max_function_evaluations' - maximum of function evaluations (`int`, default: `np.Inf`),
                * 'max_runtime'              - maximal runtime to be allowed (`float`, default: `np.Inf`),
                * 'seed_rng'                 - seed for random number generation (RNG) needed to be *explicitly* set (`int`);
              and with the following particular settings (`keys`):
                * 'f'  - mutation factor (`float`, default: `0.5`),
                * 'cr' - crossover probability (`float`, default: `0.9`).

    Examples
    --------
    Use the optimizer `DCDE` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

    Attributes
    ----------
    cr            : `float`
                    crossover probability.
    f             : `float`
                    mutation factor.
    n_individuals : `int`
                    number of parallel offspring, aka offspring population size. For `DE`, typically a *large*
                    (often >=100) population size is used to better explore for multimodal functions. Obviously
                    the *optimal* population size is problem-dependent, which can be fine-tuned in practice.

    References
    ----------
    Price, K.V., 2013.
    Differential evolution.
    In Handbook of optimization (pp. 187-214). Springer, Berlin, Heidelberg.
    https://link.springer.com/chapter/10.1007/978-3-642-30504-7_8

    Price, K.V., Storn, R.M. and Lampinen, J.A., 2005.
    Differential evolution: A practical approach to global optimization.
    Springer Science & Business Media.
    https://link.springer.com/book/10.1007/3-540-31306-0

    Storn, R.M. and Price, K.V. 1997.
    Differential evolution – a simple and efficient heuristic for global optimization over continuous spaces.
    Journal of Global Optimization, 11(4), pp.341–359.
    https://link.springer.com/article/10.1023/A:1008202821328
    (Kenneth Price&Rainer Storn won the **2017** `Evolutionary Computation Pioneer Award from IEEE CIS
    <https://cis.ieee.org/awards/past-recipients#EvolutionaryComputationPioneerAward>`_.)
    """
    def __init__(self, problem, options):
        """Initialize the class with two inputs (problem arguments and optimizer options)."""
        DDE.__init__(self, problem, options)
        self.f = options.get('f', 0.5)  # mutation factor
        assert 0.0 <= self.f <= 1.0
        self.cr = options.get('cr', 0.9)  # crossover probability
        assert 0.0 <= self.cr <= 1.0

    def initialize(self, args=None):
        """Only for the initialization stage."""
        x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary,
                                            size=(self.n_individuals, self.ndim_problem))  # population
        self.start_function_evaluations = time.time()
        y = p_e(self.fitness_function, x, args)  # to evaluate these parallel points
        self.time_function_evaluations += time.time() - self.start_function_evaluations
        self.n_function_evaluations += len(y)
        v = np.empty((self.n_individuals, self.ndim_problem))  # for mutation
        # update best-so-far solution and fitness
        i = np.argmin(y)
        if y[i] < self.best_so_far_y:
            self.best_so_far_x, self.best_so_far_y = np.copy(x[i]), y[i]
        return x, y, v

    def mutate(self, x=None, v=None):
        for i in range(self.n_individuals):
            r = self.rng_optimization.permutation(self.n_individuals)[:4]
            r = r[r != i][:3]  # a simple yet effective trick
            v[i] = x[r[0]] + self.f*(x[r[1]] - x[r[2]])
        return v

    def crossover(self, v=None, x=None):
        """Binomial crossover (uniform discrete crossover)."""
        for i in range(self.n_individuals):
            j_r = self.rng_optimization.integers(self.ndim_problem)
            # to avoid loop (a simple yet effective trick)
            tmp = v[i, j_r]
            co = self.rng_optimization.random(self.ndim_problem) > self.cr
            v[i, co] = x[i, co]
            v[i, j_r] = tmp
        return v

    def select(self, v=None, x=None, y=None, args=None):
        self.start_function_evaluations = time.time()
        yy = p_e(self.fitness_function, v, args)  # to evaluate these parallel points
        self.time_function_evaluations += time.time() - self.start_function_evaluations
        self.n_function_evaluations += len(y)
        for i in range(self.n_individuals):
            if yy[i] < y[i]:
                x[i], y[i] = v[i], yy[i]
        # update best-so-far solution and fitness
        i = np.argmin(y)
        if yy[i] < self.best_so_far_y:
            self.best_so_far_x, self.best_so_far_y = np.copy(x[i]), yy[i]
        return x, y

    def iterate(self, x=None, y=None, v=None, args=None):
        """Only for the iteration stage."""
        v = self.mutate(x, v)
        v = self.crossover(v, x)
        x, y = self.select(v, x, y, args)
        self._n_generations += 1
        return x, y

    def optimize(self, fitness_function=None, args=None):
        """For the entire optimization/evolution stage: initialization + iteration."""
        fitness = DDE.optimize(self, fitness_function)
        x, y, v = self.initialize(args)
        while not self._check_terminations():
            self._print_verbose_info(fitness, y)
            x, y = self.iterate(x, y, v, args)
        return self._collect(fitness, y)
