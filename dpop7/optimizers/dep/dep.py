import numpy as np  # engine for numerical computing

# abstract class of all distributed optimizers
from dpop7.optimizers.core.distributed_optimizer import DistributedOptimizer as DO


class DEP(DO):
    """Distributed Evolutionary Programming (DEP).

    This is the **base** class for all `DEP` classes. Please use any of its instantiated
    subclasses to optimize the black-box problem at hand on CPU-based distributed computing
    platforms.

    .. note:: `EP` is one of three classical families of evolutionary algorithms (EAs),
       proposed originally by Lawrence J. Fogel, the recipient of `IEEE Evolutionary
       Computation Pioneer Award 1998 <https://tinyurl.com/456as566>`_ and
       `IEEE Frank Rosenblatt Award 2006 <https://tinyurl.com/yj28zxfa>`_. When used for
       continuous optimization, most of modern `EP` versions share much similarities
       (e.g. self-adaptation) with `ES <https://pypop.readthedocs.io/en/latest/es/es.html>`_,
       another representative EA.

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
                * 'n_individuals'            - number of parallel offspring (`int`, default: `100`),
                * 'max_function_evaluations' - maximum of function evaluations (`int`, default: `np.Inf`),
                * 'max_runtime'              - maximal runtime to be allowed (`float`, default: `np.Inf`),
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`);
              and with the following particular settings (`keys`):
                * 'sigma' - initial global step-size, aka mutation strength (`float`).

    Attributes
    ----------
    n_individuals : `int`
                    number of parallel offspring, aka offspring population size.
    sigma         : `float`
                    initial global step-size, aka mutation strength.

    Methods
    -------

    References
    ----------
    Lee, C.Y. and Yao, X., 2004.
    Evolutionary programming using mutations based on the Lévy probability distribution.
    IEEE Transactions on Evolutionary Computation, 8(1), pp.1-13.
    https://ieeexplore.ieee.org/document/1266370

    Yao, X., Liu, Y. and Lin, G., 1999.
    Evolutionary programming made faster.
    IEEE Transactions on Evolutionary Computation, 3(2), pp.82-102.
    https://ieeexplore.ieee.org/abstract/document/771163
    (Xin Yao won the **2013** `Evolutionary Computation Pioneer Award from IEEE CIS
    <https://cis.ieee.org/awards/past-recipients#EvolutionaryComputationPioneerAward>`_.)

    Fogel, D.B., 1999.
    An overview of evolutionary programming.
    In Evolutionary Algorithms (pp. 89-109). Springer, New York, NY.
    https://link.springer.com/chapter/10.1007/978-1-4612-1542-4_5

    Fogel, D.B. and Fogel, L.J., 1995, September.
    An introduction to evolutionary programming.
    In European Conference on Artificial Evolution (pp. 21-33). Springer, Berlin, Heidelberg.
    https://link.springer.com/chapter/10.1007/3-540-61108-8_28

    Fogel, D.B., 1994.
    An introduction to simulated evolutionary optimization.
    IEEE Transactions on Neural Networks, 5(1), pp.3-14.
    https://ieeexplore.ieee.org/abstract/document/265956

    Fogel, D.B., 1994.
    Evolutionary programming: An introduction and some current directions.
    Statistics and Computing, 4(2), pp.113-129.
    https://link.springer.com/article/10.1007/BF00175356
    (David B. Fogel won the **2008** `Evolutionary Computation Pioneer Award from IEEE CIS
    <https://cis.ieee.org/awards/past-recipients#EvolutionaryComputationPioneerAward>`_.)

    Bäck, T. and Schwefel, H.P., 1993.
    An overview of evolutionary algorithms for parameter optimization.
    Evolutionary Computation, 1(1), pp.1-23.
    https://direct.mit.edu/evco/article-abstract/1/1/1/1092/An-Overview-of-Evolutionary-Algorithms-for
    (Thomas Bäck/Hans-Paul Schwefel won the **2008**/**2002** `Evolutionary Computation Pioneer Award
    from IEEE CIS <https://cis.ieee.org/awards/past-recipients#EvolutionaryComputationPioneerAward>`_.)
    """
    def __init__(self, problem, options):
        """Initialize the class with two inputs (problem arguments and optimizer options)."""
        DO.__init__(self, problem, options)
        if self.n_individuals is None:
            self.n_individuals = 100  # number of offspring, aka offspring population size
        assert self.n_individuals > 0
        self.sigma = options.get('sigma')  # initial global step-size, aka mutation strength
        assert self.sigma > 0.0
        self._n_generations = 0  # number of generations
        self._printed_evaluations = 0  # only for printing

    def initialize(self):
        """Only for the initialization stage."""
        raise NotImplementedError

    def iterate(self):
        """Only for the iteration stage."""
        raise NotImplementedError

    def _print_verbose_info(self, fitness, y, is_print=False):
        """Print verbose information with a predefined frequency for logging."""
        if y is not None and self.saving_fitness:  # to save all fitnesses
            if not np.isscalar(y):
                fitness.extend(y)
            else:
                fitness.append(y)
        if self.verbose:
            is_verbose = self._printed_evaluations != self.n_function_evaluations  # to avoid repeated printing
            is_verbose_1 = (not self._n_generations % self.verbose) and is_verbose
            is_verbose_2 = self.termination_signal > 0 and is_verbose
            is_verbose_3 = is_print and is_verbose
            if is_verbose_1 or is_verbose_2 or is_verbose_3:
                info = '  * Generation {:d}: best_so_far_y {:7.5e}, min(y) {:7.5e} & Evaluations {:d}'
                print(info.format(self._n_generations, self.best_so_far_y, np.min(y), self.n_function_evaluations))
                self._printed_evaluations = self.n_function_evaluations

    def _collect(self, fitness=None, y=None):
        """Collect final optimization states shared by all `DEP` classes."""
        self._print_verbose_info(fitness, y)
        results = DO._collect(self, fitness)
        results['_n_generations'] = self._n_generations
        return results
