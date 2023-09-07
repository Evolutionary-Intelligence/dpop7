import time
import numpy as np  # engine for numerical computing

# base class for Distributed Estimation of Distribution Algorithms and Ray-based parallel fitness evaluations
from dpop7.optimizers.deda.deda import DEDA
from dpop7.optimizers.core.distributed_optimizer import parallelize_evaluations as p_e


class DUMDA(DEDA):
    """Distributed Univariate Marginal Distribution Algorithm for normal models (DUMDA).

    .. note:: `UMDA` learns only the *diagonal* elements of covariance matrix of the Gaussian sampling
       distribution, resulting in a *linear* time complexity w.r.t. each sampling. Therefore, it can be
       seen as a *baseline* for large-scale black-box optimization.

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
                * 'n_individuals'            - number of parallel offspring (`int`, default: `200`),
                * 'max_function_evaluations' - maximum of function evaluations (`int`, default: `np.Inf`),
                * 'max_runtime'              - maximal runtime (`float`, default: `np.Inf`),
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`);
              and with the following particular settings (`keys`):
                * 'n_parents' - number of parents, aka parental population size (`int`, default:
                  `int(self.n_individuals/2)`).

    Examples
    --------
    Use the optimizer `DUMDA` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

    Attributes
    ----------
    n_individuals : `int`
                    number of parallel offspring, aka offspring population size.
    n_parents     : `int`
                    number of parents, aka parental population size.

    References
    ----------
    Mühlenbein, H. and Mahnig, T., 2002.
    Evolutionary computation and Wright's equation.
    Theoretical Computer Science, 287(1), pp.145-165.
    https://www.sciencedirect.com/science/article/pii/S0304397502000981

    Larrañaga, P. and Lozano, J.A. eds., 2001.
    Estimation of distribution algorithms: A new tool for evolutionary computation.
    Springer Science & Business Media.
    https://link.springer.com/book/10.1007/978-1-4615-1539-5

    Mühlenbein, H. and Mahnig, T., 2001.
    Evolutionary algorithms: From recombination to search distributions.
    In Theoretical Aspects of Evolutionary Computing (pp. 135-173). Springer, Berlin, Heidelberg.
    https://link.springer.com/chapter/10.1007/978-3-662-04448-3_7

    Larranaga, P., Etxeberria, R., Lozano, J.A. and Pena, J.M., 2000.
    Optimization in continuous domains by learning and simulation of Gaussian networks.
    Technical Report, Department of Computer Science and Artificial Intelligence,
    University of the Basque Country.
    https://tinyurl.com/3bw6n3x4

    Larranaga, P., Etxeberria, R., Lozano, J.A. and Pe, J.M., 1999.
    Optimization by learning and simulation of Bayesian and Gaussian networks.
    Technical Report, Department of Computer Science and Artificial Intelligence,
    University of the Basque Country.
    https://tinyurl.com/5dktrdwc

    Mühlenbein, H., 1997.
    The equation for response to selection and its use for prediction.
    Evolutionary Computation, 5(3), pp.303-346.
    https://tinyurl.com/yt78c786
    """
    def __init__(self, problem, options):
        """Initialize the class with two inputs (problem arguments and optimizer options)."""
        DEDA.__init__(self, problem, options)

    def initialize(self, args=None):
        """Only for the initialization stage."""
        x = self.rng_optimization.uniform(self.initial_lower_boundary, self.initial_upper_boundary,
                                          size=(self.n_individuals, self.ndim_problem))  # population
        self.start_function_evaluations = time.time()
        y = p_e(self.fitness_function, x, args)  # to evaluate these parallel points
        self.time_function_evaluations += time.time() - self.start_function_evaluations
        self.n_function_evaluations += len(y)
        # update best-so-far solution and fitness
        i = np.argmin(y)
        if y[i] < self.best_so_far_y:
            self.best_so_far_x, self.best_so_far_y = np.copy(x[i]), y[i]
        return x, y

    def iterate(self, x=None, y=None, args=None):
        """Only for the iteration stage."""
        order = np.argsort(y)[:self.n_parents]
        mean, sigmas = np.mean(x[order], axis=0), np.std(x[order], axis=0)
        for i in range(self.n_individuals):
            x[i] = mean + sigmas*self.rng_optimization.standard_normal(size=(self.ndim_problem,))
        self.start_function_evaluations = time.time()
        y = p_e(self.fitness_function, x, args)  # to evaluate these parallel points
        self.time_function_evaluations += time.time() - self.start_function_evaluations
        self.n_function_evaluations += len(y)
        # update best-so-far solution and fitness
        i = np.argmin(y)
        if y[i] < self.best_so_far_y:
            self.best_so_far_x, self.best_so_far_y = np.copy(x[i]), y[i]
        return x, y

    def optimize(self, fitness_function=None, args=None):
        """For the entire optimization/evolution stage: initialization + iteration."""
        fitness = DEDA.optimize(self, fitness_function)
        x, y = self.initialize(args)
        while not self._check_terminations():
            self._print_verbose_info(fitness, y)
            x, y = self.iterate(x, y, args)
            self._n_generations += 1
        return self._collect(fitness, y)
