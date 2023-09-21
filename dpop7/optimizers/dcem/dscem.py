import time

import numpy as np  # engine for numerical computing

# base class for Distributed Cross-Entropy Method and Ray-based parallel fitness evaluations
from dpop7.optimizers.dcem.dcem import DCEM
from dpop7.optimizers.core.distributed_optimizer import parallelize_evaluations as p_e


class DSCEM(DCEM):
    """Distributed Standard Cross-Entropy Method (DSCEM).

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
                * 'n_individuals'            - number of parallel offspring (`int`, default: `1000`),
                * 'max_function_evaluations' - maximum of function evaluations (`int`, default: `np.Inf`),
                * 'max_runtime'              - maximal runtime to be allowed (`float`, default: `np.Inf`),
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`);
              and with the following particular settings (`keys`):
                * 'sigma'     - initial global step-size, aka mutation strength (`float`),
                * 'mean'      - initial (starting) point, aka mean of Gaussian search distribution (`array_like`),

                  * if not given, it will draw a random sample from the uniform distribution whose search range is
                    bounded by `problem['lower_boundary']` and `problem['upper_boundary']`.

                * 'n_parents' - parent population size (`int`, default: `200`),
                * 'alpha'     - smoothing factor (`float`, default: `0.8`).

    Examples
    --------
    Use the optimizer `DSCEM` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import ray
       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from dpop7.optimizers.dcem.dscem import DSCEM
       >>> @ray.remote
       ... def f(x):  # for parallel function evaluations
       ...     return rosenbrock(x)
       >>> problem = {'fitness_function': f,  # define problem arguments
       ...            'ndim_problem': 100,
       ...            'lower_boundary': -5.0*numpy.ones((100,)),
       ...            'upper_boundary': 5.0*numpy.ones((100,))}
       >>> options = {'max_function_evaluations': 1000000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'sigma': 0.3}  # the global step-size may need to be tuned for better performance
       >>> dscem = DSCEM(problem, options)  # initialize the optimizer class
       >>> results = dscem.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"DSCEM: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       DSCEM: 1000000, 45712.10913791263

    Attributes
    ----------
    alpha         : `float`
                    smoothing factor.
    mean          : `array_like`
                    initial (starting) point, aka mean of Gaussian search distribution.
    n_individuals : `int`
                    number of parallel offspring, aka offspring population size.
    n_parents     : `int`
                    number of parents, aka parental population size.
    sigma         : `float`
                    initial global step-size, aka mutation strength.

    References
    ----------
    Kroese, D.P., Porotsky, S. and Rubinstein, R.Y., 2006.
    The cross-entropy method for continuous multi-extremal optimization.
    Methodology and Computing in Applied Probability, 8(3), pp.383-407.
    https://link.springer.com/article/10.1007/s11009-006-9753-0
    (See [Appendix B Main CE Program] for the official Matlab code.)

    De Boer, P.T., Kroese, D.P., Mannor, S. and Rubinstein, R.Y., 2005.
    A tutorial on the cross-entropy method.
    Annals of Operations Research, 134(1), pp.19-67.
    https://link.springer.com/article/10.1007/s10479-005-5724-z
    """
    def __init__(self, problem, options):
        """Initialize the class with two inputs (problem arguments and optimizer options)."""
        DCEM.__init__(self, problem, options)
        self.alpha = options.get('alpha', 0.8)  # smoothing factor
        assert 0.0 <= self.alpha <= 1.0

    def initialize(self, is_restart=False):
        """Only for the initialization stage."""
        mean = self._initialize_mean(is_restart)
        x = np.empty((self.n_individuals, self.ndim_problem))  # samples (population)
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        return mean, x, y

    def iterate(self, mean=None, x=None, y=None, args=None):
        """Only for the iteration stage."""
        for i in range(self.n_individuals):
            x[i] = mean + self._sigmas*self.rng_optimization.standard_normal((self.ndim_problem,))
        self.start_function_evaluations = time.time()
        y = p_e(self.fitness_function, x, args)  # to evaluate these parallel points
        self.time_function_evaluations += time.time() - self.start_function_evaluations
        self.n_function_evaluations += len(y)
        # update best-so-far solution and fitness
        i = np.argmin(y)
        if y[i] < self.best_so_far_y:
            self.best_so_far_x, self.best_so_far_y = np.copy(x[i]), y[i]
        return x, y

    def _update_parameters(self, mean=None, x=None, y=None):
        xx = x[np.argsort(y)[:self.n_parents]]
        mean = self.alpha*np.mean(xx, axis=0) + (1.0-self.alpha)*mean
        self._sigmas = self.alpha*np.std(xx, axis=0) + (1.0-self.alpha)*self._sigmas
        return mean

    def optimize(self, fitness_function=None, args=None):
        """For the entire optimization/evolution stage: initialization + iteration."""
        fitness = DCEM.optimize(self, fitness_function)
        mean, x, y = self.initialize()
        while not self._check_terminations():
            x, y = self.iterate(mean, x, y, args)
            self._print_verbose_info(fitness, y)
            self._n_generations += 1
            mean = self._update_parameters(mean, x, y)
        return self._collect(fitness, y, mean)
