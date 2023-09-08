import time
import numpy as np  # engine for numerical computing

# base class for Distributed Evolution Strategies and Ray-based parallel fitness evaluations
from dpop7.optimizers.des.des import DES
from dpop7.optimizers.core.distributed_optimizer import parallelize_evaluations as p_e


class DSAES(DES):
    """Distributed Self-Adaptation Evolution Strategy (DSAES).

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
                * 'n_individuals'            - number of parallel offspring (`int`, default:
                  `4 + int(3*np.log(problem['ndim_problem']))`),
                * 'max_function_evaluations' - maximum of function evaluations (`int`, default: `np.Inf`),
                * 'max_runtime'              - maximal runtime to be allowed (`float`, default: `np.Inf`),
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`);
              and with the following particular settings (`keys`):
                * 'sigma'     - initial global step-size, aka mutation strength (`float`),
                * 'mean'      - initial (starting) point, aka mean of Gaussian search distribution (`array_like`),

                  * if not given, it will draw a random sample from the uniform distribution whose search range is
                    bounded by `problem['lower_boundary']` and `problem['upper_boundary']`.

                * 'n_parents' - number of parents, aka parental population size (`int`, default:
                  `int(options['n_individuals']/2)`),
                * 'lr_sigma'  - learning rate of global step-size (`float`, default:
                  `1.0/np.sqrt(2*problem['ndim_problem'])`).

    Examples
    --------
    Use the optimizer `DSAES` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import ray
       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from dpop7.optimizers.des.dsaes import DSAES
       >>> @ray.remote
       ... def f(x):  # for parallel function evaluations
       ...     return rosenbrock(x)
       >>> problem = {'fitness_function': f,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5.0*numpy.ones((2,)),
       ...            'upper_boundary': 5.0*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'mean': 3.0*numpy.ones((2,)),
       ...            'sigma': 0.1}  # the global step-size may need to be tuned for better performance
       >>> dsaes = DSAES(problem, options)  # initialize the optimizer class
       >>> results = dsaes.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"DSAES: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       DSAES: 5004, 0.07968852575335955

    Attributes
    ----------
    lr_sigma      : `float`
                    learning rate of global step-size.
    mean          : `array_like`
                    initial (starting) point, aka mean of Gaussian search/sampling/mutation distribution.
    n_individuals : `int`
                    number of parallel offspring, aka offspring population size.
    n_parents     : `int`
                    number of parents, aka parental population size.
    sigma         : `float`
                    final global step-size, aka mutation strength.

    References
    ----------
    Beyer, H.G., 2020, July.
    Design principles for matrix adaptation evolution strategies.
    In Proceedings of Annual Conference on Genetic and Evolutionary Computation Companion (pp. 682-700). ACM.
    https://dl.acm.org/doi/abs/10.1145/3377929.3389870

    http://www.scholarpedia.org/article/Evolution_strategies

    See its official Matlab/Octave version from Prof. Beyer:
    https://homepages.fhv.at/hgb/downloads/mu_mu_I_lambda-ES.oct
    """
    def __init__(self, problem, options):
        """Initialize the class with two inputs (problem arguments and optimizer options)."""
        DES.__init__(self, problem, options)
        if self.lr_sigma is None:
            self.lr_sigma = 1.0/np.sqrt(2.0*self.ndim_problem)
        assert self.lr_sigma > 0.0

    def initialize(self, is_restart=False):
        """Only for the initialization stage."""
        x = np.empty((self.n_individuals, self.ndim_problem))  # offspring population
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        sigmas = np.ones((self.n_individuals,))  # global step-size for each offspring
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        self._list_initial_mean.append(np.copy(mean))
        return x, mean, sigmas, y

    def iterate(self, x=None, mean=None, sigmas=None, y=None, args=None):
        """Only for the iteration stage."""
        for k in range(self.n_individuals):  # to sample offspring population
            sigmas[k] = self.sigma*np.exp(self.lr_sigma*self.rng_optimization.standard_normal())
            x[k] = mean + sigmas[k]*self.rng_optimization.standard_normal((self.ndim_problem,))
        self.start_function_evaluations = time.time()
        y = p_e(self.fitness_function, x, args)  # to evaluate these parallel points
        self.time_function_evaluations += time.time() - self.start_function_evaluations
        self.n_function_evaluations += len(y)
        # update best-so-far solution and fitness
        i = np.argmin(y)
        if y[i] < self.best_so_far_y:
            self.best_so_far_x, self.best_so_far_y = np.copy(x[i]), y[i]
        return x, sigmas, y

    def _restart_initialize(self, y):
        min_y = np.min(y)
        if min_y < self._list_fitness[-1]:
            self._list_fitness.append(min_y)
        else:
            self._list_fitness.append(self._list_fitness[-1])
        is_restart_1, is_restart_2 = self.sigma < self.sigma_threshold, False
        if len(self._list_fitness) >= self.stagnation:
            is_restart_2 = (self._list_fitness[-self.stagnation] - self._list_fitness[-1]) < self.fitness_diff
        is_restart = bool(is_restart_1) or bool(is_restart_2)
        if is_restart:
            self._print_verbose_info([], y, True)
            if self.verbose:
                print(' ....... *** restart *** .......')
            self._n_restart += 1
            self._list_generations.append(self._n_generations)  # for each restart
            self._n_generations = 0
            self.n_individuals *= 2
            self.n_parents = int(self.n_individuals/2)
            self._list_fitness = [np.Inf]
        return is_restart

    def restart_initialize(self, x=None, mean=None, sigmas=None, y=None):
        if self._restart_initialize(y):
            self.sigma = np.copy(self._sigma_bak)
            x, mean, sigmas, y = self.initialize(True)
        return x, mean, sigmas, y

    def optimize(self, fitness_function=None, args=None):
        """For the entire optimization/evolution stage: initialization + iteration."""
        fitness = DES.optimize(self, fitness_function)
        x, mean, sigmas, y = self.initialize()
        while not self._check_terminations():
            # sample and evaluate offspring population
            x, sigmas, y = self.iterate(x, mean, sigmas, y, args)
            self._print_verbose_info(fitness, y)
            self._n_generations += 1
            order = np.argsort(y)[:self.n_parents]
            # use intermediate multi-recombination
            mean = np.mean(x[order], axis=0)
            self.sigma = np.mean(sigmas[order])
            if self.is_restart:
                x, mean, sigmas, y = self.restart_initialize(x, mean, sigmas, y)
        return self._collect(fitness, y, mean)
