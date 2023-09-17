import time
import numpy as np  # engine for numerical computing

# base class for Distributed Evolutionary Programming and Ray-based parallel fitness evaluations
from dpop7.optimizers.dep.dep import DEP
from dpop7.optimizers.core.distributed_optimizer import parallelize_evaluations as p_e


class DCEP(DEP):
    """Distributed Classical Evolutionary Programming with self-adaptive mutation (CEP).

    .. note:: To obtain satisfactory performance for large-scale black-box optimization,
       the number of parallel offspring (`n_individuals`) and also initial global step-size
       (`sigma`) may need to be **carefully** tuned (e.g. via manual trial-and-error or
       automatical hyper-parameter optimization).

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
                * 'sigma'          - initial global step-size, aka mutation strength (`float`),
                * 'q'              - number of opponents for pairwise comparisons (`int`, default: `10`),
                * 'tau'            - learning rate of individual step-sizes self-adaptation (`float`, default:
                  `1.0/np.sqrt(2.0*np.sqrt(problem['ndim_problem']))`),
                * 'tau_apostrophe' - learning rate of individual step-sizes self-adaptation (`float`, default:
                  `1.0/np.sqrt(2.0*problem['ndim_problem'])`.

    Examples
    --------
    Use the optimizer `DCEP` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import ray
       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> @ray.remote
       ... def f(x):  # for parallel function evaluations
       ...     return rosenbrock(x)
       >>> from dpop7.optimizers.dep.dcep import DCEP
       >>> problem = {'fitness_function': f,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5.0*numpy.ones((2,)),
       ...            'upper_boundary': 5.0*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'sigma': 0.1}
       >>> dcep = DCEP(problem, options)  # initialize the optimizer class
       >>> results = dcep.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"DCEP: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       DCEP: 5000, 0.3544823323771589

    Attributes
    ----------
    n_individuals  : `int`
                     number of parallel offspring, aka offspring population size.
    q              : `int`
                     number of opponents for pairwise comparisons.
    sigma          : `float`
                     initial global step-size, aka mutation strength.
    tau            : `float`
                     learning rate of individual step-sizes self-adaptation.
    tau_apostrophe : `float`
                     learning rate of individual step-sizes self-adaptation.

    References
    ----------
    Yao, X., Liu, Y. and Lin, G., 1999.
    Evolutionary programming made faster.
    IEEE Transactions on Evolutionary Computation, 3(2), pp.82-102.
    https://ieeexplore.ieee.org/abstract/document/771163
    (Xin Yao won the **2013** `Evolutionary Computation Pioneer Award from IEEE CIS
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
        DEP.__init__(self, problem, options)
        self.q = options.get('q', 10)  # number of opponents for pairwise comparisons
        assert self.q > 0
        # set two learning-rates of individual step-sizes adaptation
        self.tau = options.get('tau', 1.0/np.sqrt(2.0*np.sqrt(self.ndim_problem)))
        assert self.tau > 0.0
        self.tau_apostrophe = options.get('tau_apostrophe', 1.0/np.sqrt(2.0*self.ndim_problem))
        assert self.tau_apostrophe > 0.0

    def initialize(self, args=None):
        """Only for the initialization stage."""
        x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary,
                                            size=(self.n_individuals, self.ndim_problem))
        sigmas = self.sigma*np.ones((self.n_individuals, self.ndim_problem))  # eta
        self.start_function_evaluations = time.time()
        y = np.array(p_e(self.fitness_function, x, args))  # to evaluate these parallel points
        self.time_function_evaluations += time.time() - self.start_function_evaluations
        self.n_function_evaluations += len(y)
        # update best-so-far solution and fitness
        i = np.argmin(y)
        if y[i] < self.best_so_far_y:
            self.best_so_far_x, self.best_so_far_y = np.copy(x[i]), y[i]
        xx = np.empty((self.n_individuals, self.ndim_problem))
        ss = np.empty((self.n_individuals, self.ndim_problem))  # eta
        yy = np.copy(y)
        return x, sigmas, y, xx, ss, yy

    def iterate(self, x=None, sigmas=None, y=None, xx=None, ss=None, yy=None, args=None):
        """Only for the iteration stage."""
        for i in range(self.n_individuals):
            if self._check_terminations():
                return x, sigmas, y, xx, ss, yy
            # base = self.rng_optimization.standard_normal()
            # ss[i] = sigmas[i]*np.exp(self.tau_apostrophe*base + self.tau*self.rng_optimization.standard_normal(
            #     size=(self.ndim_problem,)))
            ss[i] = sigmas[i]*np.exp(self.tau_apostrophe*self.rng_optimization.standard_normal(
                size=(self.ndim_problem,)) + self.tau*self.rng_optimization.standard_normal(
                size=(self.ndim_problem,)))
            xx[i] = x[i] + ss[i]*self.rng_optimization.standard_normal(size=(self.ndim_problem,))
        self.start_function_evaluations = time.time()
        yy = np.array(p_e(self.fitness_function, xx, args))  # to evaluate these parallel points
        self.time_function_evaluations += time.time() - self.start_function_evaluations
        self.n_function_evaluations += len(yy)
        # update best-so-far solution and fitness
        i = np.argmin(yy)
        if yy[i] < self.best_so_far_y:
            self.best_so_far_x, self.best_so_far_y = np.copy(xx[i]), yy[i]
        new_x = np.vstack((xx, x))
        new_sigmas = np.vstack((ss, sigmas))
        new_y = np.hstack((yy, y))
        n_win = np.zeros((2*self.n_individuals,))  # number of win
        for i in range(2*self.n_individuals):
            for j in self.rng_optimization.choice([k for k in range(2*self.n_individuals) if k != i],
                                                  size=self.q, replace=False):
                if new_y[i] < new_y[j]:
                    n_win[i] += 1
        order = np.argsort(-n_win)[:self.n_individuals]
        x[:self.n_individuals] = new_x[order]
        sigmas[:self.n_individuals] = new_sigmas[order]
        y[:self.n_individuals] = new_y[order]
        self._n_generations += 1
        return x, sigmas, y, xx, ss, yy

    def optimize(self, fitness_function=None, args=None):
        """For the entire optimization/evolution stage: initialization + iteration."""
        fitness = DEP.optimize(self, fitness_function)
        x, sigmas, y, xx, ss, yy = self.initialize(args)
        while not self._check_terminations():
            self._print_verbose_info(fitness, yy)
            x, sigmas, y, xx, ss, yy = self.iterate(x, sigmas, y, xx, ss, yy, args)
        return self._collect(fitness, yy)
