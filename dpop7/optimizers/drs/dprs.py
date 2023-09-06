import numpy as np  # engine for numerical computing

from dpop7.optimizers.drs.drs import DRS  # base class for Distributed Random Search


class DPRS(DRS):
    """Distributed Pure Random Search (DPRS).

    .. note:: As pointed out in `Probabilistic Machine Learning <https://probml.github.io/pml-book/book2.html>`_,
       *"Pure Random Search should always be tried as a baseline"*.

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
                * 'n_individuals'            - number of parallel samples/individuals of each iteration (`int`),
                * 'max_function_evaluations' - maximum of function evaluations (`int`, default: `np.Inf`),
                * 'max_runtime'              - maximal runtime to be allowed (`float`, default: `np.Inf`),
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`).

    Examples
    --------
    Use the `DPRS` optimizer to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import ray
       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from dpop7.optimizers.drs.dprs import DPRS
       >>> @ray.remote
       ... def f(x):  # for parallel function evaluations
       ...     return rosenbrock(x)
       >>> problem = {'fitness_function': f,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5.0*numpy.ones((2,)),
       ...            'upper_boundary': 5.0*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,  # seed for random number generation
       ...            'n_individuals': 4}  # number of parallel samples to be evaluated
       >>> dprs = DPRS(problem, options)  # initialize the optimizer class
       >>> results = dprs.optimize()  # run the parallel optimization process
       >>> # return the number of used function evaluations and found best-so-far fitness
       >>> print(f"DPRS: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       DPRS: 5000, 0.11497678820610932

    References
    ----------
    Bergstra, J. and Bengio, Y., 2012.
    Random search for hyper-parameter optimization.
    Journal of Machine Learning Research, 13(2).
    https://www.jmlr.org/papers/v13/bergstra12a.html

    Schmidhuber, J., Hochreiter, S. and Bengio, Y., 2001.
    Evaluating benchmark problems by random guessing.
    A Field Guide to Dynamical Recurrent Networks, pp.231-235.
    https://ml.jku.at/publications/older/ch9.pdf

    Brooks, S.H., 1958.
    A discussion of random methods for seeking maxima.
    Operations Research, 6(2), pp.244-251.
    https://pubsonline.informs.org/doi/abs/10.1287/opre.6.2.244
    """
    def __init__(self, problem, options):
        """Initialize the class with two inputs (problem arguments and optimizer options)."""
        DRS.__init__(self, problem, options)

    def initialize(self):  # population-based sampling
        """Only for the initialization stage."""
        return self.rng_optimization.uniform(self.lower_boundary, self.upper_boundary,
                                             size=(self.n_individuals, self.ndim_problem))

    def iterate(self):  # population-based sampling
        """Only for the iteration stage."""
        return self.rng_optimization.uniform(self.lower_boundary, self.upper_boundary,
                                             size=(self.n_individuals, self.ndim_problem))
