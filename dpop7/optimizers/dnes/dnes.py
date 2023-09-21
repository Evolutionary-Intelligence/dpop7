import numpy as np  # engine for numerical computing

from dpop7.optimizers.des.des import DES  # Distributed Evolution Strategies


class DNES(DES):
    """Distributed Natural Evolution Strategies (DNES).

    This is the **base** class for all `DNES` classes. Please use any of its instantiated
    subclasses to optimize the black-box problem at hand on CPU-based distributed computing
    platforms.

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
                * 'n_individuals'            - number of parallel offspring/descendants (`int`),
                * 'max_function_evaluations' - maximum of function evaluations (`int`, default: `np.Inf`),
                * 'max_runtime'              - maximal runtime to be allowed (`float`, default: `np.Inf`),
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`);
              and with the following particular settings (`keys`):
                * 'mean'      - initial (starting) point (`array_like`),

                  * if not given, it will draw a random sample from the uniform distribution whose search range is
                    bounded by `problem['lower_boundary']` and `problem['upper_boundary']`.

                * 'sigma'     - initial global step-size, aka mutation strength (`float`),
                * 'n_parents' - number of parents/ancestors, aka parental population size (`int`).

    Attributes
    ----------
    mean          : `array_like`
                    initial (starting) point, aka mean of Gaussian search/sampling/mutation distribution.
    n_individuals : `int`
                    number of parallel offspring/descendants, aka offspring population size.
    n_parents     : `int`
                    number of parents/ancestors, aka parental population size.
    sigma         : `float`
                    global step-size, aka mutation strength (i.e., overall std of Gaussian search distribution).

    Methods
    -------

    References
    ----------
    Wierstra, D., Schaul, T., Glasmachers, T., Sun, Y., Peters, J. and Schmidhuber, J., 2014.
    Natural evolution strategies.
    Journal of Machine Learning Research, 15(1), pp.949-980.
    https://jmlr.org/papers/v15/wierstra14a.html

    Schaul, T., 2011.
    Studies in continuous black-box optimization.
    Doctoral Dissertation, Technische Universität München.
    https://people.idsia.ch/~schaul/publications/thesis.pdf

    Yi, S., Wierstra, D., Schaul, T. and Schmidhuber, J., 2009, June.
    Stochastic search using the natural gradient.
    In Proceedings of International Conference on Machine Learning (pp. 1161-1168).
    https://dl.acm.org/doi/10.1145/1553374.1553522

    Wierstra, D., Schaul, T., Peters, J. and Schmidhuber, J., 2008, June.
    Natural evolution strategies.
    In IEEE Congress on Evolutionary Computation (pp. 3381-3387). IEEE.
    https://ieeexplore.ieee.org/abstract/document/4631255

    https://github.com/pybrain/pybrain
    """
    def __init__(self, problem, options):
        """Initialize the class with two inputs (problem arguments and optimizer options)."""
        DES.__init__(self, problem, options)
        self._u = None  # for fitness shaping

    def initialize(self):
        """Only for the initialization stage."""
        r, _u = np.arange(self.n_individuals), np.zeros((self.n_individuals,))
        better = r >= self.n_individuals*0.5
        _u[better] = r[better] - self.n_individuals*0.5
        self._u = _u/np.max(_u)

    def iterate(self):
        """Only for the iteration stage."""
        raise NotImplementedError
