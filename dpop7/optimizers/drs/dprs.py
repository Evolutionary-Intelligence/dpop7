import numpy as np  # engine for numerical computing

from dpop7.optimizers.drs.drs import DRS  # base class for Distributed Random Search


class DPRS(DRS):
    """Distributed Pure Random Search (DPRS).

    .. note:: Here we include `DPRS` mainly for *benchmarking* purpose. As pointed out in
       `Probabilistic Machine Learning <https://probml.github.io/pml-book/book2.html>`_,
       *"Pure Random Search should always be tried as a baseline"*.

    Parameters
    ----------
    problem : dict
              problem arguments with the following common settings (`keys`):
                * 'fitness_function' - objective function to be **minimized** (`func`),
                * 'ndim_problem'     - number of dimensionality (`int`),
                * 'upper_boundary'   - upper boundary of search range (`array_like`),
                * 'lower_boundary'   - lower boundary of search range (`array_like`).
    options : dict
              optimizer options with the following common settings (`keys`):
                * 'n_individuals'            - swarm (population) size, aka parallel number of particles (`int`),
                * 'max_function_evaluations' - maximum of function evaluations (`int`, default: `np.Inf`),
                * 'max_runtime'              - maximal runtime to be allowed (`float`, default: `np.Inf`),
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`).

    Examples
    --------
    Use the `DPRS` optimizer to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

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
        # set as default: 1 -> uniformly distributed random sampling (only for initialization)
        self._sampling_type = options.get('_sampling_type', 1)
        if self._sampling_type not in [0, 1]:  # 0 -> normally distributed random sampling
            info = 'Support uniform (1) or normal (0) random sampling (only for initialization).'
            raise ValueError(info.format(self.__class__.__name__))
        elif self._sampling_type == 0:
            self.sigma = options.get('sigma')  # initial global step-size (only for initialization)
            assert self.sigma is not None

    def _sample(self, rng):
        """Helper function to random sampling only for initialization."""
        if self._sampling_type == 0:
            x = self.x + self.sigma*rng.standard_normal(size=(self.ndim_problem,))
        else:
            x = rng.uniform(self.initial_lower_boundary, self.initial_upper_boundary)
        return x

    def initialize(self):
        """Only for the initialization stage."""
        if self.x is None:
            x = self._sample(self.rng_initialization)
        else:
            x = np.copy(self.x)
        assert len(x) == self.ndim_problem
        return x

    def iterate(self):  # population-based sampling
        """Only for the iteration stage."""
        return self.rng_optimization.uniform(self.lower_boundary, self.upper_boundary,
                                             size=(self.n_individuals, self.ndim_problem))
