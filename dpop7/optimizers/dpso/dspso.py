import time
import numpy as np  # engine for numerical computing

# abstract class of all distributed optimizers and Ray-based parallel fitness evaluations
from dpop7.optimizers.dpso.dpso import DPSO
from dpop7.optimizers.core.distributed_optimizer import parallelize_evaluations as p_e


class DSPSO(DPSO):
    """Distributed Standard Particle Swarm Optimizer with a global topology (SPSO).

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
                * 'max_function_evaluations' - maximum of function evaluations (`int`, default: `np.Inf`),
                * 'max_runtime'              - maximal runtime to be allowed (`float`, default: `np.Inf`),
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`);
              and with the following particular settings (`keys`):
                * 'n_individuals' - swarm (population) size, aka parallel number of particles (`int`, default: `20`),
                * 'cognition'     - cognitive learning rate (`float`, default: `2.0`),
                * 'society'       - social learning rate (`float`, default: `2.0`),
                * 'max_ratio_v'   - maximal ratio of velocities w.r.t. search range (`float`, default: `0.2`).

    Examples
    --------
    Use the optimizer `DSPSO` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

    Attributes
    ----------
    cognition     : `float`
                    cognitive learning rate, aka acceleration coefficient.
    max_ratio_v   : `float`
                    maximal ratio of velocities w.r.t. search range.
    n_individuals : `int`
                    swarm (population) size, aka parallel number of particles.
    society       : `float`
                    social learning rate, aka acceleration coefficient.

    References
    ----------
    Floreano, D. and Mattiussi, C., 2008.
    Bio-inspired artificial intelligence: Theories, methods, and technologies.
    MIT Press.
    https://mitpress.mit.edu/9780262062718/bio-inspired-artificial-intelligence/
    (See [Chapter 7.2 Particle Swarm Optimization] for details.)

    Venter, G. and Sobieszczanski-Sobieski, J., 2003.
    Particle swarm optimization.
    AIAA Journal, 41(8), pp.1583-1589.
    https://arc.aiaa.org/doi/abs/10.2514/2.2111

    Eberhart, R.C., Shi, Y. and Kennedy, J., 2001.
    Swarm intelligence.
    Elsevier.
    https://www.elsevier.com/books/swarm-intelligence/eberhart/978-1-55860-595-4

    Shi, Y. and Eberhart, R., 1998, May.
    A modified particle swarm optimizer.
    In IEEE World Congress on Computational Intelligence (pp. 69-73). IEEE.
    https://ieeexplore.ieee.org/abstract/document/699146

    Kennedy, J. and Eberhart, R., 1995, November.
    Particle swarm optimization.
    In Proceedings of International Conference on Neural Networks (pp. 1942-1948). IEEE.
    https://ieeexplore.ieee.org/document/488968
    """
    def __init__(self, problem, options):
        """Initialize the class with two inputs (problem arguments and optimizer options)."""
        DPSO.__init__(self, problem, options)

    def iterate(self, v=None, x=None, y=None, p_x=None, p_y=None, n_x=None, args=None):
        """Only for the iteration stage."""
        for i in range(self.n_individuals):
            n_x[i] = p_x[np.argmin(p_y)]  # update within global topology
            cognition_rand = self.rng_optimization.uniform(size=(self.ndim_problem,))
            society_rand = self.rng_optimization.uniform(size=(self.ndim_problem,))
            v[i] = (self._w[min(self._n_generations, len(self._w) - 1)]*v[i] +
                    self.cognition*cognition_rand*(p_x[i] - x[i]) +
                    self.society*society_rand*(n_x[i] - x[i]))  # velocity update
            v[i] = np.clip(v[i], self._min_v, self._max_v)
            x[i] += v[i]  # position update
            if self.is_bound:
                x[i] = np.clip(x[i], self.lower_boundary, self.upper_boundary)
        self.start_function_evaluations = time.time()
        y = p_e(self.fitness_function, x, args)  # to evaluate these parallel points
        self.time_function_evaluations += time.time() - self.start_function_evaluations
        self.n_function_evaluations += len(y)
        for i in range(self.n_individuals):
            if y[i] < p_y[i]:
                p_x[i], p_y[i] = x[i], y[i]
        self._n_generations += 1
        return v, x, y, p_x, p_y, n_x
