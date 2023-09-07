import time
import numpy as np  # engine for numerical computing

# abstract class of all distributed optimizers and Ray-based parallel fitness evaluations
from dpop7.optimizers.core.distributed_optimizer import DistributedOptimizer as DO
from dpop7.optimizers.core.distributed_optimizer import parallelize_evaluations as p_e


class DRS(DO):
    """Distributed Random (stochastic) Search (optimization) (DRS).

    This is the **base** class for all `DRS` classes. Please use any of its instantiated
    subclasses to optimize the black-box problem at hand on CPU-based distributed computing
    platforms. Recently, all of its *state-of-the-art* versions adopt the *population-based*
    random sampling strategy for better exploration in *complex* search spaces.

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

    Attributes
    ----------

    Methods
    -------

    References
    ----------
    Gao, K. and Sener, O., 2022, June.
    Generalizing Gaussian smoothing for random search.
    In International Conference on Machine Learning (pp. 7077-7101). PMLR.
    https://proceedings.mlr.press/v162/gao22f.html

    Nesterov, Y. and Spokoiny, V., 2017.
    Random gradient-free minimization of convex functions.
    Foundations of Computational Mathematics, 17(2), pp.527-566.
    https://link.springer.com/article/10.1007/s10208-015-9296-2

    Bergstra, J. and Bengio, Y., 2012.
    Random search for hyper-parameter optimization.
    Journal of Machine Learning Research, 13(2).
    https://www.jmlr.org/papers/v13/bergstra12a.html

    Appel, M.J., Labarre, R. and Radulovic, D., 2004.
    On accelerated random search.
    SIAM Journal on Optimization, 14(3), pp.708-731.
    https://epubs.siam.org/doi/abs/10.1137/S105262340240063X

    Schmidhuber, J., Hochreiter, S. and Bengio, Y., 2001.
    Evaluating benchmark problems by random guessing.
    A Field Guide to Dynamical Recurrent Networks, pp.231-235.
    https://ml.jku.at/publications/older/ch9.pdf

    Rastrigin, L.A., 1986.
    Random search as a method for optimization and adaptation.
    In Stochastic Optimization.
    https://link.springer.com/chapter/10.1007/BFb0007129

    Solis, F.J. and Wets, R.J.B., 1981.
    Minimization by random search techniques.
    Mathematics of Operations Research, 6(1), pp.19-30.
    https://pubsonline.informs.org/doi/abs/10.1287/moor.6.1.19

    Schrack, G. and Choit, M., 1976.
    Optimized relative step size random searches.
    Mathematical Programming, 10(1), pp.230-244.
    https://link.springer.com/article/10.1007/BF01580669

    Schumer, M.A. and Steiglitz, K., 1968.
    Adaptive step size random search.
    IEEE Transactions on Automatic Control, 13(3), pp.270-276.
    https://ieeexplore.ieee.org/abstract/document/1098903

    Matyas, J., 1965.
    Random optimization.
    Automation and Remote control, 26(2), pp.246-253.
    https://tinyurl.com/25339c4x
    (*Since it was written originally in Russian, we cannot read it. However, owing to its historical position,
    we still choose to include it here, which causes a nonstandard citation.*)

    Rastrigin, L.A., 1963.
    The convergence of the random search method in the extremal control of a many parameter system.
    Automaton & Remote Control, 24, pp.1337-1342.
    https://tinyurl.com/djfdnpx4

    Brooks, S.H., 1958.
    A discussion of random methods for seeking maxima.
    Operations Research, 6(2), pp.244-251.
    https://pubsonline.informs.org/doi/abs/10.1287/opre.6.2.244
    """
    def __init__(self, problem, options):
        """Initialize the class with two inputs (problem arguments and optimizer options)."""
        DO.__init__(self, problem, options)
        assert self.n_individuals > 0
        self._n_generations = 0  # number of generations

    def initialize(self):
        """Only for the initialization stage."""
        raise NotImplementedError

    def iterate(self):
        """Only for the iteration stage."""
        raise NotImplementedError

    def _print_verbose_info(self, fitness, y):
        """Print verbose information with a predefined frequency for logging."""
        if self.saving_fitness:  # to save all fitnesses
            if not np.isscalar(y):
                fitness.extend(y)
            else:
                fitness.append(y)
        if self.verbose and ((not self._n_generations % self.verbose) or (self.termination_signal > 0)):
            info = '  * Generation {:d}: best_so_far_y {:7.5e}, min(y) {:7.5e} & Evaluations {:d}'
            print(info.format(self._n_generations, self.best_so_far_y, np.min(y), self.n_function_evaluations))

    def _collect(self, fitness, y=None):
        """Collect final optimization states shared by all `DRS` classes."""
        if y is not None:
            self._print_verbose_info(fitness, y)
        results = DO._collect(self, fitness)
        results['_n_generations'] = self._n_generations
        return results

    def optimize(self, fitness_function=None, args=None):
        """For the entire optimization/evolution stage: initialization + iteration."""
        fitness = DO.optimize(self, fitness_function)
        while not self._check_terminations():
            x = self.iterate()  # to sample new points
            self.start_function_evaluations = time.time()
            y = p_e(self.fitness_function, x, args)  # to evaluate these parallel points
            self.time_function_evaluations += time.time() - self.start_function_evaluations
            self.n_function_evaluations += len(y)
            # update best-so-far solution and fitness
            i = np.argmin(y)
            if y[i] < self.best_so_far_y:
                self.best_so_far_x, self.best_so_far_y = np.copy(x[i]), y[i]
            self._print_verbose_info(fitness, y)
            self._n_generations += 1
        return self._collect(fitness, y)
