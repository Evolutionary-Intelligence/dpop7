import ray  # engine for distributed computing
import numpy  # engine for numerical computing
from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
from dpop7.optimizers.drs.dprs import DPRS  # Distributed Pure Random Search


if __name__ == '__main__':
    ray.init()  # to configure the clustering computing platform here


    @ray.remote
    def ff(x):  # fitness_function
        return rosenbrock(x)


    problem = {'fitness_function': ff,  # define problem arguments
               'ndim_problem': 2000,
               'lower_boundary': -5.0*numpy.ones((2000,)),
               'upper_boundary': 5.0*numpy.ones((2000,))}
    options = {'max_function_evaluations': 50000*2,  # set optimizer options
               'n_individuals': 10,  # number of parallel samples (individuals)
               'seed_rng': 2023}
    dprs = DPRS(problem, options)  # initialize the optimizer class
    results = dprs.optimize()  # run the optimization process
    # return the number of used function evaluations and found best-so-far fitness
    print(f"DPRS: {results['n_function_evaluations']}, {results['best_so_far_y']}")
    # DPRS: 100000, 23210329.872200638
