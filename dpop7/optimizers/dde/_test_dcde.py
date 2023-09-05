import ray  # engine for distributed computing
import numpy  # engine for numerical computing
from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
from dpop7.optimizers.dde.dcde import DCDE  # Distributed Classic Differential Evolution


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
    dcde = DCDE(problem, options)  # initialize the optimizer class
    results = dcde.optimize()  # run the optimization process
    # return the number of used function evaluations and found best-so-far fitness
    print(f"DCDE: {results['n_function_evaluations']}, {results['best_so_far_y']}")
