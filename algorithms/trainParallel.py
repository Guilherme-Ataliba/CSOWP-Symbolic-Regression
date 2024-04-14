from trainAlgorithm import *
import concurrent.futures
import typing


class TrainInstance():

    def __init__(self, func, x_range, n_points, dir_path, population,
                 generations, max_expression_size, normalize=False, 
                 const_range=(0,1), normalize_range=(0,1), ignore_warning=True,
                 overwrite=False, n_runs=1):
        self.func = func
        self.x_range = x_range
        self.n_points = n_points
        self.dir_path = dir_path
        self.population = population
        self.generations = generations
        self.max_expression_size = max_expression_size
        self.normalize = normalize
        self.const_range = const_range
        self.normalize_range = normalize_range
        self.igore_warning = ignore_warning
        self.overwrite = overwrite
        self.n_runs = n_runs

    def auxTrainFunc(self, arguments):
        testAlgorithm(self.func, self.x_range, self.n_points, self.dir_path,
                        self.population, self.generations, self.max_expression_size,
                        self.normalize, self.const_range, self.normalize_range,
                        self.ignore_warning, self.overwrite, self.n_runs)
        
    def auxTrainPopulation(self, arguments):
        paths = arguments[0]
        populations = arguments[1]
        testAlgorithm(self.func, self.x_range, self.n_points, paths,
                        populations, self.generations, self.max_expression_size,
                        self.normalize, self.const_range, self.normalize_range,
                        self.ignore_warning, self.overwrite, self.n_runs)
    

if __name__ == "__main__":

    # Defining Parameters
    populations = [100, 200]
    generations = 3
    expression_size = 3

    paths = ["output_test1", "output_test2"]
    paths = ["Outputs/" + str(i) for i in paths]

    def func(x):
        a=10
        b=-0.5
        c=-0.5
        d=2
        return a*np.exp(b*np.exp(c*x + d))
    x_range = (-5, 15)
    n_points = 1000
    const_range = (-10, 10)

    # Parallel ===========================================
    trainTest = TrainInstance(func, x_range, n_points, paths[0], populations[0], 
                              generations, expression_size, const_range=const_range)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(trainTest.auxTrainPopulation, [paths, populations])
        
        # executor.submit(
        #     testAlgorithm(func, x_range, n_points, paths[0], populations[0],
        #                   generations, expression_size, const_range=const_range)
        # )

        # executor.submit(
        #     testAlgorithm(func, x_range, n_points, paths[1], populations[1],
        #                   generations, expression_size, const_range=const_range)
        # )