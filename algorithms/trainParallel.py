from trainAlgorithm import *
import concurrent.futures

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
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.submit(
            testAlgorithm(func, x_range, n_points, paths[0], populations[0],
                          generations, expression_size, const_range=const_range)
        )

        executor.submit(
            testAlgorithm(func, x_range, n_points, paths[1], populations[1],
                          generations, expression_size, const_range=const_range)
        )