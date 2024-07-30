from trainSR import trainSR
# from trainAlgorithm import *
from CSOWP_SR import *
import multiprocessing 
from ExpressionTree import *
import time

if __name__ == '__main__':
    TSR = trainSR("Outputs/tests", 100, 3, overwrite=True, n_points=1000, 
                  x_range=[-5, 5], optimization_kind="LS", n_runs=1)

    tasks = [["test1", "test2", "test3", "test4"], [lambda x: 2*x+3, lambda x: x**2, lambda x: x**3, lambda x: x+2],
              [{"Expression": "2x+3"}, {"Expression": "x^2"}, {"Expression": "x^3"}, {"Expression": "x^4"}]]
    TSR.fit(tasks)

    results = TSR.runParallel()

    for i in results:
        print(i[2])
