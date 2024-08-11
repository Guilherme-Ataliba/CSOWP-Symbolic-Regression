from trainSR import trainSR
from trainAlgorithm import *
from CSOWP_SR import *
from ExpressionTree import *

if __name__ == "__main__":
    def func(X):
        return np.sqrt(X)

    # X = np.linspace(-5, -1, 1000)
    # y = func(X)

    # tree = ExpressionTree()
    # p = tree.add_root("sqrt", "function")
    # tree.add_left(p, "x", e_type="feature")
    # tree.visualize_tree()

    # SR = SymbolicRegression(3)
    # SR.fit(np.c_[X], y, feature_names=["x"])
    # print(SR.evaluate_tree(tree))

    TSR = trainSR(population=20, generations=2, dir_path=None, x_range=(-5, -1),
                n_points=1000, optimization_kind="LS")
    TSR.fit(file_name=["oi", "ola"], func=[func, func])
    TSR.runParallel(max_processes=2)