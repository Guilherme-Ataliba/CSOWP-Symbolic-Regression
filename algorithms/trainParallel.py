from trainAlgorithm import *
import concurrent.futures
import typing
from CSOWP_SR import *
from ExpressionTree import *

def call_predict(SR:SymbolicRegression):
    return SR.predict()

X1 = np.linspace(0, 10, 1000)
X2 = np.linspace(0, 10, 1000)
y1 = X1**2
y2 = 3*X2

SR1 = SymbolicRegression(3)
SR1.fit(X1, y1)

SR2 = SymbolicRegression(3)
SR2.fit(X2, y2)

if __name__ == "__main__":

    with concurrent.futures.ProcessPoolExecutor() as pool:
        output = pool.map(call_predict, [SR1, SR2])