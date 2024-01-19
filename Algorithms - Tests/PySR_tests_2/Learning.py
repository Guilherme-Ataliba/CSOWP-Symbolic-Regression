import pysr
import sympy
import numpy as np
from matplotlib import pyplot as plt
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split

pysr.install(precompile=False)

np.random.seed(0)
X = 2 * np.random.randn(100, 5)
y = 2.5382 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 2

default_pysr_params = dict(
    populations=30,
    model_selection="best",
)

print("Started Learning Process")
# Learn equations
model = PySRRegressor(
    timeout_in_seconds=30,
    binary_operators=["plus", "mult"],
    unary_operators=["cos", "exp", "sin"],
    **default_pysr_params
)

model.fit(X, y)

print(model)