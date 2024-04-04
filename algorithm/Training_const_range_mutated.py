import os
from functools import singledispatchmethod
from typing import Any, List
import pandas as pd
import numpy as np
from random import randint, choice, uniform
import graphviz
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.preprocessing import MinMaxScaler


from CSOWP_SR import *
from ExpressionTree import *
os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/Graphviz-10.0.1-win64/bin/'

###=====================================================================================================================

# Generating data
def func(x):
    a=10
    b=-0.5
    c=-0.5
    d=2
    return a*np.exp(b*np.exp(c*x + d))
vfunc = np.vectorize(func)

X = np.linspace(0, 15, 1000)
y = vfunc(X)

#Standardizing input data
scaler = MinMaxScaler((0,1))
X_scaled = scaler.fit_transform(np.c_[X])
y_scaled = scaler.fit_transform(np.c_[y]); y_scaled = y_scaled.reshape(-1, )


# Populations to train
populations = [20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200,
               250, 300, 350, 400, 550, 600, 750, 800, 950, 1000, 
               2000, 3000, 4000, 5000, 6000, 7000]


# Training the Model
DIR_PATH = "output_const_range_mutated/"

if os.path.isfile(DIR_PATH + "results.csv"):
    raise OSError("File exists")
else:
    with open(DIR_PATH + "results.csv", "w") as file:
        file.write("population_size, fitness_score, training_time\n")

for population in populations:
    print(f"Training for population {population}")
    
    SR = SymbolicRegression(3, max_expression_size=3, max_population_size=population,
                            max_island_count=int(population/10), random_const_range=(-10, 10))
    SR.fit(X_scaled, y_scaled, feature_names=["x"])
    
    start_time = time.time()
    output_AEG = SR.predict()
    end_time = time.time()
    data = SR.evaluate_tree(output_AEG.sexp)
    data = pd.DataFrame(np.c_[X_scaled, data], columns=["x", "y"])
    data.to_csv(DIR_PATH + f"data/data-{population}.csv", sep=",", index=False)
    
    graph = output_AEG.sexp.visualize_tree()
    graph.render(DIR_PATH + f"trees/tree-{population}", format="svg")
    
    with open(DIR_PATH + "results.csv", "a") as file:
        file.write(f"{population}, {SR.fitness_score(output_AEG)}, {end_time - start_time}\n")


