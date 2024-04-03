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

X_intensified = np.linspace(-5, 15, 1000)
y_intensified = vfunc(X_intensified)

#Standardizing input data
scaler = MinMaxScaler((0,1))
X_inten_scaled = scaler.fit_transform(np.c_[X_intensified])
y_inten_scaled = scaler.fit_transform(np.c_[y_intensified]); y_inten_scaled = y_inten_scaled.reshape(-1, )

# Generations to train
generations = [i for i in range(1, 201)]
print(generations)

DIR_PATH = "output_generations_increased/"

if os.path.isfile(DIR_PATH + "results.csv"):
    raise OSError("File exists")
else:
    with open(DIR_PATH + "results.csv", "w") as file:
        file.write("max_generations, fitness_score, training_time\n")


# Training the Model
population = 2000

for generation in generations:
    print(f"Training for generations {generation}")
    
    SR = SymbolicRegression(generation, max_expression_size=3, max_population_size=population,
                            max_island_count=int(population/10), random_const_range=(-10, 10))
    SR.fit(X_inten_scaled, y_inten_scaled, feature_names=["x"])
    
    start_time = time.time()
    output_AEG = SR.predict()
    end_time = time.time()
    data = SR.evaluate_tree(output_AEG.sexp)
    data = pd.DataFrame(np.c_[X_inten_scaled, data], columns=["x", "y"])
    data.to_csv(DIR_PATH + f"data/data-{generation}.csv", sep=",", index=False)
    
    graph = output_AEG.sexp.visualize_tree()
    graph.render(DIR_PATH + f"trees/tree-{generation}", format="svg")
    
    with open(DIR_PATH + "results.csv", "a") as file:
        file.write(f"{generation}, {SR.fitness_score(output_AEG)}, {end_time - start_time}\n")