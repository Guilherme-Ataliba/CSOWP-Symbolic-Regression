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


# Diferentiating y
y_differentiated = np.gradient(y_intensified.reshape(1000, ), X_intensified.reshape(1000, )).reshape(-1, )
y_diff_scaled = scaler.fit_transform(np.c_[y_differentiated]); y_diff_scaled = y_diff_scaled.reshape(-1, )

# Training the Model - Nonscaled
DIR_PATH = "output_test/"

def train(population):
    print(f"=-=-=-=-=-=-=-=-=- Training for population {population} =-=-=-=-=-=-=-=-=-=-")
    
    SR = SymbolicRegression(3, max_expression_size=3, max_population_size=population,
                            max_island_count=int(population/10), random_const_range=(-10, 10))
    SR.fit(np.c_[X_intensified], y_differentiated, feature_names=["x"])

    start_time = time.time()
    output_AEG = SR.predict()
    end_time = time.time()
    data = SR.evaluate_tree(output_AEG.sexp)
    
    if data.shape[0] == 1:
        data = np.array([data[0] for i in range(0, 1000)])

    data = pd.DataFrame(np.c_[X_intensified, data], columns=["x", "y"])
    data.to_csv(DIR_PATH + f"data/data-{population}.csv", sep=",", index=False)
    
    graph = output_AEG.sexp.visualize_tree()
    graph.render(DIR_PATH + f"trees/tree-{population}", format="svg")
    
    with open(DIR_PATH + "results.csv", "a") as file:
        file.write(f"{population}, {SR.fitness_score(output_AEG)}, {end_time - start_time}\n")
    
    print(f"=-=-=-=-=-=-=-=- Ended training for population {population} =-=-=-=-=-=-=-=-")
