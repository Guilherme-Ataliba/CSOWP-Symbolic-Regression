import sys
sys.path.append('../src/')

from CSOWP_SR import *
from ExpressionTree import *
import utils

import pandas as pd
import numpy as np
import pickle
import re
import sympy as smp
import edist.ted as ted

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

import itertools

import signal
import time
from copy import Error
import logging

# Expected Data ===================================

def recursive_simplify_evalf(expr, precision=15):
    # Base case: if the expr has no args, it is an atom (number or symbol)
    if not expr.args:
        return expr.evalf(precision).simplify()

    # Recursively process all arguments
    simplified_args = [recursive_simplify_evalf(arg, precision) for arg in expr.args]
    
    # Reconstruct the expression with simplified arguments
    simplified_expr = expr.func(*simplified_args)

    # Apply simplify and evalf on the resulting expression
    return simplified_expr.evalf(precision).simplify()

funcs_dict = {
    "F1_specific": "lambda x: 1.57 + 24.3*x", "F4_specific": "lambda x: -2.3 + 0.13*np.sin(x)", 
    "F5_specific": "lambda x: 3 + 2.13*np.log(x)", "F6_specific": "lambda x: 1.3 + 0.13*np.sqrt(x)",
    "F7_specific": "lambda x: 213.809408*(1-np.exp(-0.547237*x))", 
    "F11_specific": "lambda x: 6.87 + 11*np.cos(7.23*x**3)",
    "F1": "lambda x: 1.57 + 24.3*x", "F4": "lambda x: -2.3 + 0.13*np.sin(x)", 
    "F5": "lambda x: 3 + 2.13*np.log(x)", "F6": "lambda x: 1.3 + 0.13*np.sqrt(x)",
    "F7": "lambda x: 213.809408*(1-np.exp(-0.547237*x))", 
    "F11": "lambda x: 6.87 + 11*np.cos(7.23*x**3)",
    "logistic_specific": "lambda x: 10*np.exp(-0.5*np.exp(-0.5*x + 2))",
    "logistic": "lambda x: 10*np.exp(-0.5*np.exp(-0.5*x + 2))",
    "logistic_noTanh": "lambda x: 10*np.exp(-0.5*np.exp(-0.5*x + 2))",
    "projectile_motion_specific": "lambda x: 6*x -9.8*x**2",
    "projectile_motion": "lambda x: 6*x -9.8*x**2",
    "damped_pendulum_specific": "lambda x: np.exp(-x/10)*(3*np.cos(2*x))",
    "damped_pendulum": "lambda x: np.exp(-x/10)*(3*np.cos(2*x))",
    "radioactive_decay_specific": "lambda x: 10*np.exp(-0.5*x)", 
    "radioactive_decay": "lambda x: 10*np.exp(-0.5*x)"
    }

expected_data = pd.DataFrame(columns=["nodes", "adj", "problem", "expected_string"])
expected_data

symbols = {"x": smp.symbols("x", positive=True, real=True)}

for name, function in funcs_dict.items():
    function = function[10:]
    function = smp.parse_expr(function.replace("np.", ""), local_dict=symbols).simplify()
    function = recursive_simplify_evalf(function)

    tree = utils.exprToTree(function, single_name=True)
    node, adj = tree.parentChildRepr().values()
    df = pd.DataFrame([{"nodes": node, "adj": adj, "problem": name, "expected_string": function,
                          "expected_tree": tree}])
    expected_data = pd.concat([expected_data, df])

expected_data.set_index("problem", drop=True, inplace=True)


# Actual Data ======================================


data = utils.create_all_data("../Outputs-first")


# Define a handler that raises a TimeoutError
def handler(signum, frame):
    raise TimeoutError("Execution took longer than 10 seconds!")

# Set the signal handler for SIGALRM
signal.signal(signal.SIGALRM, handler)

def try_simplify(solution):
    return smp.nsimplify(solution, tolerance=1e-5).evalf(20).simplify()

dataFrames = []

# print(len(data))

logging.basicConfig(level=logging.ERROR, filename='info.log', format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Total: ", len(data))

for index, row in data.iterrows():
    solution = row.solution_string
    
    # print(index)
    logging.info(index)


    try:
      signal.alarm(5)
      solution = try_simplify(solution)
      signal.alarm(0)
    except:
      pass

    smp_tree = utils.exprToTree(solution, single_name=True)
    tree_dict = smp_tree.parentChildRepr()
    tree_dict["solution_string"] = solution
    df = pd.DataFrame([tree_dict])
    dataFrames.append(df)

new_data = data.copy()
new_data.drop(["nodes", "adj", "solution_string"], axis=1, inplace=True)
df = pd.concat(dataFrames).reset_index(drop=True)
new_data = pd.concat([new_data, df], axis=1)
data = new_data[["nodes", "adj", "optimization", "problem", "index", "MSE", "training_time(s)", "solution_string", "expression"]]

problem_ted = {}
for problem in data["problem"].unique():
  problem_data = data[data["problem"] == problem]
  expected_problem = expected_data.loc[problem]

  for index, row in problem_data.iterrows():
    ted_value = ted.standard_ted(row.nodes, row.adj, expected_problem.nodes, expected_problem.adj)
    problem_ted[index] = ted_value

problem_ted_df = pd.DataFrame(problem_ted.values(), index=problem_ted.keys(), columns=["TED"])
data = pd.concat([data, problem_ted_df], axis=1)

with open("ted_data.pickle", "wb") as file:
  pickle.dump(data, file)