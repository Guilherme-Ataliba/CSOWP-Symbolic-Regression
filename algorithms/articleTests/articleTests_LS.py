import sys
sys.path.append("../src/")

from trainSR import trainSR
from CSOWP_SR import *
from ExpressionTree import *
import os

if __name__ == '__main__':
    optimizations = [
        "LS"
    ]

    file_names = [
        "F5_specific", 
        "F1", "F4", "F5", "F6", "F7", 
        "radioactive_decay_specific"
    ]

    funcs_dict = {
        "F5_specific": "lambda x: 3 + 2.13*np.log(x)", 
        "F1": "lambda x: 1.57 + 24.3*x", "F4": "lambda x: -2.3 + 0.13*np.sin(x)", 
        "F5": "lambda x: 3 + 2.13*np.log(x)", "F6": "lambda x: 1.3 + 0.13*np.sqrt(x)",
        "F7": "lambda x: 213.809408*(1-np.exp(-0.547237*x))", 
        "radioactive_decay_specific": "lambda x: 10*np.exp(-0.5*x)"
    }
    funcs = list(funcs_dict.values())

    infos = [
        {"Expression": "3 + 2.13ln(x)"},
        {"Expression": "1.57 + 24.3x"}, {"Expression": "-2.3 + 0.13sin(x)"},
        {"Expression": "3 + 2.13ln(x)"}, {"Expression": "1.3 + 0.13sqrt(x)"},
        {"Expression": "213.809408(1 - e^{-0.547237x})"}, {"Expression": "10e^{-0.5x}"}
    ]

    functions_dict = {
        "F5_specific": ["log"], 
        "F1": None, "F4": None, "F5": None, "F6": None, "F7": None,
        "radioactive_decay_specific": ["exp", "exp-"]
    }
    functions = list(functions_dict.values())

    x_ranges_dict = {
        "F5_specific": (0.1, 10), "F1": (-5, 5), 
        "F4": (-5, 5), "F5": (0.1, 10), "F6": (0.1, 10),
        "F7": (0.1, 20), "radioactive_decay_specific": (-5, 5)
    }
    x_ranges = list(x_ranges_dict.values())

    print(file_names)
    print(funcs)
    print(infos)
    print(functions)
    print(x_ranges)
    print(len(file_names), len(funcs), len(infos), len(functions), len(x_ranges))

    for name in file_names:
        if name not in x_ranges_dict:
            print(name)

    print(f"All optimizations to train are: {optimizations}")

    for opt in optimizations:
        print(f"Training optimization {opt}")

        current_path = f"Outputs/{opt}"
        if not os.path.isdir(current_path):
            os.makedirs(current_path)

        TSR = trainSR(2000, 6, dir_path=current_path, overwrite=True, 
        n_points=1000, x_range=[-10, 15], optimization_kind=opt, n_runs=30, SEED=42)
        
        TSR.addFunction("exp-", {
            "function": lambda a : np.exp(-a),
            "weight": 4,
            "custom_function_dict": ["np.exp(-", ")"]
        })

        TSR.addFunction("cube", {
            "function": lambda a: a**3,
            "weight": 3,
            "custom_function_dict": ["(", ")**3"]
        })
        
        TSR.fit(file_name=file_names, func=funcs, info=infos, 
                functions=functions, x_range=x_ranges)

        results = TSR.runParallel()

