from trainSR import trainSR
from CSOWP_SR import *
from ExpressionTree import *
import os

if __name__ == '__main__':
    # optimizations = [
    #     "NoOpt", "PSO", "PSO_NEW", "LS", "random_LS",
    #     "differential_evolution", "dual_annealing"
    # ]
    optimizations = [
        "BFGS_random"
    ]

    file_names = [
        "logistic"
    ]

    funcs_dict = {
        "logistic": "lambda x: 10*np.exp(-0.5*np.exp(-0.5*x + 2))"
    }
    funcs = list(funcs_dict.values())

    infos = [
        {"Expression": "10e^{-0.5e^{-0.5x + 2}}"}, 
    ]

    functions_dict = {
        "logistic": None,
    }
    functions = list(functions_dict.values())

    x_ranges_dict = {
        "logistic": (-5, 15),
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
            os.mkdir(current_path)

        TSR = trainSR(2000, 6, dir_path=current_path, overwrite=True, 
        n_points=1000, x_range=[-10, 15], optimization_kind=opt, n_runs=3, SEED=42)
        
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

