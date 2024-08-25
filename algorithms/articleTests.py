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
        "PSO_NEW", "LS", "random_LS",
        "differential_evolution", "dual_annealing"
    ]

    file_names = [
        "F1_specific", "F4_specific", "F5_specific", 
        "F6_specific", "F7_specific", "F11_specific",
        "F1", "F4", "F5", "F6", "F7", "F11",
        "logistic_specific", "logistic", "logistic_noTanh",
        "projectile_motion_specific", "projectile_motion",
        "damped_pendulum_specific", "damped_pendulum",
        "radioactive_decay_specific", "radioactive_decay"
    ]

    funcs_dict = {
        "F1_specific": "lambda x: 1.57 + 24.3*x", "F4_specific": "lambda x: -2.3 + 0.13*np.sin(x)", 
        "F5_specific": "lambda x: 3 + 2.13*np.log(x)", "F6_specific": "lambda x: 1.3 + 0.13*np.sqrt(x)",
        "F7_specific": "lambda x: 213.809408*(1-np.exp(-0.547237*x))", 
        "F11_specific": "lambda x: 6.87 + 11*np.cos(7.23*x**3)",
        "F1": "lambda x: 1.57 + 24.3*x", "F4": "lambda x: -2.3 + 0.13*np.sin(x)", 
        "F5": "lambda x: 3 + 2.13*np.log(x)", "F6": "lambda x: 1.3 + 0.13*np.sqrt(x)",
        "F7": "lambda x: 213.809408*(1-np.exp(-0.547237*x))", 
        "F11": "lambda x: 6.87 + 11*np.cos(7.23*x**3)",
        "logistic_specified": "lambda x: 10*np.exp(-0.5*np.exp(-0.5*x + 2))",
        "logistic": "lambda x: 10*np.exp(-0.5*np.exp(-0.5*x + 2))",
        "logistic_noTanh": "lambda x: 10*np.exp(-0.5*np.exp(-0.5*x + 2))",
        "projectile_motion_specific": "lambda x: 6*x -9.8*x**2",
        "projectile_motion": "lambda x: 6*x -9.8*x**2",
        "damped_pendulum_specific": "lambda x: np.exp(-x/10)*(3*np.cos(2*x))",
        "damped_pendulum": "lambda x: np.exp(-x/10)*(3*np.cos(2*x))",
        "radioactive_decay_specific": "lambda x: 10*np.exp(-0.5*x)", 
        "radioactive_decay": "lambda x: 10*np.exp(-0.5*x)"
    }
    funcs = list(funcs_dict.values())

    infos = [
        {"Expression": "1.57 + 24.3x"}, {"Expression": "-2.3 + 0.13sin(x)"},
        {"Expression": "3 + 2.13ln(x)"}, {"Expression": "1.3 + 0.13sqrt(x)"},
        {"Expression": "213.809408(1 - e^{-0.547237x})"},
        {"Expression": "6.87 + 11cos(7.23x^3)"},
        {"Expression": "1.57 + 24.3x"}, {"Expression": "-2.3 + 0.13sin(x)"},
        {"Expression": "3 + 2.13ln(x)"}, {"Expression": "1.3 + 0.13sqrt(x)"},
        {"Expression": "213.809408(1 - e^{-0.547237x})"},
        {"Expression": "6.87 + 11cos(7.23x^3)"},
        {"Expression": "10e^{-0.5e^{-0.5x + 2}}"}, 
        {"Expression": "10e^{-0.5e^{-0.5x + 2}}"},
        {"Expression": "10e^{-0.5e^{-0.5x + 2}}"},
        {"Expression": "6x - 9.8x^2"}, {"Expression": "6x - 9.8x^2"},
        {"Expression": "e^{-x/10}*3cos(2x)"}, {"Expression": "e^{-x/10}*3cos(2x)"},
        {"Expression": "10e^{-0.5x}"}, {"Expression": "10e^{-0.5x}"}
    ]

    functions_dict = {
        "F1_specific": ["square"], "F4_specific": ["sin", "cos"],
        "F5_specific": ["log"], "F6_specific": ["sqrt"],
        "F7_specific": ["exp", "exp-"], "F11_specific": ["cos", "sin", "cube"],
        "F1": None, "F4": None, "F5": None, "F6": None, "F7": None, "F11": None,
        "logistic_specified": ["exp", "exp-"], "logistic": None,
        "logistic_noTan": ["abs", "square", "cos", "sin", "tan", "exp", "exp-", "sqrt"],
        "projectile_motion_specific": ["square"], "projectile_motion": None,
        "damped_pendulum_specific": ["exp", "exp-", "cos", "sin"], "damped_pendulum": None,
        "radioactive_decay_specific": ["exp", "exp-"], "radioactive_decay": None
    }
    functions = list(functions_dict.values())

    x_ranges_dict = {
        "F1_specific": (-5, 5), "F4_specific": (-5, 5),
        "F5_specific": (0.1, 10), "F6_specific": (0.1, 10),
        "F7_specific": (0.1, 20),
        "F11_specific": (-5, 5), "F1": (-5, 5), 
        "F4": (-5, 5), "F5": (0.1, 10), "F6": (0.1, 10),
        "F7": (0.1, 20), "F11": (-5, 5),
        "logistic_specific": (-5, 15), "logistic": (-5, 15),
        "logistic_noTanh": (-5, 15),
        "projectile_motion_specific": (-5, 5), "projectile_motion": (-5, 5),
        "damped_pendulum_specific": (-5, 5), "damped_pendulum": (-5, 5),
        "radioactive_decay_specific": (-5, 5), "radioactive_decay": (-5, 5)
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


    for opt in optimizations:
        current_path = f"Outputs/articleTests/{opt}"
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

