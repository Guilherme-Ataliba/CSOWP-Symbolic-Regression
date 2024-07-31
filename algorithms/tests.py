from trainSR import trainSR
from CSOWP_SR import *
from ExpressionTree import *

if __name__ == '__main__':
    # new_function = {"function": {"exp-": lambda a : np.exp(-a)},
    #                 "weight": {"exp-": 4},
    #                 "custom_function_dict": {"exp-": ["np.exp(-", ")"]}}

    TSR = trainSR("Outputs/tests", 20, 3, overwrite=True, n_points=1000, 
                  x_range=[-5, 5], optimization_kind="LS", n_runs=1)
    
    TSR.addFunction("exp-", {
        "function": lambda a : np.exp(-a),
        "weight": 4,
        "custom_function_dict": ["np.exp(-", ")"]
    })

    file_names = ["test1", "test2", "test3", "test4"]
    funcs = [lambda x: 2*x+3, lambda x: x**2, lambda x: x**3, lambda x: x+2]
    infos = [{"Expression": "2x+3", "opt": "oi"}, {"Expression": "x^2"}, 
             {"Expression": "x^3"}, {"Expression": "x^4"}]
    
    functions=[["exp-", "exp"], None, None, None]
    
    TSR.fit(file_name=file_names, func=funcs, info=infos, functions=functions)

    results = TSR.runParallel()

    for i in results:
        print(i[3])
