from  CSOWP_SR import *
from ExpressionTree import *
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from time import time
import os 
import warnings

os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/Graphviz-10.0.1-win64/bin/'

def testAlgorithm(func, x_range, n_points, dir_path, population, generations, 
                  max_expression_size, normalize=False, const_range=(0,1),
                  normalize_range=(0,1), ignore_warning=True, overwrite=False,
                  n_runs=1):

    # Initial Definitions ==============================
    if ignore_warning:
        warnings.filterwarnings("ignore")
    if not os.path.isdir(dir_path): 
        os.mkdir(dir_path,)
        os.mkdir(dir_path + "/data")
        os.mkdir(dir_path + "/trees")
    if os.path.isfile(dir_path + "/results.csv") and not overwrite:
        raise OSError("File exists")
    else:
        with open(dir_path + "/results.csv", "w") as file:
            file.write("fitness_score, population, generations, training_time\n")


    # Defining the data ================================
    vfunc = np.vectorize(func)
    X = np.linspace(x_range[0], x_range[1], n_points)
    y = vfunc(X)

    if normalize:
        scaler = MinMaxScaler(normalize_range)
        X = scaler.fit_transform(np.c_[X])
        y = scaler.fit_transform(np.c_[y]).reshape(-1, )

    
    # Training the model ===============================
    for i in range(n_runs):
        print(f"-=-=-=-=-=-=-=-= Training for population {population} and generation {generations} - {dir_path[dir_path.find('/')+1:]} =-=-=-=-=-=-=-=-")
        SR = SymbolicRegression(generations, max_expression_size, max_population_size=population,
                                max_island_count=int(population/10), random_const_range=const_range)
        SR.fit(np.c_[X], y, feature_names=["x"])
        
        start_time = time()
        output_AEG = SR.predict()
        end_time = time()
        data = SR.evaluate_tree(output_AEG.sexp)
        
        print(f"-=-=-=-=-=-=-= Done training for population {population} and generation {generations} - {dir_path[dir_path.find('/')+1:]} =-=-=-=-=-=-=-")

        # Writing the data =================================

        # In case the output is a constant function
        if data.shape[0] == 1:
            data = np.array([data[0] for i in range(0, 1000)])

        data = pd.DataFrame(np.c_[X, data], columns=["x", "y"])
        data.to_csv(dir_path + f"/data/data-{population}.csv", sep=",", index=False)

        graph = output_AEG.sexp.visualize_tree()
        graph.render(dir_path + f"/trees/tree-{population}", format="svg")

        with open(dir_path + "/results.csv", "a") as file:
            file.write(f"{SR.fitness_score(output_AEG)}, {population}, {generations}, {end_time - start_time}\n")
    
