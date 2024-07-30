from trainAlgorithm import *
from pathos.multiprocessing import ProcessingPool as Pool
import typing
from CSOWP_SR import *
from ExpressionTree import *

class trainSR():
    def __init__(self, dir_path, population, generations, max_expression_size=None,
                 normalize=False, const_range=(0,1), normalize_range=(0,1),
                 n_points=None, x_range=None, ignore_warning=True, overwrite=False, n_runs=1, operators=None,
                 functions=None, weights=None, island_interval=None,
                 optimization_kind="PSO", custom_functions_dict=None, SEED=None, 
                 gen_fit_path=None):
        
        self.SR_instances = ()

        self.dir_path = dir_path
        self.population = population
        self.generations = generations
        self.max_expression_size = max_expression_size
        self.normalize = normalize
        self.const_range = const_range
        self.normalize_range = normalize_range
        self.overwrite = overwrite
        self.n_runs = n_runs
        self.operators = operators
        self.functions = functions
        self.weights = weights
        self.island_interval = island_interval
        self.optimization_kind = optimization_kind
        self.custom_functions_dict = custom_functions_dict
        self.SEED = SEED
        self.gen_fit_path = gen_fit_path
        self.n_points = n_points
        self.x_range = x_range
        self.ignore_warning = ignore_warning
        

    def fit(self, file_name:List, func:List, x_range:List=None,
             n_points:List=None, info:List=None):
        """
            instances: List[Dict] = [ {"file_name": ..., "func": ...,}, ... ]
        """

        if x_range is None:
            x_range = [None for _ in file_name]
        if n_points is None:
            n_points = [None for _ in file_name]
        if info is None:
            info = [None for _ in file_name]

        instances = [ 
            {"file_name": file_name[i], "func": func[i], "x_range": x_range[i],
             "n_points": n_points[i], "info": info[i]} 
             for i in range(len(file_name))
         ]
        
        self.instances = instances

    def call_predict(self, SR:SymbolicRegression):
        return SR.predict()

    def runParallel(self, max_processes=8):
        n_processes = len(self.instances)
        if n_processes > max_processes: n_processes = max_processes
        # print(n_processes)
        # Every dict should be composed of {"feature_names": [options]}
        with Pool(processes=n_processes) as pool:
            results = pool.map(self.testAlgorithm, self.instances)
        return results
    
    def testAlgorithm(self, instances:Dict):

        # Filtering
        file_name = instances["file_name"]
        func = instances["func"]
        x_range = instances["x_range"]
        n_points = instances["n_points"]
        info = instances["info"]

        
        if self.ignore_warning:
            warnings.filterwarnings("ignore")
        
        if x_range is None:
            if self.x_range is None:
                raise RuntimeError("You must inform x_range either here or on class init")
            else:
                x_range = self.x_range
        if n_points is None:
            if self.n_points is None:
                raise RuntimeError("You must inform x_range either here or on class init")
            else:
                n_points = self.n_points
            
        if info is not None and type(info) != dict:
            raise TypeError("Info must be a dictionary.")

        # Initial Definitions ==============================
        file_path = os.path.join(self.dir_path, file_name)
        if not os.path.isdir(file_path): 
            os.mkdir(file_path,)
            os.mkdir(file_path + "/data")
            os.mkdir(file_path + "/trees")
        if os.path.isfile(file_path + "/results.csv") and not self.overwrite:
            raise OSError("File exists")
        else:
            with open(file_path + "/results.csv", "w") as file:
                file.write("MSE_error,population,generations,training_time,i_run\n")


        # Defining the data ================================
        # vfunc = np.vectorize(func)
        X = np.linspace(x_range[0], x_range[1], n_points)
        y = func(X)

        if self.normalize:
            originX = X
            originy = y
            scaler = MinMaxScaler(self.normalize_range)
            X = scaler.fit_transform(np.c_[X])
            y = scaler.fit_transform(np.c_[y]).reshape(-1, )
        else:
            originX = X
            originy = y
        
        np.random.seed(self.SEED)
        seed(self.SEED)

        # Training the model ===============================
        for i in range(self.n_runs):
            print(f"-=-=-=-=-=-=-=-= Training for population {self.population} and generation {self.generations} - {file_path[file_path.find('/')+1:]} =-=-=-=-=-=-=-=-")
            SR = SymbolicRegression(self.generations, self.max_expression_size, max_population_size=self.population,
                                    max_island_count=int(self.population/10), random_const_range=self.const_range,
                                    operators=self.operators, functions=self.functions, weights=self.weights,
                                    island_interval=self.island_interval, optimization_kind=self.optimization_kind,
                                    custom_functions_dict=self.custom_functions_dict)
            SR.fit(np.c_[X], y, feature_names=["x"])    
            
            start_time = time()
            if self.gen_fit_path is not None:
                output_AEG = SR.predict(gen_fit_path=f"{self.gen_fit_path}/gen_fit-{self.population}-{self.generations}-{i}")
            else:
                output_AEG = SR.predict()
            end_time = time()
            data = SR.evaluate_tree(output_AEG.sexp)
            
            print(f"-=-=-=-=-=-=-= Done training for population {self.population} and generation {self.generations} - {file_path[file_path.find('/')+1:]} =-=-=-=-=-=-=-")

            # Writing the data =================================

            # In case the output is a constant function
            # if data.shape[0] == 1:
            #     data = np.array([data[0] for i in range(0, 1000)])

            # data = pd.DataFrame(np.c_[X, data], columns=["x", "y"])
            # data.to_csv(dir_path + f"/data/data-{population}.csv", sep=",", index=False)

            # graph = output_AEG.sexp.visualize_tree()
            # graph.render(dir_path + f"/trees/tree-{population}", format="svg")

            with open(file_path + f"/trees/tree-{self.population}-{self.generations}-{i}", "wb") as file:
                pickle.dump(output_AEG.sexp, file)

            with open(file_path + f"/results.csv", "a") as file:
                file.write(f"{SR.fitness_score(output_AEG)},{self.population},{self.generations},{end_time - start_time},{i}\n")
            
            if info is not None:
                with open(file_path + "/info.csv", "w") as file:
                    for item in info.items():
                        file.write(f"{item[0]}, {item[1]}\n")

        
        return originX, originy, SR._operators, SR._functions   
