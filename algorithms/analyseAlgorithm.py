from CSOWP_SR import *
from ExpressionTree import *
from trainAlgorithm import *
import matplotlib.pyplot as plt
import pandas as pd
import graphviz
from re import sub
import numpy as np
from scipy.optimize import curve_fit

plt.rcParams.update({"text.usetex": True})

class Plotter():
    
    def __init__(self, original_func, real_func_string=None, set_seaborn=False):
        self.original_func = original_func
        
        if real_func_string is None:
            self.real_func_string = "Real"
        else:
            self.real_func_string = fr"Real - ${real_func_string}$"
        
        if set_seaborn:
            import seaborn as sns
            sns.set()
            
            
    def getSmpTreeGraph(self, real=False):
    
        if real:
            sexp = self.real_sexp
        else:
            sexp = self.sexp
    
        for i in smp.preorder_traversal(sexp):
            if isinstance(i, smp.Float):
                sexp = sexp.subs(i, round(i, 2))

        s = smp.printing.dot.dotprint(sexp).replace("Add", "+").replace("Mul", "*")
        s = sub('0+(?=")', "0", s)

        if self.img_path is None:
            s = graphviz.Source(s)
        else:
            s = graphviz.Source(s, filename=self.img_path+".gv", format="png")
            s.render()

        return s

    def convert_smp_to_latex(self, sexp, floating_points=2):
        if isinstance(sexp, str):
            return sexp
        
        plot_sexp = sexp
        for i in smp.preorder_traversal(plot_sexp):
                if isinstance(i, smp.Float):
                    plot_sexp = plot_sexp.subs(i, round(i, floating_points))
        
        pred_label = smp.latex(plot_sexp)
        pred_label = sub("operatorname", "mathrm", pred_label)
        pred_label = fr"Predicted - ${pred_label}$"
        return pred_label
        
    def fit(self, dir_path, X, y, operators, functions, feature_names, cleaned_data="results.csv"):
        self.dir_path = dir_path
        self.X = X
        self.y = y
        self.operators = operators
        self.functions = functions
        self.feature_names = feature_names
        self.cleaned_data = cleaned_data
        self.title = dir_path[dir_path.find("/")+1:]
        
        #Continue from here
        data = pd.read_csv(dir_path + "/" + cleaned_data)
        self.best_tree = data.sort_values("fitness_score", axis=0)
        self.best_tree = self.best_tree.reset_index(drop=True)

        self.best_population = str(self.best_tree.population[0])
        self.best_generation = str(self.best_tree.generations[0])
        self.best_time = self.best_tree.training_time[0]/60.0
        self.best_fitness = self.best_tree.fitness_score[0]
        self.best_i_run = str(self.best_tree.i_run[0])

        self.img_path = self.dir_path + "/trees/tree-" + self.best_population + "-" + self.best_generation + "-" + self.best_i_run

        with open(self.img_path, "rb") as file:
            self.best_tree = pickle.load(file)

        self.func, self.sexp, self.real_sexp = self.best_tree.toFunc(operators, functions, feature_names,  inv_data = {
            "Xmin": self.X.min(),
            "Xmax": self.X.max(),
            "ymin": self.y.min(),
            "ymax": self.y.max()
        })
        
        if len(self.sexp.free_symbols) == 0:
            value = self.func(0)
            def func(X):
                try:
                    return [value for _ in X]
                except:
                    return value
            self.func = func

        self.pred_label = self.convert_smp_to_latex(self.sexp)
        

        self.x_range = self.X.max() - self.X.min()
        self.y_range = self.y.max() - self.y.min()
        
    
    
    
    def least_squares(self, y_pred):
        return np.mean( (self.y-y_pred)**2 )


    def plot_pred_graph(self, real=False):
        self.getSmpTreeGraph(real=real)
        
        fig, axs = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [4, 1]})


        ax=axs[0]

        ax.set_title(r"Real Function X Predicted Function - " + f"{self.title}", fontsize=14)
        ax.set_ylabel("y")
        ax.set_xlabel("x")

        ax.set_ylim(self.y.min(), self.y.max())

        ax.plot(self.X, self.y, label=self.real_func_string) 
        ax.plot(self.X, self.func(self.X), label=self.pred_label)
        ax.legend(fontsize=14, loc="upper left")


        ax = axs[1] 
        ax.annotate(f"""Training Time: {self.best_time:.2f} minutes
            Normalized Fitness Score: {self.best_fitness:.2E}
            Real Fitness Score: {self.least_squares(self.func(self.X)):.2E}
            Population Size: {self.best_population}""", 
                (-self.x_range/self.X.max()/3, ax.get_ylim()[1]-1.7/self.y_range), bbox={"facecolor": "lightgray", "edgecolor": "black"},
                linespacing = 1.5, annotation_clip=False)

        img = plt.imread(self.img_path+".gv.png")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_facecolor((1,1,1))
        ins = ax.inset_axes([-self.x_range/self.X.max()/5, 0, 0.8, 0.8])
        ins.imshow(img)
        ins.get_xaxis().set_visible(False)
        ins.get_yaxis().set_visible(False)
        ins.set_title(f"Output Expression")
        plt.show()
        
    def plot_expanded_domain(self, x_lim,n_points, y_lim=None):
        X = np.linspace(x_lim[0], x_lim[1], n_points)
        y = self.original_func(X)

        plt.figure(figsize=(12,6))
        plt.title(r"Expanded Domain Comparison - " + f"{self.title}", fontsize=14)
        plt.ylabel("y")
        plt.xlabel("x")

        if y_lim:
            plt.ylim(y_lim)
        else:
            plt.ylim(y.min(), y.max())

        plt.plot(X, y, label=self.real_func_string) 
        plt.plot(X, self.func(X), label=self.pred_label)
        plt.legend(fontsize=14, loc="upper left")
        
        plt.show()
    
    def least_squares_expression(self, guess=None, max_iter=10000, mode="real", symb="x", functions=None):
        def find_exp(string):
            original_string = str(string)
            indexes = []
            values = []
            ant=0
            while True:
                ind = string.find("^")
                if ind == -1: break
                if (string[ind+1] == "{" and string[ind+3] == "}"):
                    indexes.append(ind+ant+2)
                string = string[ind+1:]
                ant = ind+ant+1
            
            for i in range(0, len(indexes)):
                values.append(int(original_string[indexes[i]]))
                
            return values

        if mode == "real":
            sexp = self.real_sexp
        elif mode == "normalized":
            sexp = self.sexp
        elif mode == "overfitted":
            sexp = self.real_sexp
            if not functions:
                raise TypeError("Must pass parameter functions if chosen overfitted exp")
                
            variables = symb.split(", ")
            
            for i in smp.preorder_traversal(sexp):
                if (str(type(i)) in functions) or str(i) in variables:
                    
                    sexp = sexp.replace(i, np.random.rand()*i + np.random.rand())
        else:
            raise TypeError("Invalid option of exp")

        if len(sexp.free_symbols) == 0:
            value = self.func(0)
            def func(X):
                try:
                    return [value for _ in X]
                except:
                    return value
            return func, sexp

        string = self.convert_smp_to_latex(sexp)
        black_list = find_exp(string)
    
        count=0
        symbols = [smp.symbols(symb)]
        for i in smp.preorder_traversal(sexp):
            if isinstance(i, smp.Float) or (isinstance(i, smp.Integer) and str(i) != "-1") and (i not in black_list):
                if float(i) > 1/1e6:
                    symbols.append(smp.symbols(f"a_{count}", Real=True))
                    sexp = sexp.subs(i, symbols[-1])
                    count += 1
                
        if count == 0:
            return smp.lambdify(symbols[0], self.sexp), self.sexp
        
        # Inverse trasnform for real sexp
        if mode == "real":
            sexp = (self.y.max() - self.y.min())*sexp.subs(symbols[0], (symbols[0] - self.X.min())/(self.X.max() - self.X.min()) ) + self.y.min()

        func = smp.lambdify(symbols, sexp)
        
        if not guess:
            guess = [1 for i in range(len(symbols)-len(symb))]
        
        try:
            params, covariance = curve_fit(func, self.X, self.y, guess, maxfev=max_iter)
        except RuntimeError:
            print("Optimal parameters not found")
            return None, None
        
        for i in range(len(symb), len(symbols)):
            sexp = sexp.subs(symbols[i], params[i-1])
        
        func = smp.lambdify(symbols[0], sexp)
        
        return func, sexp
    

    def plot_least_squares(self, guess=None, max_iter=10000, exp="real", functions=None, expanded_domain=False, x_lim=None, n_points=None, ylim=None):
        real_least_func, real_least_sexp = self.least_squares_expression(guess, max_iter, mode="real")
        norm_least_func, norm_least_sexp = self.least_squares_expression(guess, max_iter, mode="normalized")
        over_least_func, over_least_sexp = self.least_squares_expression(guess, max_iter, mode="overfitted", functions=functions)
        
        if(real_least_func is None):
            real_least_func = lambda x:np.zeros(x.shape)
            real_least_sexp = "Least squares did not converge"
        if(norm_least_func is None):
            norm_least_func = lambda x:np.zeros(x.shape)
            norm_least_sexp = "Least squares did not converge"
        if(over_least_func is None):
            over_least_func = lambda x:np.zeros(x.shape)
            over_least_sexp = "Least squares did not converge"
        
        if expanded_domain:
            if not x_lim or not n_points:
                raise TypeError("If expanded domain is True, must pass x_lim and n_points")
                
            X = np.linspace(x_lim[0], x_lim[1], n_points)
            y = self.original_func(X)
        else:
            X = self.X
            y = self.y
            
        
        fig, axs = plt.subplots(2, 2, constrained_layout=True, figsize=(14,12))
        
        if ylim:
            for ax in axs.reshape(-1, ):
                ax.set_ylim(ylim[0], ylim[1])
        
        ax=axs[0,0]
        ax.set_title("Predicted Function")
        ax.plot(X, y, label=self.real_func_string)
        ax.plot(X, self.func(X), label=self.pred_label)
        ax.legend()
        
        ax.set_ylim(y.min(), y.max())

        ax=axs[0,1]
        ax.set_title("Least Squares on Real Function")
        ax.plot(X, y, label=self.real_func_string)
        ax.plot(X, real_least_func(X), label=self.convert_smp_to_latex(real_least_sexp))
        ax.legend()
        
        ax.set_ylim(y.min(), y.max())

        ax=axs[1,0]
        ax.set_title("Least Squares on Normalized Function")
        ax.plot(X, y, label=self.real_func_string)
        ax.plot(X, norm_least_func(X), label=self.convert_smp_to_latex(norm_least_sexp))
        ax.legend()
        
        ax.set_ylim(y.min(), y.max())

        ax=axs[1,1]
        ax.set_title("Least Squares on Overfitted Function")
        ax.plot(X, y, label=self.real_func_string)
        ax.plot(X, over_least_func(X), label=self.convert_smp_to_latex(over_least_sexp))
        ax.legend()
        
        ax.set_ylim(y.min(), y.max())

        plt.show()
        
    def plot_diff_test(self, variable, guess=None, max_iter=10000, exp="real", functions=None):
        sexp = self.sexp
        real_least_func, real_least_sexp = self.least_squares_expression(guess, max_iter, mode="real")
        norm_least_func, norm_least_sexp = self.least_squares_expression(guess, max_iter, mode="normalized")
        over_least_func, over_least_sexp = self.least_squares_expression(guess, max_iter, mode="overfitted", functions=functions)
        
        x = smp.symbols(variable)        

        if(real_least_func is None):
            real_least_func = lambda x:np.zeros(x.shape)
            real_least_sexp = "Least squares did not converge"
        else:
            real_least_sexp = real_least_sexp.diff(x)
            real_least_func = smp.lambdify(x, real_least_sexp)
        if(norm_least_func is None):
            norm_least_func = lambda x:np.zeros(x.shape)
            norm_least_sexp = "Least squares did not converge"
        else:
            norm_least_sexp = norm_least_sexp.diff(x)
            norm_least_func = smp.lambdify(x, norm_least_sexp)
        if(over_least_func is None):
            over_least_func = lambda x:np.zeros(x.shape)
            over_least_sexp = "Least squares did not converge"
        else:
            over_least_sexp = over_least_sexp.diff(x)
            over_least_func = smp.lambdify(x, over_least_sexp)

        X = self.X
        y = self.y
        y = np.gradient(y, X)

        sexp = sexp.diff(x)
        func = smp.lambdify(x, sexp)

        if len(sexp.free_symbols) == 0:
            value = self.func(0)
            def func(X):
                return np.zeros(X.shape)
            real_least_func = func
            norm_least_func = func
            over_least_func = func
            
        
        fig, axs = plt.subplots(2, 2, constrained_layout=True, figsize=(14,12))
        
        plt.suptitle("Derivative Test")

        ax=axs[0,0]
        ax.set_title("Predicted Function")
        ax.plot(X, y, label=self.real_func_string)
        ax.plot(X, func(X), label=self.convert_smp_to_latex(func))
        
        ax.set_ylim(y.min(), y.max())
        
        ax=axs[0,1]
        ax.set_title("Least Squares on Real Function")
        ax.plot(X, y, label=self.real_func_string)
        ax.plot(X, real_least_func(X), label=self.convert_smp_to_latex(real_least_sexp))
        
        ax.set_ylim(y.min(), y.max())

        ax=axs[1,0]
        ax.set_title("Least Squares on Normalized Function")
        ax.plot(X, y, label=self.real_func_string)
        ax.plot(X, norm_least_func(X), label=self.convert_smp_to_latex(norm_least_sexp))

        ax.set_ylim(y.min(), y.max())

        ax=axs[1,1]
        ax.set_title("Least Squares on Overfitted Function")
        ax.plot(X, y, label=self.real_func_string)
        ax.plot(X, over_least_func(X), label=self.convert_smp_to_latex(over_least_sexp))

        ax.set_ylim(y.min(), y.max())

        plt.show()

def analyseData(func, x_range, n_points, dir_path, population, generations, max_expression_size, real_func_string, x_lim, 
                variables=["x"], normalize=False, const_range=None, ignore_warning=True, overwrite=False, n_runs=1, set_seaborn=True,
                n_expanded_points=None, real=True):
    
    if n_expanded_points is None:
        n_expanded_points = n_points
    
    X, y, operators, functions = testAlgorithm(func, x_range, n_points, dir_path, population, generations, 
                  max_expression_size, normalize=normalize, const_range=const_range,
                  ignore_warning=ignore_warning, overwrite=overwrite, n_runs=n_runs)
    
    plotter = Plotter(real_func_string=real_func_string, set_seaborn=set_seaborn, original_func=func)
    plotter.fit(dir_path, X, y, operators, functions, ["x"])
    plotter.plot_pred_graph(real=real)
    plotter.plot_expanded_domain([x_lim[0], x_lim[1]], n_points=n_expanded_points)
    plotter.plot_least_squares(functions=functions)
    plotter.plot_least_squares(functions=functions, expanded_domain=True, x_lim=[x_lim[0], x_lim[1]], n_points=n_expanded_points)
    plotter.plot_diff_test(variables[0], functions=functions)

def plotData(X, y, operators, functions, func, dir_path, real_func_string, x_lim, n_expanded_points, variables=["x"], 
             set_seaborn=True, real=True):
    plotter = Plotter(real_func_string=real_func_string, set_seaborn=set_seaborn, original_func=func)
    plotter.fit(dir_path, X, y, operators, functions, ["x"])
    plotter.plot_pred_graph(real=real)
    plotter.plot_expanded_domain([x_lim[0], x_lim[1]], n_points=n_expanded_points)
    plotter.plot_least_squares(functions=functions)
    plotter.plot_least_squares(functions=functions, expanded_domain=True, x_lim=[x_lim[0], x_lim[1]], n_points=n_expanded_points)
    plotter.plot_diff_test(variables[0], functions=functions)
        