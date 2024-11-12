from CSOWP_SR import *
from ExpressionTree import *

from typing import Callable, Tuple
from random import seed
import warnings
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns


class rollingWindow():

    def __init__(self, G: int = 3, maxPop: int = 100, SEED: int = 42, ignore_warning=False,
                  const_range=(0,1), operators=None, functions=None, verbose=False,
                  weights=None, island_interval=None, optimization_kind="LS", return_func=False,
                  custom_functions_dict=None, feature_names=["x"], dir_path=None):
        
        self.SEED = SEED
        self.ignore_warning = ignore_warning
        self.verbose = verbose
        self.return_func = return_func
        self.feature_names = feature_names
        self.functions = None
        self.dir_path = dir_path

        self.SR = SymbolicRegression(G=G, max_population_size=maxPop,
                            random_const_range=const_range, operators=operators, functions=functions,
                            weights=weights, island_interval=island_interval, optimization_kind=optimization_kind,
                            custom_functions_dict=custom_functions_dict)
        
        if (self.dir_path is not None) and (not os.path.isdir(self.dir_path)):
            print(f"Created missing directorie(s): {self.dir_path}")
            os.makedirs(self.dir_path)


    def fit(self, X: np.ndarray, y: np.ndarray, x_range:Tuple[float, float]=None,
             L: float = None, nPics: int = None):
        
        self.X = X
        self.y = y

        if x_range is None:
            self.x_range = (X.min(), X.max())
        else:
            self.x_range = x_range

        self.a = self.x_range[0]
        self.b = self.x_range[1]
        self.L = L
        self.nPics = nPics
        
        if L is None and nPics is None:
            raise ValueError("L and nPics can't be simultaneously None, one or both must be informed.")
        elif L is None:
            self.L = (self.b-self.a)/nPics
        elif nPics is None:
            self.nPics = (self.b-self.a)/L

        self.nPics = int(self.nPics)
        self.stepSize = (self.b - self.a)/self.nPics


    def _filter(self, step):
        XStep = self.X[(self.X >= step) & (self.X < step + self.L)]
        yStep = self.y[(self.X >= step) & (self.X < step + self.L)]
        return XStep, yStep
    
    def _do(self, X, y, trees, step):
        self.SR.fit(np.c_[X], y, feature_names=self.feature_names)
        outputAEG = self.SR.predict()
        trees[f"({step}, {step+self.L})"] = [outputAEG.sexp, self.SR.fitness_score(outputAEG)]
        return trees

    def run(self):
        if self.ignore_warning:
            warnings.filterwarnings("ignore")
        
        np.random.seed(self.SEED)
        seed(self.SEED)

        end = self.b-self.stepSize
        step = self.a

        trees = {}
        
        # Rolling Window
        XStep, yStep = self._filter(step)
        trees = self._do(XStep, yStep, trees, step)
        step += self.stepSize
        while step < end:
            XStep, yStep = self._filter(step)
            trees = self._do(XStep, yStep, trees, step)
            step += self.stepSize
        XStep, yStep = self._filter(step)
        trees = self._do(XStep, yStep, trees, step)
        


        # Return
        fTrees = []
        for i in trees.values():
            fTrees.append(i[0])
        aFuncs = [self.SR.toFunc(tree) for tree in fTrees]

        self.functions = aFuncs

        if self.verbose:
            to_return = trees
        else:
            to_return = tuple(trees.values())
        

        # Export result to dir
        if self.dir_path is not None:
            for c, tree in enumerate(to_return):
                tree = tree[0]
                
                file_name = f"tree_{self.a + c*self.stepSize}-{self.a+(c+1)*self.stepSize}"
                path = os.path.join(self.dir_path, file_name)
                with open(path, "wb") as file:
                    pickle.dump(tree, file)


        if self.return_func:
            return to_return, aFuncs
        else:
            return to_return
        
        
        
    def multi_plots(self, x_range, n_points=1000):
        if self.functions is None:
            raise RuntimeError("You must first execute the algorithm with run and after call multi_plots")
        
        X = np.linspace(x_range[0], x_range[1], n_points)
        
        for c, func in enumerate(self.functions):
            y = func(X)
            plt.plot(X, y, label=c)
        plt.legend()
        plt.show()

    def visualize(self, bg_palette="inferno", bg_alpha=0.2,
                  bg_linecolor="black", bg_linewidth=1,
                  linecolor="black", linewidth=3,
                  linestyle="dashed"):
        ax = plt.gca()
        ax.plot(self.X, self.y, linewidth=linewidth, 
                c=linecolor, linestyle=linestyle)
        ax.set_xlim(self.X.min(), self.X.max())
        ax.set_ylim(self.y.min(), self.y.max())

        color_palette = sns.color_palette(bg_palette, self.nPics)

        Xmin = self.X.min()
        ymin = self.y.min()
        ymax = abs(self.y.min() - self.y.max())

        # Add a rectangle with a background color
        for c in range(self.nPics):
            x_start = Xmin + c*self.stepSize

            rect = patches.Rectangle((x_start, ymin), self.L, ymax,
                                      linewidth=bg_linewidth, edgecolor=bg_linecolor,
                                        facecolor=color_palette[c], alpha=bg_alpha)
            ax.add_patch(rect)

        # Display the plot
        plt.show()