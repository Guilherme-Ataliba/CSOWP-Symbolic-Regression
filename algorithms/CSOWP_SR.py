from functools import singledispatchmethod
from typing import Any, List

import pandas as pd
import numpy as np
from random import randint, choice, uniform, seed
from ExpressionTree import *
from scipy.optimize import curve_fit, differential_evolution, dual_annealing
from copy import deepcopy
import pickle
import pyswarms

class Particle():
    "v: velocity vector"
    __slots__ = ("vector", "velocity", "best")
    def __init__(self, vector: np.ndarray = np.array([]), velocity: np.ndarray = None, best: np.ndarray = None):
        self.vector = vector
        self.velocity = velocity
        self.best = best
        
    def copy_particle(self):
        new_particle = Particle()
        new_particle.vector = np.copy(self.vector)
        new_particle.velocity = np.copy(self.velocity)
        new_particle.best = np.copy(self.best)
        return new_particle
    

class AEG():
    """aexp: Abstract Expression, where the constants are substituted by reference to the c vector
    sexp: The original Expression Tree, with no abstract constant
    c: Particle
    pool: List of particles
    
    Particle's a class (structure) that contains a vector of real constants and a velocity vector"""

    __slots__ = ("aexp", "sexp", "c", "pool")
    def __init__(self, aexp: ExpressionTree = None, sexp: ExpressionTree = None, c: Particle = None, pool: List[Particle] = None):
        self.aexp = aexp
        self.sexp = sexp
        self.c = c
        self.pool = pool     
        
    def _print_out(self):
        print(f"""aexp: {self.aexp.visualize_tree()}
        sexp: {self.sexp.visualize_tree()}
        c: {self.c}
        pool: {self.pool}""")
    
    def copy_AEG(self):
        new_AEG = AEG(pool = [])
        
        new_AEG.aexp = self.aexp.copy_tree(self.aexp.root())
        new_AEG.sexp = self.sexp.copy_tree(self.sexp.root())
        new_AEG.c = self.c.copy_particle()
        
        for particle in self.pool:
            new_AEG.pool.append(particle.copy_particle())
        
        return new_AEG
    
    def toFunc(self, operators, functions, features, custom_functions_dict={}):
        expr_string = self.aexp.toString(operators, functions, custom_functions_dict=custom_functions_dict)
        expr_string = expr_string.replace("[", "").replace("]", "")

        if len(self.pool) > 0:
            n_params = len(self.pool[0].vector)
            
            params_dict = {}
            for i in features:
                feature = smp.symbols(f"{i}")
                params_dict[f"{i}"] = feature
            
            for i in range(n_params):
                feature = smp.symbols(f"c{i}")
                params_dict[f"c{i}"] = feature

            # print(expr_string)
            # print(params_dict)
            smp_expr = smp.sympify(expr_string, locals=params_dict)
        else:
            smp_expr = smp.sympify(expr_string)

        

        symbols_list = params_dict.keys()

        symbols_string = ""
        for i in symbols_list:
            symbols_string += f"{i}, "

        symbols = smp.symbols(symbols_string)
        
        return smp.lambdify(symbols, smp_expr)


class SymbolicRegression():
    """X: N vector of independent M-featured training points
    Y: N vector f dependent variables
    G: Number of generations to train
    Output: Champion s-expression individual
    Parameters: maxPopSize MasIslandCount
    
    Summary: Brute force elitist GP searches for a champion s-expression by randomly growing and scoring a
                large number of candidate s-expressions, then iteratively creating and scoring new candidate
                s-expressions via mutation and crossover. After each iteration, the population of candidate
                s-expressions is truncated to those with the best fitness score. After the final iteration, the
                champion is the s-expression with the best fitness score"""
    
    __slots__ = ("X", "y", "G", "_feature_names", "label_name", "max_population_size", "max_expression_size",
                "_operators", "_functions", "_options", "_operators_func", "_functions_func", "_features", 
                "max_island_count", "max_island_size", "_weights", "max_pool_size", "random_const_range",
                "_mult_tree", "_add_tree", "_linear_tree", "island_interval", "optimization_kind", "custom_functions_dict")
    def __init__(self, G, feature_names=None, label_name="y", max_population_size=5000, max_expression_size = 5, max_island_count=None, 
                max_island_size=None, max_pool_size = 15, random_const_range=(0,1), operators=None, functions=None, weights=None,
                island_interval=None, optimization_kind="PSO", custom_functions_dict=None):
        """
            - feature_names: A list containing the names of every feature in X
            - island_interval: (islands bellow the current one, islands above the current one)
        """
        
        self.y = None
        self.G = G
        
        self.max_population_size = max_population_size
        self.max_pool_size = max_pool_size
        self.max_expression_size = max_expression_size
        self.optimization_kind = optimization_kind
        

        if max_island_count is None:
            self.max_island_count = int(max_population_size/10)
        else:
            self.max_island_count = max_island_count

        if max_island_size is None:
            self.max_island_size = int(max_population_size / self.max_island_count)
        else:
            self.max_island_size = max_island_size
        
        self.random_const_range = random_const_range

        if island_interval == None:
            self.island_interval = (1,0)
        else:
            self.island_interval = island_interval
        
        
        """ I've chosen to let _operatos and _functions here to reduce call 
            of list() and .keys() from _operators_func and _functions_func 
        having efficiency in mind """
        
        # Operators
        if operators is None:
            self._operators = ["+", "-", "*", "/"] 
            self._operators_func = {"+": lambda a,b: np.add(a,b), "-": lambda a,b: np.subtract(a,b),
                                "*": lambda a,b: np.multiply(a, b), "/": lambda a,b: np.divide(a,b)}
        else:
            self._operators_func = operators
            self._operators = list(operators.keys())

        # Functions
        if functions is None:
            self._functions = ["abs", "square", "cos", "sin",
                        "tan", "tanh", "exp", "sqrt", "log"] # "max", "min"
            self._functions_func = {"abs": lambda a: np.abs(a), "exp": lambda a: np.exp(a), "square": lambda a: a**2,
                            "cos": lambda a: np.cos(a),
                            "sin": lambda a: np.sin(a), "tan": lambda a: np.tan(a), "tanh": lambda a: np.tanh(a),
                            "sqrt": lambda a: np.sqrt(a), "log": lambda a: np.log(a)}
        else:
            self._functions_func = functions
            self._functions = list(functions.keys())

            
        self._options = {"operator": lambda: choice(self._operators), "function": lambda: choice(self._functions),
                        "feature": lambda: choice(self._feature_names),
                        "constant": lambda: round(uniform(self.random_const_range[0],self.random_const_range[1]), 3)}
        
        
    
        self._weights = {
            "+": 1, "-": 1, "*": 2, "/": 2,
            "sqrt": 3, "square": 2, "cube": 2, "quart": 2,
            "log": 4, "exp": 4, "cos": 5, "sin": 5, 
            "tan": 6, "tanh": 6, "abs": 1
        }
        if weights is not None:
            for i, j in weights.items():
                self._weights[i] = j

        # Function Dictionary
        self.custom_functions_dict = {"sin": ["np.sin(",")"], "cos": ["np.cos(",")"],
                                      "abs": ["np.abs(", ")"], "square": ["(", ")**2"],
                                      "tan": ["np.tan(", ")"], "tanh": ["np.tanh(", ")"],
                                      "exp": ["np.exp(", ")"], "sqrt": ["np.sqrt(", ")"],
                                      "log": ["np.log(", ")"]
                                      }
        if custom_functions_dict is not None:
            self.custom_functions_dict.update(custom_functions_dict)

        # Linear Transform Trees
        self._mult_tree = ExpressionTree()
        p = self._mult_tree.add_root("*", e_type="operator")
        self._mult_tree.add_left(p, "a", e_type="constant")
        self._mult_tree.add_right(p, "x", e_type="feature")

        self._add_tree = ExpressionTree()
        p = self._add_tree.add_root("+", e_type="operator")
        self._add_tree.add_left(p, "a", e_type="constant")
        self._add_tree.add_right(p, "x", e_type="feature")

        self._linear_tree = ExpressionTree()
        p = self._linear_tree.add_root("+", e_type="operator")
        self._linear_tree.add_right(p, "a", e_type="constant")
        p = self._linear_tree.add_left(p, "*", e_type="operator")
        self._linear_tree.add_left(p, "b", e_type="constant")
        self._linear_tree.add_right(p, "x", e_type="feature")

        
    def fit(self, X, y, feature_names=None, label_name="y"):
        if type(X) != np.ndarray:
            raise TypeError("X must be an array")
    
        self.y = y
        self.label_name = label_name
        
        if feature_names == None:
            # This is necessary for a one dimensional array
            if X.shape[0] == 1:
                self._feature_names = ["x0"]
            else:
                self._feature_names = ["x" + f"{i}" for i in range(0, X.shape[1])]
        else:
            self._feature_names = feature_names
            
        self.label_name = label_name
        
        # self._features is used in place of X, to express the data points
        self._features = {}
        for c, name in enumerate(self._feature_names):
            self._features[name] = X[:, c]
    
    
    def _generate_placeholder_tree(self, size):
        """Generates a placeholder tree that later will be randomly filled to create a random expression, respecting
        the operations."""
        tree = ExpressionTree()
        
        p = tree.add_root(0)
        
        def auxilary_generator(size, p):
            if size <= 0:
                return None            

            left_size = randint(0, size-1) 
            right_size = size - left_size - 1

            # Sometimes skips the left node - thus making possible to generate functions
            if randint(0,1): 
                left_p = tree.add_left(p, 0)
                auxilary_generator(left_size, left_p)
                
            right_p = tree.add_right(p, 0)
            auxilary_generator(right_size, right_p)
        
        auxilary_generator(size, p)
        
        return tree
    
    def generate_expr(self, size=None):
        """Creates and returns a random expression tree of a given size."""
        if size == None:
            size = self.max_expression_size
        tree = self._generate_placeholder_tree(size)
        
        
        for p in tree.inorder():    
            if tree.is_leaf(p):
                if randint(0,1):
                    tree.replace(p, choice(self._feature_names), "feature")
                else:
                    tree.replace(p, round(uniform(self.random_const_range[0], self.random_const_range[1]), 3), "constant")
            elif tree.num_children(p) > 1:
                tree.replace(p, choice(self._operators), "operator")
            elif tree.num_children(p) == 1:
                tree.replace(p, choice(self._functions), "function")
                
        return tree
    
    # def evaluate_tree(self, tree):     
    #     saida = np.array([])
        
        
    #     previous_left_value = False
            
    #     # Calcula o valor da árvore para um x
    #     for p in tree.postorder():
    #         num_children = tree.num_children(p)
    #         if num_children == 2: # é operador
    #             left_value = self._operators_func[p.element()](left_value, right_value)
    #             previous_left_value = True

    #         elif num_children == 1: # é função
    #             if previous_left_value:
    #                 left_value = self._functions_func[p.element()](left_value)
    #             else:
    #                 right_value = self._functions_func[p.element()](right_value)

    #         else: # é constante ou feature
    #             if type(p.element()) != str: #é constante
    #                 element = p.element()
    #             else: # é feature
    #                 element = self._features[p.element()]

    #             if previous_left_value:
    #                 right_value = element
    #                 previous_left_value = False
    #             else:
    #                 left_value = element
    #                 previous_left_value = True
    #     saida = np.append(saida, left_value)
            
    #     return saida

    @singledispatchmethod
    def toFunc(self, individual: Any):
        raise(f"type {type(individual)} is not valid")
    
    @toFunc.register
    def _(self, individual: ExpressionTree):
        func_string = individual.toString(self._operators, self._functions, self.custom_functions_dict)
        
        #Dealing with abstract constants
        func_string = func_string.replace("[", "").replace("]", "") 
        
        features = ""
        for i in range(len(self._feature_names)-1):
            features += self._feature_names[i]
            features += ", "
        features += self._feature_names[-1]
                

        func = eval(f"lambda {features}: {func_string}")
        return func
    
    @toFunc.register
    def _(self, individual: AEG):
        func_string = individual.aexp.toString(self._operators, self._functions, self.custom_functions_dict)
        
        #Dealing with abstract constants
        func_string = func_string.replace("[", "").replace("]", "") 
        
        features = ""
        for i in range(len(self._feature_names)-1):
            features += self._feature_names[i]
            features += ", "
        features += self._feature_names[-1]

        n_const = len(individual.c.vector)
        if n_const > 0:
            additional_features = [f"c{i}" for i in range(n_const)]
            features += ", "
            for i in range(len(additional_features)-1):
                features += additional_features[i]
                features += ", "
            features += additional_features[-1]
                

        func = eval(f"lambda {features}: {func_string}")
        return func

    def evaluate_tree(self, tree):
        """I could probably create a version that stores all created functions
        in a dictionary and then just access this dictionary, instead of creating
        the function every time. But i'd need a way to erase functions of individuals
        that are no longer in the population"""

        func = self.toFunc(tree)
        features = np.array(list(self._features.values()))
        return func(*features)

    @singledispatchmethod
    def fitness_score(self, individual: Any, custom_func = None):
        raise(f"type {individual} is not valid")
    
    @fitness_score.register
    def _(self, individual: ExpressionTree, custom_func=None):
        if not custom_func: # mean squared error
            def custom_func(y, y_pred):
                return np.mean((y - y_pred)**2)
            
        predicted = self.evaluate_tree(individual)
        return custom_func(self.y, predicted)
    
    def sort_tree_array(self, array):
        def get_fitness(tree):
            return tree.fitness_score
        
        vectorized_function = np.vectorize(get_fitness)
        
        fitness_array = vectorized_function(array)
        fitness_array = np.argsort(fitness_array)
        return array[fitness_array]
    
    
    # ------------------ Constant Optimization --------------
    
    def sort_AEG_array(self, array: np.ndarray) -> np.ndarray:
        def get_fitness(tree):
            return self.fitness_score(tree.sexp)
        
        vectorized_function = np.vectorize(get_fitness)
        
        fitness_array = vectorized_function(array)
        
        for c,i in enumerate(fitness_array):
            if i == None:
                for element in array:
                    fitness = self.fitness_score(element)
                    if fitness == None:
                        print(f"None: {self.fitness_score(element)}", sep=" ")
                        
        fitness_array = np.argsort(fitness_array)
        return array[fitness_array]
    
    @fitness_score.register
    def _(self, individual: AEG, custom_func = None):
        if not custom_func: # mean squared error
            def custom_func(y, y_pred):
                return np.mean((y - y_pred)**2)
            
        tree = self._convert_to_ExpTree(individual)
            
        predicted = self.evaluate_tree(tree)
        return custom_func(self.y, predicted)
    
    def _convert_to_AEG(self, me: ExpressionTree) -> AEG:
        """
        Input: me (tree)
        Output: out (AEG annotated individual)
        Summary: Converts an expression tree into an AEG individual. AEG conversion removes 
        all of the constants from an input expression tree and places them in a vector where 
        swarm intelligence algorithms can easily optimize them. The output is a constant 
        vector and the original expression tree modified to refer indirectly into the 
        constant vector instead of referencing the constants directly. 
        """
        
        if not isinstance(me, ExpressionTree):
            raise TypeError(f"Type {type(me)} not accept, it should only be an ExpressionTree")
        
        out = AEG(me.copy_tree(me.root()), me.copy_tree(me.root()), Particle(), [])
        N = len(out.aexp)
        
        for i in out.aexp.inorder():
            # print("In converto to AGE:")
            # display(me.visualize_tree())
            # for i in me.inorder():
            #     print(f"Element: {i.element()}, element_type: {i.element_type()}")
            if i.element_type() == "constant":
                r = i.element()
                k = len(out.c.vector)
                out.c.vector = np.append(out.c.vector, r)
                i.Node._element = f"c[{k}]"
                i.Node._element_type = "absConstant"
        
        out.pool.append(out.c)
        return out
    
    def _convert_to_ExpTree(self, me: AEG) -> ExpressionTree:
        """
        Input: me // AEG formatted individual
        Output: out // Koza-style s-expression individual
        Summary: Converts an AEG formatted individual into an s-expression individual.
        All AEG constant vector references, like "c[k]", are replaced with the actual
        constant values in the constant vector.
        AEG formatted individuals are structured as: <aexp, sexp, c, pool>
        """
        
        out = me.aexp.copy_tree(me.aexp.root())
        N = len(out)
        k = 0
        
#         print("---- Convert To Expression Tree ------")
#         display(out.visualize_tree())
#         for i in out.inorder():
#             print(f"element: {i.element()}, element_type: {i.element_type()}")
#         print(f"me vector: {me.c.vector}")
        
        
        for i in out.inorder():
            if i.Node._element_type == "absConstant":
                # display(me.aexp.visualize_tree())
                # print(f"me.c: {me.c}")
                # print(f"me.c.vector: {me.c.vector}")
                try: 
                    r = me.c.vector[k]
                except:
                    # display(me.aexp.visualize_tree())
                    # print(me.c)
                    raise(TypeError("ERRO"))
                i.Node._element = r
                i.Node._element_type = "constant"
                k += 1
                
            # print("got out")
        return out
    
    def sort_pool_array(self, lamb: AEG):
        """!!! REALLY INNEFICIENT, should create all from the ground up to the AEG
        Convert back and forth is inneficient"""
        iterated_AEG = AEG(aexp = lamb.aexp)
        trees_array = np.array([])
        AEG_array = np.array([])
        pool = []

        
        # print("----- sort pool -------")
        # display(lamb.aexp.visualize_tree())
#         for i in lamb.pool:
#             print(i.vector)
        
        # print(f"pool: {lamb.pool}")
        for particle in lamb.pool:
            # print("----- sort pool -------")
            # display(lamb.aexp.visualize_tree())
            # for i in lamb.aexp.inorder():
            #     print(f"Element: {i.element()}, element_type: {i.element_type()}")
            # print(f"vector: {particle.vector}, whole pool: {lamb.pool}" )
            iterated_AEG.c = particle
            iterated_tree = self._convert_to_ExpTree(iterated_AEG)
            iterated_tree.fitness_score = self.fitness_score(iterated_tree)
            trees_array = np.append(trees_array, iterated_tree)

        sorted_tree_array = self.sort_tree_array(trees_array)

        for tree in sorted_tree_array:
            # display(tree.visualize_tree())
            # for i in tree.inorder():
            #     print(f"Element: {i.element()}, element_type: {i.element_type()}")
            iterated_AEG = self._convert_to_AEG(tree)
            # print(f"new_AEG vector: {iterated_AEG.c.vector}")
            AEG_array = np.append(AEG_array, iterated_AEG)
            
        for individual in AEG_array:
            pool.append(individual.c)

#         print("----- sort pool out -------")
#         for i in pool:
#             print(i.vector)        
        return pool

                

    def insertLambda(self, population: np.ndarray, lamb: AEG):
        """
        Summary: Accepts an input individual (lamb) and converts it into 
        AEG format. It then searches the population of AEG individuals 
        for a constant homeomorphic AEG (an AEG with matching form and
        constant locations although the value of the constants may differ).
        If a constant homeomorphic AEG is found, the input lambda is merged
        with the existing AEG version already in the population; otherwise,
        The input lambda is inserted in at the end of the population
        """
        
        # print("--- IN insert lambda ---")
        # display(lamb.aexp.visualize_tree())
        # for i in lamb.pool:
        #     print(i.vector)
        if len(population) <= 0:
            population = np.array([])
        
        if not isinstance(lamb, AEG): lamb = self._convert_to_AEG(lamb)
        P = len(population)
        
        for p in range(0, P): # Search population
            w = population[p]
            
            if (w.aexp == lamb.aexp and len(lamb.pool) >= 1 ): # Checking if abstract trees are equal 
                # print("w")
                # display(w.aexp.visualize_tree())
                # print("lambda")
                # display(lamb.aexp.visualize_tree())
                for particle in lamb.pool: w.pool.append(particle)
                # print(f"----- w insert lambda -----")
                # display(lamb.aexp.visualize_tree())
                # for i in w.pool:
                #     print(i.vector)
                
                w.pool = self.sort_pool_array(w)
                
                
                    
                w.pool = w.pool[0:self.max_pool_size] # truncating to max_pool_size
                w.c = w.pool[0]
                w.sexp = self._convert_to_ExpTree(w)
                return population
        #if is not already in population
        population = np.append(population, lamb)
        # print(population)
        return population
                

    # def mutateSExp(self, me: AEG) -> AEG:
    #     """mutateSExp randomly alters an input s-expression by replacing a randomly selected sub expression 
    #     with a new randomly grown sub expression
        
    #     The new version is pretty much equal to the other. The only exception is the convertion back and forth to AEG"""
        
    #     copied = me.sexp.copy_tree(me.sexp.root())
    #     L = len(copied)
    #     n_steps = randint(0, L)
        
    #     for c, p in enumerate(copied.inorder()):
    #         if c == n_steps:
                
    #             if not copied.is_leaf(p):
    #                 random_number = randint(0,1)
                    
    #                 # attach subtree above
    #                 if random_number == 0:
    #                     parent = copied.parent(p)
    #                     new_subtree = self.generate_expr(1)
                        
    #                     left_most_element = next(new_subtree.inorder())
    #                     if new_subtree.is_root(left_most_element):
    #                         left_most_element = next(new_subtree.postorder())
                        
    #                     left_most_parent = new_subtree.parent(left_most_element)
    #                     new_subtree.delete(left_most_element)
                        
    #                     if copied.is_root(p):
    #                         left_most_parent.Node._left = copied.root().Node
    #                         copied.root().Node._parent = left_most_parent.Node
    #                         copied._root = new_subtree.root().Node
                            
    #                     else:
    #                         copied_parent = copied.parent(p)
    #                         if copied.is_left(p):
    #                             copied_parent.Node._left = new_subtree._root
    #                         else:
    #                             copied_parent.Node._right = new_subtree._root
                            
    #                         left_most_parent.Node._left = p.Node
    #                         p.Node._parent = left_most_parent.Node
    #                         new_subtree._root._parent = copied_parent.Node
                            
                            
                    
    #                 # Change by the same type
    #                 else:    
    #                     copied.replace(p, self._options[p.element_type()](), p.element_type())
                    
    #             # its a leaf
    #             else:
    #                 random_number = randint(0,2)                    
                    
    #                 # Change by the same time
    #                 if random_number == 0:  
    #                     copied.replace(p, self._options[p.element_type()](), p.element_type())

    #                 # Attach a random subtree
    #                 elif random_number == 1:
    #                     size = randint(1, 2)
    #                     subtree = self.generate_expr(size)
                        
    #                     copied.attach_subtree(p,subtree)
                    
    #                 # Delete the node
    #                 elif random_number == 2:
    #                     parent = copied.parent(p)
    #                     e_type = parent.element_type()
    #                     copied.delete(p)
                        
    #                     if e_type == "function": # if the parent was function becomes feature or constant
    #                         if randint(0,1):
    #                             copied.replace(parent, choice(self._feature_names), "feature")
    #                         else:
    #                             copied.replace(parent, uniform(self.random_const_range[0], self.random_const_range[1]), "constant")
    #                     else: # can only be operator
    #                         copied.replace(parent, choice(self._functions), "function")
        
    #             break
            
    #     copied = self._convert_to_AEG(copied)
    #     return copied

    def mutateSExp(self, me: AEG) -> AEG:
        """mutateSExp randomly alters an input s-expression by replacing a randomly selected sub expression 
        with a new randomly grown sub expression
        
        The new version is pretty much equal to the other. The only exception is the convertion back and forth to AEG"""
        
        copied = me.sexp.copy_tree(me.sexp.root())
        L = len(copied)

        if L <= 1: 
            return copied

        n_steps = randint(0, L)
        for c, p in enumerate(copied.inorder()):
            if c == n_steps:

                if not copied.is_leaf(p):
                    random_number = randint(0,1)
                    
                    # attach subtree above
                    if random_number == 0:
                        parent = copied.parent(p)
                        new_subtree = self.generate_expr(1)
                        
                        left_most_element = next(new_subtree.inorder())
                        if new_subtree.is_root(left_most_element):
                            left_most_element = next(new_subtree.postorder())
                        
                        left_most_parent = new_subtree.parent(left_most_element)
                        new_subtree.delete(left_most_element)
                        
                        if copied.is_root(p):
                            left_most_parent.Node._left = copied.root().Node
                            copied.root().Node._parent = left_most_parent.Node
                            copied._root = new_subtree.root().Node
                            
                        else:
                            copied_parent = copied.parent(p)
                            if copied.is_left(p):
                                copied_parent.Node._left = new_subtree._root
                            else:
                                copied_parent.Node._right = new_subtree._root
                            
                            left_most_parent.Node._left = p.Node
                            p.Node._parent = left_most_parent.Node
                            new_subtree._root._parent = copied_parent.Node
                            
                            
                    
                    # Change by the same type
                    else:    
                        copied.replace(p, self._options[p.element_type()](), p.element_type())
                    
                # its a leaf
                elif copied.is_leaf(p):
                    random_number = randint(1,2)                    

                    # Attach a random subtree
                    if random_number == 1:
                        size = randint(1, 2)
                        subtree = self.generate_expr(size)
                        copied.attach_subtree(p,subtree)
                        # try:
                        #     copied.attach_subtree(p,subtree)
                        # except:
                        #     display(copied.visualize_tree())
                        #     print(p, type(p))
                        #     display(subtree.visualize_tree())
                        #     raise("ERRO na posição p - mutação")
                    
                    # Delete the node
                    elif random_number == 2:
                        parent = copied.parent(p)
                        e_type = parent.element_type()
                        
                        # try:
                        #     e_type = parent.element_type()
                        # except:
                        #     print(p, type(p))
                        #     display(copied.visualize_tree())
                        #     print(c)
                        #     raise("ERRO na posição p - mutação")

                        copied.delete(p)
                        
                        if e_type == "function": # if the parent was function becomes feature or constant
                            if randint(0,1):
                                copied.replace(parent, choice(self._feature_names), "feature")
                            else:
                                copied.replace(parent, uniform(self.random_const_range[0], self.random_const_range[1]), "constant")
                        else: # can only be operator
                            copied.replace(parent, choice(self._functions), "function")
        
                break
        
        sc = 0
        for _ in copied.preorder():
            sc += 1
        copied._size = sc

        copied = self._convert_to_AEG(copied)
        return copied
    
    def crossoverSExp(self, mom, dad):
        """crossoverSExp randomly alters a mom input s-expression by replacing a randomly selected sub expression
        in mom with a randomly selected sub expression from dad.
        
        The new version is pretty much equal to the other. The only exception is the convertion back and forth to AEG"""
        
        dad = dad.sexp.copy_tree(dad.sexp.root())
        mom = mom.sexp.copy_tree(mom.sexp.root())
        Ld = len(dad)
        Lm = len(mom)

        # If the mom's expression is too small (a single node) there can't be crossover
        if Lm <= 1:
            return mom

        n = randint(0, Ld-1)
        m = randint(1, Lm-1)  
        
        # since m starts at one and we use preorder traversal,
        # the (mom's) root can't be changed by crossover

        # Generating father sub expression
        for c, p in enumerate(dad.preorder()):
            if c == n:
                sub_expression = dad.copy_tree(p)
                break
            d=c
        # getting mother location
        for c, p in enumerate(mom.preorder()):
            if (c == m):
                try:
                    mom.attach_subtree(p, sub_expression)
                except:
                    print(n, len(dad), dad, d)
                    display(dad.visualize_tree())
                break
        
        sc = 0
        for _ in mom.preorder():
            sc += 1
        mom._size = sc
        mom = self._convert_to_AEG(mom)
        return mom
        
    def _generate_random_velocity(self, dimensions, max_speed = 1.0):
        return np.random.uniform(0, max_speed, dimensions)
        
    #-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #                  Constant Optimization
    #-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def optimizeConstants(self, me:AEG, g: int, Ic: int, check_pool: bool):
        # Individuals must have two or more particles in the pool
        # Only the "best solutions" get optimized constants
        if len(me.c.vector) <= 0:
                return me.copy_AEG(), 0
        
        if len(me.pool) <= 1 and check_pool == True: 
            return me.copy_AEG(), 0

        # Every individual (that has constants) gets optimized
        # if len(me.c.vector) <= 0:
        #         return me.copy_AEG(), 0


        if self.optimization_kind == "NoOpt":
            # Baseline for comparison, no optimization method
            me = me.copy_AEG()
            return me, 0


        if self.optimization_kind == "PSO":
            if check_pool == False:
                return me.copy_AEG(), 0
            else:
                r_me, r_Ic = self.PSO(me, g, Ic)
                return r_me, r_Ic
            
        if self.optimization_kind == "PSO_NEW":
            me = me.copy_AEG()

            # Set-up hyperparameters
            options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
            n_particles = 30
            iterations = 300


            n_params = len(me.pool[0].vector)
            if n_params <= 0:
                print("nada para otimizar")
                return me, 0    # In this case there's nothing to optimize


            # Creating a string that later will be converted to a function call
            fcall_string = "func(X, "
            i=0
            while i<n_params-1:
                fcall_string += f"params[:, {i}], "
                i+=1
            fcall_string += f"params[:, {i}])"

            

            def cost_function(params):
                func = self.toFunc(me)
                X = np.c_[self._features[self._feature_names[0]]]
                y = np.c_[self.y]
                y_pred = eval(fcall_string)
                return np.mean((np.c_[self.y] - y_pred)**2, axis=0)            

            # Call instance of PSO
            optimizer = pyswarms.single.GlobalBestPSO(n_particles=n_particles, dimensions=n_params, options=options)

            # Perform optimization
            _, pos = optimizer.optimize(cost_function, iters=iterations, verbose=False)  


            particle = Particle(pos, 
                                    self._generate_random_velocity(me.pool[0].vector.shape[0]),
                                    me.pool[0].vector)  
            me.pool.append(particle)
            me.pool = self.sort_pool_array(me)
            me.c = me.pool[0]
            me.sexp.fitness_score = self.fitness_score(me)
            me.sexp = self._convert_to_ExpTree(me)            

            return me, 0
        
        if self.optimization_kind == "LS":
            me = me.copy_AEG()
            
            try:
                # print(me.pool[0].vector)
                params, _ = curve_fit(self.toFunc(me), 
                                      self._features[self._feature_names[0]], self.y, me.pool[0].vector)
                # print(params)
                particle = Particle(params, 
                                    self._generate_random_velocity(me.pool[0].vector.shape[0]),
                                    me.pool[0].vector) 
                me.pool.append(particle)
                me.pool = self.sort_pool_array(me)
                me.c = me.pool[0]
                me.sexp.fitness_score = self.fitness_score(me)
                me.sexp = self._convert_to_ExpTree(me)
                # print("reached")
            except RuntimeError:
                params = me.pool[0]
            
            return me, 0
    
        if self.optimization_kind == "random_LS":
            me = me.copy_AEG()
            # print(me.pool)
            
            try:
                # Vector of random numbers
                guess = np.random.uniform(low=self.random_const_range[0], 
                                          high=self.random_const_range[1],
                                          size=len(me.pool[0].vector))
                params, _ = curve_fit(self.toFunc(me),
                                       self._features[self._feature_names[0]], self.y, guess)

                particle = Particle(params, 
                                    self._generate_random_velocity(me.pool[0].vector.shape[0]),
                                    me.pool[0].vector)              
                me.pool.append(particle)
                me.pool = self.sort_pool_array(me)
                me.c = me.pool[0]
                me.sexp.fitness_score = self.fitness_score(me)
                me.sexp = self._convert_to_ExpTree(me)
            
            except RuntimeError:
                params = me.pool[0]
            
            return me, 0

        if self.optimization_kind == "differential_evolution":
            me = me.copy_AEG()
            # display(me.aexp.visualize_tree())
            # print(len(me.pool))

            func = self.toFunc(me)
            n_params = len(me.pool[0].vector)

            def cost_function(params):
                X = self._features[self._feature_names[0]]
                y_pred = func(X, *params)
                return np.mean((self.y - y_pred)**2)
            
            bounds = [(self.random_const_range[0], self.random_const_range[1]) for _ in range(n_params)]
            # print(bounds)

            result = differential_evolution(cost_function, bounds)
            params = result.x

            particle = Particle(params, 
                                    self._generate_random_velocity(me.pool[0].vector.shape[0]),
                                    me.pool[0].vector)  
            me.pool.append(particle)
            me.pool = self.sort_pool_array(me)
            me.c = me.pool[0]
            me.sexp.fitness_score = self.fitness_score(me)
            me.sexp = self._convert_to_ExpTree(me)            

            return me, 0
        
        if self.optimization_kind == "dual_annealing":
            me = me.copy_AEG()
            func = self.toFunc(me)
            n_params = len(me.pool[0].vector)

            def cost_function(params):
                X = self._features[self._feature_names[0]]
                y_pred = func(X, *params)
                return np.mean((self.y - y_pred)**2)
            
            bounds = [(self.random_const_range[0], self.random_const_range[1]) for _ in range(n_params)]
    

            result = dual_annealing(cost_function, bounds)
            params = result.x

            particle = Particle(params, 
                                    self._generate_random_velocity(me.pool[0].vector.shape[0]),
                                    me.pool[0].vector)  
            me.pool.append(particle)
            me.pool = self.sort_pool_array(me)
            me.c = me.pool[0]
            me.sexp.fitness_score = self.fitness_score(me)
            me.sexp = self._convert_to_ExpTree(me)            

            return me, 0
        

    def PSO(self, me: AEG, g: int, Ic: int):
        """
        Parameters: WL, WG, WV, maxPoolSize
        Summary: Particle Swarm constant optimization optimizes a pool of vectors,
        in an AEG formatted individual, by randomly selecting a pair of constant vectors
        from the pool of constant vectors. A new vectors is produced when the pair of 
        vectors, together with the global best vector, are randomly nudged closed together
        based upon their previous approaching velocities. The new vector is scored.
        After scoring, the population of vectors is truncated to those with the best scores
        
        Remember that his code is supposed to run inside the genetic loop, and use that loop
        as its own. So it'll try to modify external variables that are present on the genetic loop.
        These variables are: WL WG WV Ic
        It also uses external variables, that it doesn't modify. These are: maxPoolSize
        """
        
        me = me.copy_AEG()
        
        
        # vars (Ic starts at 0)
        J = len(me.pool)
        if J <= 0: return J
    
        i = Ic
        c = np.copy(me.pool[i].vector)
        # print("c:", c)
        
        v = np.copy(me.pool[i].velocity)
        # initializing velocity vectors in range (0,1)
        if (v.all() == None):
            # print(c.shape[0])
            v = self._generate_random_velocity(c.shape[0])
            me.pool[i].velocity = v
        
        lbest = me.pool[i].best
        if (lbest.all() == None):
            lbest = c.copy()
            # print("entrei: ", c)
            me.pool[i].best = lbest

        
        gbest = me.c.best
        if (gbest.all() == None) or gbest is None:
            gbest = c
            me.c.best = gbest
        
        # Compute the velocity weight parameters
        maxg = self.G
        # g = current generation count in the main GP search
        WL = 0.25 + ((maxg - g)/maxg)  # local weight
        WG = 0.75 + ((maxg -g)/maxg)  # global weight
        WV = 0.50 + ((maxg -g)/maxg)  # velocity weight
        I = len(c)
        r1 = np.random.uniform(0,1)
        r2 = np.random.uniform(0,1) 
        
        # Update the particle's velocity and position
        # a coordinate at a time
        
        if lbest == None:
            lbest = c.copy()
        if gbest == None:
            gbest = c.copy()
        if v == None:
            v = self._generate_random_velocity(c.shape)

        for i in range(0, I):
            lnudge = (WL * r1 * (lbest[i] - c[i]))
            gnudge = (WG * r2 * (gbest[i] - c[i]))
            v[i] = (WV * v[i]) + lnudge + gnudge
            c[i] = c[i] + v[i]
        
        # Score the new particle's position
        me.c.vector = c
        me.c.velocity = v
        
        me.sexp.fitness_score = self.fitness_score(me)
        
        # Defining lbest fitness and gbest fitness
        # temporarily using the me variable to calcualte lbest and gbest
        me.c.vector = lbest  
        lbest_fitness = self.fitness_score(me)
        me.c.vector = gbest 
        gbest_fitness = self.fitness_score(me)
        me.c.vector = c   # returning to original value
        
        # Update the best particle position
        if (me.sexp.fitness_score > lbest_fitness): lbest = c
        if (me.sexp.fitness_score > gbest_fitness): gbest = c
        me.c.best = gbest
        me.pool[-1].vector = c
        me.pool[-1].best = lbest
        me.pool[-1].velocity = v
        
        # Enforce elitis constant pool
        me.pool = self.sort_pool_array(me)
        me.pool[0:self.max_pool_size]
        me.c = me.pool[0]
        me.sexp = self._convert_to_ExpTree(me)
        
        #Enfornce iterative search of contant pool
        Ic = Ic + 1
        if (Ic >= self.max_pool_size): Ic = 0
        return me, Ic
        
    
        
    # ------------------ Population pruning --------------
    
#     def _weightedComlexity(self, tree: ExpressionTree):
#         """Returns a complexity score where each function operator has a different weigh"""
        
#         complexity = 0

#         for element in tree.preorder():
#             n_child = tree.num_children(element)
#             if n_child >= 1:
#                 complexity += (n_child+1) * self._weights[element.element()]
#                 # add one because the element itself counts as one element of the block
                
#         return complexity
    
    def _weightedComlexity(self, me: AEG):
        """Returns a complexity score where each function operator has a different weigh"""
        
        complexity = 0

        for element in me.sexp.preorder():
            n_child = me.sexp.num_children(element)
            if n_child >= 1:
                complexity += (n_child+1) * self._weights[element.element()]
                # add one because the element itself counts as one element of the block
                
        return complexity
    
    
    def populationPruning(self, in_population, out_population, islands):
        """Copies the input into the output population
        Adds random individuals to the output population
        Sorts the output population in asceding order of fitness score
        Computes the weighted complpexity score of each individuals and
        assings each individual to a complexity island
        Eliminates all non-dominant individuals in each compplexity island
        Truncates the output population to the maximum population size
        Always organizes the population into multiple separate islands by complexity"""
                
        out_population = np.array([])
        # print("cheguei 1")
        #initialize with random individuals
        if (len(in_population) <= 0):
            K = int(5*self.max_population_size)
            
            # print("entrei")
            
            # Initialize population
            for k in range(K):
                lamb = self.generate_expr()
                lamb = self._convert_to_AEG(lamb)
                # lamb.sexp.fitness_score = self.fitness_score(lamb)
                lamb.sexp.weight = None
                # print("w.c: ", lamb.c)
                # print("w.c.vector: ", lamb.c.vector)
                out_population = self.insertLambda(out_population, lamb)
            # print("sai")
        # Copy and add a few more random individuals
        else:    
            out_population = deepcopy(in_population)
            K = int(self.max_population_size/10)
            
            # Initialize new random population
            for k in range(K):
                
                lamb = self.generate_expr()
                lamb = self._convert_to_AEG(lamb)
                # lamb.sexp.fitness_score = self.fitness_score(lamb)
                lamb.sexp.weight = None
                out_population = self.insertLambda(out_population, lamb)
        # print("cheguei 2")
        # Sorting out_population by fitness score
        out_population = self.sort_AEG_array(out_population)
        out_population = out_population[0:self.max_population_size]
        # print("cheguei 3")    
        # Compute weighted complexity range
        N = len(out_population)
        high = low = self._weightedComlexity(out_population[0])
        # print("cheguei 4")
        for n in range(N):
            lamb = out_population[n]
            if (lamb.sexp.weight == None):
                lamb.sexp.weight = self._weightedComlexity(lamb)
                weight = lamb.sexp.weight
                
                if (weight < low):
                    low = weight
                if (weight > high):
                    high = weight
        # print("cheguei 5")
        weight_range = high-low
        island_counts = np.zeros(self.max_island_count)
        in_population = np.array([])
        islands = np.zeros(self.max_island_count, dtype="object")
        
        # print("cheguei 6")
        
        # Always return island structure with one island for each complexity partition
        # prune all non-dominant individuals in each pareto front complexity island
        
        # Inserting all elements in their respective islands
        for n in range(N):
            lamb = out_population[n]
            weight = lamb.sexp.weight
            island = np.abs(int(self.max_island_count * ((weight-low) / weight_range)))
            
            # The tree is too complex, then it goes to the last place
            if island >= self.max_island_count:
                island = self.max_island_count-1
            lamb.sexp.island = island            
            
            island_counts[island] = island_counts[island] + 1     # Increasing the count of element inside the island
            if (island_counts[island] <= self.max_island_size):
                in_population = np.append(in_population,lamb)
                if (type(islands[island]) is int): islands[island] = []
                islands[island] = np.append(islands[island],lamb)
        
        # print("cheguei 7")

        
        # This is the same as the first element of the out_population
        # Which was ordered is ascending order of fitness score
        champ = in_population[0]
        return champ, islands, in_population  
    
    def predict(self, gen_fit_path=None):
        """Must initialize Ic as 0"""
        
        max_island_count = 1
        in_population = np.array([])
        out_population = np.array([])
        islands = np.array([])
        
        if(self.G <= 0):
            champ, islands, in_population = self.populationPruning(in_population, out_population, islands)
        # P = len(in_population)
        
        
        for g in range(0, self.G): # Main evolution loop
            print("iniciou")
            # print("g:", g)
            P = len(in_population)
            # print(in_population)
            
            # initialize Ic for optimizeConstants
            # Constant optimization for the best individual in each island
            
            # for i in in_population:
            #     print("ilha: ",i.sexp.island)
            # print("========")
            
            for i in range(len(islands)):
                try:
                    lamb, Ic = self.optimizeConstants(islands[i][0], g, i, check_pool=False)
                    out_population = self.insertLambda(out_population, lamb)
                except TypeError: #A ilha tá vazia
                    pass

            
            # for i in in_population:
            #         print("ilha: ",i.sexp.island)
                
                # display(islands[i][0].visualize_tree())


            # Everyone gets mutated and crossed over
                
            for p in range(0, P):
                # print("p:", p)
                Ic = 0
                
                lamb, Ic = self.optimizeConstants(in_population[p], g, Ic, check_pool=True)
                out_population = self.insertLambda(out_population, lamb)
                lamb = self.mutateSExp(in_population[p])
                out_population = self.insertLambda(out_population, lamb)
                dad = in_population[p] # every one gets crossed over (gets to be a dad)
                
                
                # Cross over partner must be from the same island 
                if dad.sexp.island is None: 
                    # Have no ideia why this bug happens sometimes, some individuals get here without a defined island
                    print("empty dad island")
                    display(dad.sexp.visualize_tree())
                    K_original = 0
                else:
                    K_original = dad.sexp.island

                try:
                    K = np.random.randint(K_original-self.island_interval[0], K_original+self.island_interval[1]+1)
                except:
                    print(K_original, type(K_original), self.island_interval[0], type(self.island_interval[0]))
                    raise NotImplementedError("Erro!")
                
                if K < 0: K=0
                if K >= len(islands): K = len(islands)-1

                if type(islands[K]) is int:
                    K = K_original
                

                i = randint(0, len(islands[K])-1)  # Choosing a random individual from island K
                mom = islands[K][i]   # Getting a random tree from the same island as dad
                lamb = self.crossoverSExp(dad, mom)
                out_population = self.insertLambda(out_population, lamb)
            
            champ, islands, in_population = self.populationPruning(out_population, in_population, islands)

            if gen_fit_path is not None:
                with open(f"{gen_fit_path}.csv", "a") as file:
                    file.write(f"{g},{self.fitness_score(champ)}\n")

                with open(f"{gen_fit_path}-{g}", "wb") as file:
                    pickle.dump(champ.sexp, file)
                

        return champ
        