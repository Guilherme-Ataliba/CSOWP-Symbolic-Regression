from CSOWP_SR import *
from ExpressionTree import *
import sympy as smp
import re
import os
import logging

import sys
sys.set_int_max_str_digits(100000)

import signal

# Expressions ===================================================
op_map = {smp.Mul: "*", smp.Add: "+", smp.Pow: "**"}
def _build_expression_tree(expr, single_name=False):
    # Create a tree node for the current expression
    if expr.is_Atom:  # Atoms are constants or variables (leaves)
        if single_name:
            # print(str(expr).replace(".", ""))
            if str(expr).replace(".", "").replace("-", "").replace("/", "").isnumeric():
                return ExpressionTree.Node("C")
            else: 
                return ExpressionTree.Node(expr)
        else:
            return ExpressionTree.Node(expr)
    
    elif expr.is_Function:  # Functions (sin, cos, etc.) with one child
        node = ExpressionTree.Node(expr.func)
        node._left = _build_expression_tree(expr.args[0], single_name)  # Single argument
        return node
    
    elif expr.is_Add or expr.is_Mul or expr.is_Pow:  # Binary operators (+, *, **)
        symbol = op_map[type(expr)]
        node = ExpressionTree.Node(symbol)

        node._left = _build_expression_tree(expr.args[0], single_name)  # Left child

        if len(expr.args) > 2:
            new_expr = expr.func(*expr.args[1:])
        else:
            new_expr = expr.args[1]
        
        node._right = _build_expression_tree(new_expr, single_name)  # Right child
        
        return node
    
    else:
        raise ValueError(f"Unsupported expression: {expr}")

def exprToTree(expr, single_name=False):
    if type(expr) is str:
        expr = smp.parse_expr(expr)

    tree = ExpressionTree()
    tree._root = _build_expression_tree(expr, single_name=single_name)
    tree._size = 1
    tree._size = len(list(tree.preorder()))

    return tree


# Data Manip =====================================================

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

def handler(signum, frame):
    raise TimeoutError("Execution took longer than 10 seconds!")

def create_data(opt_path):
    # print(opt_path)
    logging.basicConfig(level=logging.ERROR, filename='info.log', format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(opt_path)

    search = re.search(".+(/.+)$", opt_path)
    optimization = search.group(1)[1:]

    dataFrames = []

    symbols = {"x": smp.symbols("x", positive=True, real=True)}
    
    for problem in os.listdir(opt_path):
        results_path = os.path.join(opt_path, problem, "results.csv")
        info_path = os.path.join(opt_path, problem, "info.csv")
        trees_path = os.path.join(opt_path, problem, "trees")


        # Results ======
        results = pd.read_csv(results_path)
        MSE = results["MSE_error"]
        training_time = results["training_time"]
        solution_string = results["solution_string"]

        # Info =======
        info = pd.read_csv(info_path, skiprows=1)
        expression = info.columns[1]

        for tree in os.listdir(trees_path):
            search = re.search(".+(-.+)$", tree)
            index = int(search.group(1)[1:])
            tree_path = os.path.join(trees_path, tree)

            # Simplifying big numbers
            expr_string = re.sub(r'(\d|\.){15,}(e(-|\+)\d{1,2}|)', lambda x: "{:.6e}".format(float(x.group())), solution_string[index])

            try:
                expr = smp.parse_expr(expr_string, local_dict=symbols)
            except MemoryError:
                print("memory error occured")
                expr = smp.parse_expr("0")
            except OverflowError: # Since all constants will be converted symbolic versions the actual value doesnt matter
                expr_string = re.sub(r'(\d|\.){1,}(e(-|\+)\d{1,2}|)', "1.5", expr_string)
                expr = smp.parse_expr(expr_string, local_dict=symbols)
                
            
            signal.signal(signal.SIGALRM, handler)
            try:
                signal.alarm(30)
                expr = expr.simplify(local_dict=symbols)
                expr = smp.nsimplify(expr, tolerance=1e-5).evalf(20)
                signal.alarm(0)
            except:
                pass
            
            try:
                signal.alarm(30)
                expr = recursive_simplify_evalf(expr)
                signal.alarm(0)
            except:
                pass


            # expr = smp.simplify(expr)
            try:
                smp_tree = exprToTree(expr, single_name=True)
            except:
                continue
            tree_dict = smp_tree.parentChildRepr()
            tree_dict["optimization"] = optimization
            tree_dict["problem"] = problem
            tree_dict["index"] = index
            tree_dict["MSE"] = MSE[index]
            tree_dict["training_time(s)"] = training_time[index]
            tree_dict["solution_string"] = expr
            tree_dict["expression"] = expression

            df = pd.DataFrame([tree_dict])
            dataFrames.append(df)
    
    return pd.concat(dataFrames)

def create_all_data(path):
    path = path + "/"
    optimizations = os.listdir(path)

    dataFrames = []

    for opt in optimizations:
        opt_path = os.path.join(path, opt)
        df = create_data(opt_path)
        dataFrames.append(df)
    
    data = pd.concat(dataFrames)
    data.reset_index(inplace=True, drop=True)
    return data


# AEG to ExprTree
def convert_to_AEG(me: ExpressionTree, single_name=False) -> AEG:
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
        if i.element_type() == "constant" or str(i.element()).replace(".", "").isnumeric():     
            r = i.element()
            k = len(out.c.vector)
            out.c.vector = np.append(out.c.vector, r)
            if single_name:
                i.Node._element = "C"
                i.Node._element_type = "absConstant"
            else:
                i.Node._element = f"c[{k}]"
                i.Node._element_type = "absConstant"
    
    out.pool.append(out.c)
    return out

def convert_to_ExpTree(me: AEG) -> ExpressionTree:
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


