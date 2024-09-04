from CSOWP_SR import *
from ExpressionTree import *
import sympy as smp
import re
import os

# Expressions ===================================================
op_map = {smp.Mul: "*", smp.Add: "+", smp.Pow: "**"}
def _build_expression_tree(expr):
    # Create a tree node for the current expression
    if expr.is_Atom:  # Atoms are constants or variables (leaves)
        return ExpressionTree.Node(expr)
    
    elif expr.is_Function:  # Functions (sin, cos, etc.) with one child
        node = ExpressionTree.Node(expr.func)
        node._left = _build_expression_tree(expr.args[0])  # Single argument
        return node
    
    elif expr.is_Add or expr.is_Mul or expr.is_Pow:  # Binary operators (+, *, **)
        symbol = op_map[type(expr)]
        node = ExpressionTree.Node(symbol)

        node._left = _build_expression_tree(expr.args[0])  # Left child

        if len(expr.args) > 2:
            new_expr = expr.func(*expr.args[1:])
        else:
            new_expr = expr.args[1]
        
        node._right = _build_expression_tree(new_expr)  # Right child
        
        return node
    
    else:
        raise ValueError(f"Unsupported expression: {expr}")

def exprToTree(expr):
    if type(expr) is str:
        expr = smp.parse_expr(expr)

    tree = ExpressionTree()
    tree._root = _build_expression_tree(expr)
    tree._size = 1
    tree._size = len(list(tree.preorder()))

    return tree


# Data Manip =====================================================
def create_data(opt_path):
    search = re.search(".+(/.+)$", opt_path)
    optimization = search.group(1)[1:]

    dataFrames = []
    
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
        info = pd.read_csv(info_path)
        expression = info.columns[1]

        for tree in os.listdir(trees_path):
            search = re.search(".+(-.+)$", tree)
            index = int(search.group(1)[1:])
            tree_path = os.path.join(trees_path, tree)

            # Tree ============
            with open(tree_path, "rb") as file:
                tree = pickle.load(file)
            
            tree_dict = tree.parentChildRepr()
            tree_dict["optimization"] = optimization
            tree_dict["problem"] = problem
            tree_dict["index"] = index
            tree_dict["MSE"] = MSE[index]
            tree_dict["training_time(s)"] = training_time[index]
            tree_dict["solution_string"] = solution_string[index]
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


