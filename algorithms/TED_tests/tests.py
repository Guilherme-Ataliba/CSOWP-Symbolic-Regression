import edist.ted as ted
import numpy as np
import edist.tree_utils as tree_utils

tree1 = {"nodes": np.array(["+", "x", "2"]),
         "adj": [[1, 2], [], []]}
tree_utils.tree_to_string(tree1["nodes"], tree1["adj"])

tree2 = {"nodes": ["x", "2", "x"],
         "adj": [[1, 2], [], []]}
tree_utils.tree_to_string(tree2["nodes"], tree2["adj"])

ted.standard_ted(tree1["nodes"], tree1["adj"], tree2["nodes"], tree2["adj"])
