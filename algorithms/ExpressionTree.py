import numpy as np
from random import randint, choice, uniform
import graphviz
from typing import Any, List, Dict
from functools import singledispatchmethod
import sympy as smp

class ExpressionTree():
    """Tree used in the symbolic regression algorithm"""
    
    class Node:
        __slots__ = "_element", "_parent", "_left", "_right", "_element_type"
        def __init__(self, element, parent=None, left=None, right=None, element_type=None):
            self._element = element
            self._parent = parent
            self._left = left
            self._right = right
            
            self._element_type = element_type
        
        def element(self):
            return self._element
    
    
    class Position():
        def __init__(self, container, node):
            self._container = container
            self.Node = node
            
        def __eq__(self, other):
            return type(other) is type(self) and other.Node is self.Node
        
        def __neq__(self, other):
            raise not (self == other)
            
        def element_type(self):
            return self.Node._element_type
            
        def element(self):
            return self.Node._element
        

    __slots__ = ("_root", "_size", "fitness_score", "island", "weight")
    def __init__(self, fitness_score = None, island=None, weight=None):
        self._root = None
        self._size = 0
        self.fitness_score = fitness_score
        self.island = island
        self.weight = weight
        
    def __eq__(self, other):
        
        a = [i.element() for i in self.postorder()]
        b = [i.element() for i in other.postorder()]
        
        return a == b
        
    
    def _validate(self, p):
        """Return associated node, if position p is valid"""
        if not isinstance(p, self.Position):
            raise TypeError("p must be proper Position type")
            
        if p._container is not self:
            raise ValueError("p does not belong to this container")
            
        if p.Node._parent is p.Node:   # Convention for deprecated nodes
            raise ValueError("p is no longer valod")
            
        return p.Node
    
    def _make_position(self, node):
        return self.Position(self, node) if node is not None else None
    
    def __len__(self):
        return self._size
    
    def root(self):
        return self._make_position(self._root)
    
    def parent(self, p):
        node = self._validate(p)
        return self._make_position(node._parent)
    
    def left(self, p):
        node = self._validate(p)
        return self._make_position(node._left)
    
    def right(self, p):
        node = self._validate(p)
        return self._make_position(node._right)
    
    def sibling(self, p):
        parent = self.parent(p)
        
        if parent is None:  # if p is root
            return None
        
        if p == self.left(parent):
            return self.right(parent)
        else:
            return self.left(parent)
        
    def children(self, p):
        if self.left(p) is not None:
            yield self.left(p)
        if self.right(p) is not None:
            yield self.right(p)
            
    def num_children(self, p):
        """Number of direct children of Position p"""
        node = self._validate(p)
        count = 0
        
        if node._left is not None:   #left child exists
            count += 1
        if node._right is not None:  #right child exists
            count += 1
        return count
    
    def is_root(self, p):
        return self.root() == p
    
    def is_leaf(self, p):
        return self.num_children(p) == 0
    
    def is_empty(self):
        return len(self) == 0    
    
    def add_root(self, e, e_type=None):
        """Place element e at the root of an empty tree and return new Position
        
        Raise ValueError is tree nonempty"""
        
        if self._root is not None: raise ValueError("Root exisits")
        self._size = 1
        self._root = self.Node(e, element_type=e_type)
        return self._make_position(self._root)
    
    def add_left(self, p, e, e_type=None):
        """Create a new left child for Position p, storing element e
        
        Return the Position of new node
        Raise ValueError is Position p is invalid or p alredy has a left child"""
        
        node = self._validate(p)
        if node._left is not None: raise ValueError("Left child exists")
        self._size += 1
        node._left = self.Node(e, node, element_type = e_type)
        return self._make_position(node._left)
        
    def add_right(self, p, e, e_type=None):
        """Create a new right child for Position p, storing element e
        
        Return the Position of new node
        Raise ValueError is Position p is invalid or p alredy has a right child"""
        
        node = self._validate(p)
        if node._right is not None: raise ValueError("Right child exists")
        self._size += 1
        node._right = self.Node(e, node, element_type = e_type)
        return self._make_position(node._right)
    
    def delete(self, p):
        """Delete the node at Position p and replace it with its child, if any.
        
        Return the element that had been stored at Position p.
        Raise ValueError if Position p is invalid or p has two children"""
        
        node = self._validate(p)
        if self.num_children(p) == 2: raise ValueError("p has two children")
        child = node._left if node._left else node._right    #might be None
        
        if child is not None:
            child._parent = node._parent
        
        if node is self._root:   # if node = root child becomes new root
            self._root = child
        else:    # if not must update grandparent
            parent = node._parent
            if node is parent._left:
                parent._left = child
            else:
                parent._right = child
            self._size -= 1
            node._parent = node   #convention for deprecated node
            return node._element
        
    def _attach(self, p, t1=None, t2=None):
        """Attach trees t1 and t2 as left and right subtrees of external p
        
        Raise ValueError if p is not a leaf
        Raise TypeError is all three tree types are not the same"""
        
        node = self._validate(p)
        if not self.is_leaf(p): raise ValueError("Position must be leaf")
        
        # check if all 3 are trees of the same type
        if not type(self) is type(t1) is type(t2):
            raise TypeError("Tree types must match")    
        self._size += len(t1) + len(t2)   # we use len becouse ._size is private and t1 and t2 are external
        
        # Attach t1 to the left
        if not t1.is_empty():
            t1._root._parent = node
            node._left = t1._root
            t1._root = None   # Set t1 instance to empty
            t1._size = 0
        # Attach t2 to the right
        if not t2.is_empty():
            t2._root._parent = node
            node._right = t2._root
            t2._root = None
            t2._size = 0
        
    
    def replace(self, p, e, e_type=None):
        """Replace the element at position p with e, and return old element"""
        
        node = self._validate(p)
        old = node._element
        node._element = e
        node._element_type = e_type
        return old
    
    def is_left(self, p):
        "Return if the position is a left child of its parent"
        return p == self.left(self.parent(p))
    
    def attach_subtree(self, p, subtree):
        """Removes the informod position p and attaches the input subtree in its place.
        If p is not a leaf, deleat everythin thing bellow it"""
        
        if not self.is_leaf(p): 
            for position in self._subtree_postorder(p):
                if position == p:
                    break
                self.delete(position)
        # now p is a leaf        

        # check if both are trees of the same type
        if not type(self) is type(subtree):
            raise TypeError("Tree types must match")    
        self._size += len(subtree) - 1 # -1 since we'll remove the leaf 
        
        if subtree.is_empty():
            raise ValueError("Tree can't be empty")
        
        parent = self.parent(p)
        node = self._validate(parent)
        # Attach to the left
        if self.is_left(p):
            self.delete(p)
            subtree._root._parent = node
            node._left = subtree._root
            subtree._root = None   
            subtree._size = 0
        # Attach to the right
        else:
            self.delete(p)
            subtree._root._parent = node
            node._right = subtree._root
            subtree._root = None
            subtree._size = 0
            
    def visualize_tree(self):
        """Not efficient AT ALL, but since this is only a visualization method that
        will be used on a single tree by user's request, and not as a part of the method itself,
        it shouldn't matter.
        
        Here we use the element_type attribute as an unique id for each node in the tree."""
        
        id_tree = self.copy_tree(self.root())
        
        for c, p in enumerate(id_tree.preorder()):
            p.Node._element_type = c
    
        dot = graphviz.Digraph()
        dot.node(str(id_tree.root().Node._element_type), str(id_tree.root().element()))
        
        for p in id_tree.preorder():
            if id_tree.left(p):
                node_content = str(id_tree.left(p).element())
                element_type = str(id_tree.left(p).Node._element_type)
                parent_element_type = str(p.Node._element_type)
                dot.node(element_type, node_content)
                dot.edge(parent_element_type, element_type)
            if id_tree.right(p):
                node_content = str(id_tree.right(p).element())
                element_type = str(id_tree.right(p).Node._element_type)
                parent_element_type = str(p.Node._element_type)
                dot.node(element_type, node_content)
                dot.edge(parent_element_type, element_type)
        return dot
            
    def toString(self, operators, functions, custom_functions_dict):
        
        def recursive_lamb(root):
            if root is None:
                return ""
            
            if root._left is None and root._right is None:
                return str(root._element)
            
            left = recursive_lamb(root._left)
            right = recursive_lamb(root._right)

            if root._element in operators:
                if root._element == "/":
                    return "np.divide(" + left + "," + right + ")"
                # else
                # print(left, str(root._element), right)
                return "(" + left + str(root._element) + right + ")"
                
            elif root._element in functions:
                if left == "":
                    node = right
                else: 
                    node = left
                
                """
                dict = {"exp-": ["exp(-", ")"] }
                elif root._element == "exp-":
                    return "exp(-" + node + ")"
                """

                # if root._element == "cube":
                #     return "(" + node + ")" + "**3"
                # elif root._element == "quart":
                #     return "(" + node + ")" + "**4"
                if root._element in custom_functions_dict:
                    return custom_functions_dict[root._element][0] + node + custom_functions_dict[root._element][1]
                else:
                    return root._element + "(" + node + ")"
        
        return recursive_lamb(self.root().Node)
    
    def toString_smp(self, operators, functions, custom_functions_dict):
        
        def recursive_lamb(root):
            if root is None:
                return ""
            
            if root._left is None and root._right is None:
                return str(root._element)
            
            left = recursive_lamb(root._left)
            right = recursive_lamb(root._right)

            if root._element in operators:
                return "(" + left + str(root._element) + right + ")"
                
            elif root._element in functions:
                if left == "":
                    node = right
                else: 
                    node = left
                
                """
                dict = {"exp-": ["exp(-", ")"] }
                elif root._element == "exp-":
                    return "exp(-" + node + ")"
                """

                # if root._element == "cube":
                #     return "(" + node + ")" + "**3"
                # elif root._element == "quart":
                #     return "(" + node + ")" + "**4"
                if root._element in custom_functions_dict:
                    return custom_functions_dict[root._element][0].replace("np.", "") + node + custom_functions_dict[root._element][1]
                else:
                    return root._element + "(" + node + ")"
        
        return recursive_lamb(self.root().Node)
    
    def toSmpExpr(self, operators, functions, custom_functions_dict, parameters=None):
        expr_string = self.toString_smp(operators, functions, custom_functions_dict)

        sexp = smp.sympify(expr_string)

        return sexp

    def toFunc(self, operators, functions, custom_functions_dict, feature_names=None, interpreter="numpy",
               inv_data: Dict = None):
        smp_expr = self.toSmpExpr(operators, functions, custom_functions_dict)
        

        if feature_names is None:
            symbols_list = list(smp_expr.free_symbols)
        else:
            symbols_list = feature_names

        symbols_string = ""
        for i in symbols_list:
            symbols_string += f"{i}, "

        symbols = smp.symbols(symbols_string)

        if inv_data is not None:
            if len(feature_names) > 1:
                raise("Only implemented for functions of one variable")
            
            inv_smp_expr = (inv_data["ymax"] - inv_data["ymin"])*smp_expr.subs(symbols[0], (symbols[0] - inv_data["Xmin"])/(inv_data["Xmax"] - inv_data["Xmin"])) + inv_data["ymin"]
            return smp.lambdify(symbols, inv_smp_expr, interpreter), inv_smp_expr, smp_expr
        
        return smp.lambdify(symbols, smp_expr, interpreter), smp_expr


    def copy_tree(self, root):
        """!!!!!!! Não muito eficiente pois precisa de um loop para definir os pais de cada nó"""
        new_tree = ExpressionTree()
        
        def auxilary_function(root):
            """Recursive part of copying the tree"""
            if root == None:
                return None
            
            new_node = new_tree.Node(root._element, element_type=root._element_type)
            parent_node = new_node
            new_node._left = auxilary_function(root._left)
            new_node._right = auxilary_function(root._right)
            
            return new_node
        
        def set_parents(new_tree):
            """Correctly assigns the parent to each node on the tree"""
            new_tree.root().Node._parent = None
            for p in new_tree.preorder():

                if new_tree.left(p):
                    p.Node._left._parent = p.Node
                if new_tree.right(p):
                    p.Node._right._parent = p.Node
        
        new_tree._root = auxilary_function(root.Node)
        new_tree._size = self._size
        set_parents(new_tree)
        
        return new_tree

     # ------------------------- Preorder -------------------------
    
    def preorder(self):
        if not self.is_empty():
             for p in self._subtree_preorder(self.root()):
                    yield p
    
    def _subtree_preorder(self, p):
        yield p
        for c in self.children(p):
            for other in self._subtree_preorder(c):
                yield other
                
    # ------------------------- Postorder -------------------------
    
    def postorder(self):
        if not self.is_empty():
            for p in self._subtree_postorder(self.root()):
                yield p
    
    def _subtree_postorder(self, p):
        for c in self.children(p):
            for other in self._subtree_postorder(c):
                yield other
        yield p
        
    # ------------------------- Inorder -------------------------
    
    def inorder(self):
        if not self.is_empty():
            for p in self._subtree_inorder(self.root()):
                yield p
    
    def _subtree_inorder(self, p):
        if self.left(p) is not None:
            for other in self._subtree_inorder(self.left(p)):
                yield other
        yield p
        if self.right(p) is not None:
            for other in self._subtree_inorder(self.right(p)):
                yield other
                
    def positions(self):
        return self.preorder()
    
    def __iter__(self):
        for p in self.positions():
            yield p.element()
        