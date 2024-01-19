font: [[ABaselineSymbolicRegressionAlgorithm.pdf]]

Constant Swarm with Operator Weighted Pruning - CSOWP

> "Nonlinear regression is the mathematical problem symbolic regression aspires to solve"

The canonical generalization of nonlinear regression is the class of Generalized Linear Models. 

> Now, the value of the GLM model is that one can use standard regression techniques and theory. While Symbolic Regression does not add anything to the standard techniques of regression, its value lies in its ability as a search technique.

### S-expressions
These are expressions that can be expressed as nested parenthesis, for example
$$(/\; (+\; x4\; 4.45)\; 2.0) \longrightarrow \frac{x4 + 4.45}{2.0}
$$
Non-terminal nodes are all operators, and the terminal nodes are always either real number constants or features. 