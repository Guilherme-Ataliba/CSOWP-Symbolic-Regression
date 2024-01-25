# 🔵 Baseline Symbolic Regression Algorithm

## 🔷 Notes About the Algorithm Functionality 

### 🔹 Constants
Constants in the algorithm are limited to the range from **zero** to **one**. This way should be easier to focus on the form of expression, instead of the tuning of the parameters. Obviously, we expect then that the data is normalized. 

## 🔷 To Implement

#### 1. Parenthesize

#### 2. Evaluation of expressions

#### 3. Log
I removed the log function and the square root function from the possible function because they don't accept negative values. Should implement again.

#### 4. Weights
Some things must have weights, equal probability is not right. For example:
1. features should appear more than constants
2. functions should appear more than operators
	1. some functions should appear more than others
	2. as some operators should appear more than others
These weights should be able to be customized via input (init) of the class
## 🔷 Issues to fix

#### 1. Unnecessary random values
Fix the "generate random expression" function

#### 2. Copy tree has a for loop
The copy tree function has, besides the recursive part of the code, a for loop to assign the right values for each parent

#### 3. Redo `graphviz` visualization 
Study the library and make a tree that shows as it is supposed to

#### 4. Do a code check-up 
I suppose there are some parts of the code that could be improved, optimized or simplified

#### 5. Elaborate a parenthesize expression for fitness future `sympy`
Elaborate a function that can translate the tree to the correct expression, for `sympy`, but specially for evaluating fitness

#### 6. Create a fit method, instead of using the init

#### 7. Simplify the initialized in the init

#### 8. Zero and negative values on the dataframe
Zero causes problems for division and negative values cause problems for square root and absolute value

