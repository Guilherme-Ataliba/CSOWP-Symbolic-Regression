# 🔵 Baseline Symbolic Regression Algorithm

## 🔷 Notes About the Algorithm Functionality 

### 🔹 Constants
Constants in the algorithm are limited to the range from **zero** to **one**. This way should be easier to focus on the form of expression, instead of the tuning of the parameters. Obviously, we expect then that the data is normalized. 

## 🔷 To Implement

#### 1. Parenthesize

#### 2. Evaluation of expressions

#### 3. Log
I removed the log function and the square root function from the possible function because they don't accept negative values. Should implement again.

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



