# CSOWP SR - Constant Swarm  with Operator Weighted Pruning

This repository implements a symbolic regression model inspired by the article [Korns - 2013 - A Baseline Symbolic Regression Algorithm.pdf](https://github.com/user-attachments/files/17723430/Korns.-.2013.-.A.Baseline.Symbolic.Regression.Algorithm.pdf) and with many improvements by the author. 

Symbolic regression is a machine learning technique that aims to discover mathematical expressions that best describe the relationship within a given dataset. Unlike traditional regression, which fits data to a specific model, symbolic regression searches for the model itself, finding both the structure and the parameters of equations. This approach often employs genetic programming, an evolutionary algorithm that generates and evolves mathematical expressions through operations inspired by natural selection. Symbolic regression is particularly useful for finding interpretable, compact, and potentially novel formulas to explain complex data. Applications range from physics to health care and economics (specially stock market).

Evolutionary search process to find the best expression fit.
![SR_gif](https://github.com/user-attachments/assets/ef794ca7-295c-448b-ae7e-4ef17909e892)

Even though the purpose of this work is not to compete with state-of-the-art models, it produced interesing results that led to the developement of a scientific paper. Such study promotes a benchmark of different constant optimization algorithms in the context of symbolic regression's evolutionary algorithms. 

## Seminar
One of the early steps for applications of this method was to simulate simple physical systems. The research lead to a seminary presentation that presented the model for simple problems such as the:

#### Projectile Motion
![oblique_projectile](https://github.com/user-attachments/assets/fac24d8b-d238-429f-bfc4-4f95c5ad1511)

#### Damped Pendulum
![damped_pendulum](https://github.com/user-attachments/assets/263b0127-8515-4d24-98d2-83acafce3234)
