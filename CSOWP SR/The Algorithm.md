In the basic GP algorithm for Symbolic Regression, an s-expression is manipulated via evolutionary techniques of mutation and crossover to produce new s-expressions to be tested, as a basis function candidate in GLM.
- Basis functions candidates that produce better fitting GLMs are promoted

**Mutation**: inserts a random s-expression in a random location in the starting s-expression. For example:
$$\displaylines{B_3 = \frac{\sqrt{x_2}}{\tan(x_5/4.56)}} \;\longrightarrow\; B_4 = \frac{\sqrt{x_2}}{\text{\textbf{cube}}(x_5/4.56)}$$
**Crossover**: Combines portions of a mother s-expression and a father s-expression to produce a child s-expression. Crossover inserts a randomly selected sub expression from the father into a randomly selected location in the mother s-expression. For example:
$$\displaylines{\text{(Father)}\;\;\;\;\;\;\;\; B_3 = \frac{\sqrt{x_2}}{\tan(x_5/4.56)}\\
\text{(Mother)} \;\; B_4 = \tanh(\cos(x_2*0.2) * \text{\textbf{cube}}(x_5 + \text{abs}(x_1)))\\
\text{(Child)}\;\; B_5 = \tanh(\tan(x_5/4.56) * \text{\textbf{cube}}(x_5 + \text{abs}(x_1)))}$$

#### The Algorithm Idea
Mutation and Crossover operations are the basis for ordinary GP. The algorithm randomly creates a population of candidate basis functions, and apply mutation and crossover over those basis functions while promoting the best fit. The winners being the collection of basis functions which receive the most favorable least squares error, with standard regression techniques.

- X: A vector of N training points.
- Y: A vector of N dependent variables.
- G: The number of generations to train.
	The fitness score is the mean-squared error, divided by the standard deviation of Y, NLSE (Non-linear square error from GLM).

