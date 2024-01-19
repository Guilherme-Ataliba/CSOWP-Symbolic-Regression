font: [[myers1997.pdf]] & https://www.youtube.com/watch?v=SqN-qlQOM5A

Generalized Linear Models (GLM) are interesting for situations in which model errors are not normally distributed - meaning, the residues (difference between the predicted value and the actual value) is not normally distributed. 
- Clearly, the normal (or not) distribution of the residues is related to the distribution of the data itself.
- The usual least square does assume a relevant dependence in normality and homogeneous variance.

Some consequences come from the assumption that the data has normally distributed residues when it doesn't, such as the inaccuracy of some metrics (such as p-values and confidence intervals). Furthermore, if the residues exhibit non-constant variance, it can lead to issues in auto-correlation. 

Other approaches to the problem of non-linearity are transformations - in which you transform your data (by log or any other form more appropriate to the specific problem) so that it can more easily be used in models that require linear/normal data. 

## 🔷 Score Equation
Take an example, the linear model can be written
$$y = \vec{X}\beta + \vec{\epsilon}$$
where $\vec{\epsilon}$ is a vector of n **normal** random errors and $\vec{X}$ is the "model matrix" - that is, the matrix that contains the predictors. The residual sum of squares is given by
$$
	(y-\vec{X}\hat{\beta})(y-\vec{X}\hat{\beta}) = \sum_{i=1}^n (y_i - \hat{y}_i)^2
$$
- Where the hat represents the predicted variables. 

The solution to $\hat{\beta}$ comes from straightforward calculus:
$$\sum_{i=1}^n (y_i - \hat{y}_i)x_i = 0$$

The equation above represent a **score equation**.


## 🔷 Nonlinear Modeling
Most applications of generalized linear models cannot assume homogeneous variance, thus we require a **weighted** nonlinear regression.

For example, consider the model
$$
y_i = \mu(x_i, \beta) + \epsilon_i
$$
Where the actual values $y_i$ are equal to the mean $\mu(x_i, \beta)$ (as a function of $x_i$ data points and $\beta$ are unknown parameters) plus the error $\epsilon_i$.

The variance at the i$^{th}$ data point is not constant, thus a function given by $g[\mu(x_i, \beta)]$, that itself is a function of the mean. The appropriate nonlinear least squares procedure is to minimize the **weighted residual sum of squares**:
$$
\sum_{i=1}^n \frac{[y_i - \mu(x_i, \beta)]^2}{g[\mu(x_i,\beta)]}
$$
If you know the application and the distribution to assume you may substitute the function $g$ for the appropriate value, as a function of the mean. 

**Solution:**
The solution to this minimization problem is achieved iteratively, where you update the values of $\mu$ and the weights $g(\mu)$ at every iteration. Thus, the procedure is called **iterated reweighted least squares** (IRWLS).


# 🔵 Generalized Linear Models
When the data is obtained from a procedure that operate in binomial or Poisson distribution, the use of ordinary least squares is not adequate. In this kind of distributions, the variance is a function of the mean - thus the estimation procedure should take the error estimation into account. 

Generalized linear models is a term used to describe different "families" of nonlinear models, for example:
- **Exponential Family:** That encompasses the Poisson, binomial and exponential distributions

> ⭐ The whole idea is, if you face yourself with one of those distributions you should use them as a way of calculating the metrics and the sum of squares - instead of assuming they're all a normal distribution (which they're not)


## 🔷 Link Function
A generalized linear model can be subdivided in a linear part, called **linear predictor** (also called systematic component), expressed as
$$x'\beta = \beta_0 + \beta_1x_1 + \beta_2x_2 + ...$$
The complete model is constructed through a relationship between the linear predictor and the distribution mean - this relation is what is called the link function. 

Thus, the link function is the relationship between the population mean and the linear predictor of a model
$$s(\mu) = x'\beta$$
As an example, we could have a log link function:
$$\displaylines{ln(\mu) = x'\beta\\
\mu = e^{x'\beta}}$$
> "One should view GLM modeling as making a choice of distribution and link function."

The link function can be seen as the function that "bends the line" of a linear model.

### 🔹 Default
Different distributions have different default link functions. This doesn't at all mean that some distributions must be confined to some specific link functions, different ones must be tested to achieve the best result.

## 🔷 Random Component
Refers to the distribution you're assuming to make your predictions, calculate metrics etc. The choice of distribution mostly depends on the problem you're dealing (/ you're trying to model)


# 🔵 Where to use cases
The thumb rule is: **Know your distributions**. Everything you learn in statistics of how a distribution works, when it should be used and what kind of problems it describes are applied here - that's just how you chose what distribution you should use. 

Here are some examples of when to use some distributions and link functions, with respect to different problems:

| Name | Problem | Distribution Name |
| ---- | ---- | ---- |
| Binomial Distribution | Output is binary, only two options | Logistic Regression |
| Poisson Distribution | You have a skewed discrete distribution, for example, the number of times cars passes by you at a given length of time. (assumes mean = variance)  | Poisson Regression |
|  | It describes the same problems a Poisson would, with the difference that it doesn't assume the variance is equal to the mean. | Negative Binomial |
- For example, this means that the residuals of the logistic regression follows the binomial distribution

![[Pasted image 20240116000456.png]]


