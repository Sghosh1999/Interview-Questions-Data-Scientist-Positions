# Regression Algorithms

This document contains an overview of various regression algorithms commonly used in machine learning.

## Table of Contents

1. [Linear Regression](#linear-regression)
2. [Ridge Regression](#ridge-regression)
3. [Lasso Regression](#lasso-regression)
4. [Polynomial Regression](#polynomial-regression)
5. [Support Vector Regression (SVR)](#support-vector-regression-svr)
6. [Decision Tree Regression](#decision-tree-regression)
7. [Random Forest Regression](#random-forest-regression)
8. [Gradient Boosting Regression](#gradient-boosting-regression)

---

## Linear Regression

Linear Regression is one of the most fundamental and widely used algorithms in statistics and machine learning. It models the relationship between a dependent variable and one or more independent variables by fitting a linear equation to observed data.

---

## 📌 When is Linear Regression Used?

Linear Regression is primarily used when the goal is to **predict a continuous numeric outcome** based on one or more predictor variables. It is effective when:

- There is a **linear relationship** between the input(s) and the output.
- You want a **simple, interpretable model**.
- Estimating the **impact of one variable on another** is important.

### 📈 Real-World Use Cases:

- Predicting **house prices** based on features like area, location, and number of rooms.
- Estimating **sales revenue** from advertising budgets.
- Modeling **growth trends** over time.
- Assessing **risk scores** in credit scoring or insurance.

---

## 🧠 Assumptions of Linear Regression

To produce reliable results, linear regression makes several key assumptions:

1. **Linearity**
   The relationship between the independent and dependent variable(s) is linear.

2. **Independence of Errors**
   Observations are independent of each other. Residuals (errors) are not correlated.

3. **Homoscedasticity**
   The variance of error terms is constant across all levels of the independent variables.

4. **Normality of Errors**
   The residuals (differences between observed and predicted values) are normally distributed.

5. **No Multicollinearity** _(for multiple linear regression)_
   Independent variables should not be too highly correlated with each other.

6. **No Autocorrelation**
   Particularly important for time-series data: residuals should not show patterns over time.

---

## ✅ Pros of Linear Regression

- 🔍 **Simple and Interpretable**: Easy to implement and understand.
- 🚀 **Fast Training**: Computationally efficient even on large datasets.
- 📊 **Statistical Foundation**: Offers insight into variable relationships via coefficients and p-values.
- 🔎 **Feature Importance**: Coefficients directly indicate influence of predictors.
- 🛠️ **Baseline Model**: Often serves as a benchmark for more complex models.

---

## ❌ Cons of Linear Regression

- 📉 **Poor Performance on Non-Linear Data**: Cannot model complex patterns without transformation.
- 🎯 **Sensitive to Outliers**: Outliers can heavily skew results.
- 🔄 **Requires Feature Engineering**: Needs manual work to handle categorical features or interactions.
- 🔬 **Assumption-Dependent**: Violations of assumptions (e.g., multicollinearity or heteroscedasticity) degrade model quality.
- ⚖️ **Overfitting with High-Dimensional Data**: Especially in multiple regression without regularization.

---

## 🧮 Mathematical Formulation

![Simple Linear Regression Example](../images/linear-1.jpg)

### 2. **Multiple Linear Regression**

For multiple input features $x_1, x_2, ..., x_n$:

$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n + \varepsilon
$$

In matrix form:

$$
\mathbf{y} = \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\varepsilon}
$$

Where:

- $\mathbf{y} \in \mathbb{R}^{m}$ is the output vector
- $\mathbf{X} \in \mathbb{R}^{m \times n}$ is the matrix of input features
- $\boldsymbol{\beta} \in \mathbb{R}^{n}$ is the coefficient vector
- $\boldsymbol{\varepsilon} \in \mathbb{R}^{m}$ is the error vector

### 3. **Cost Function (Loss Function)**

Linear regression minimizes the **Mean Squared Error (MSE)**:

$$
J(\boldsymbol{\beta}) = \frac{1}{m} \sum_{i=1}^{m} \left( y_i - \hat{y}_i \right)^2
= \frac{1}{m} \sum_{i=1}^{m} \left( y_i - (\mathbf{x}_i^T \boldsymbol{\beta}) \right)^2
$$

### 4. **Normal Equation Solution**

For optimal coefficients:

$$
\boldsymbol{\hat{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
$$

---

## Ridge Regression

Ridge regression adds a penalty term to the linear regression cost function to reduce overfitting.

**Key Points:**

- Uses L2 regularization.
- Helps in handling multicollinearity.

---

## Lasso Regression

Lasso regression introduces L1 regularization, which can shrink some coefficients to zero.

**Key Points:**

- Performs feature selection.
- Useful for sparse models.

---

## Polynomial Regression

Polynomial regression extends linear regression by fitting a polynomial equation to the data.

**Key Points:**

- Captures non-linear relationships.
- Prone to overfitting with high-degree polynomials.

---

## Support Vector Regression (SVR)

SVR uses the principles of Support Vector Machines for regression tasks.

**Key Points:**

- Effective in high-dimensional spaces.
- Can use different kernel functions.

---

## Decision Tree Regression

Decision tree regression splits the data into regions and fits a constant value in each region.

**Key Points:**

- Easy to interpret.
- Prone to overfitting without pruning.

---

## Random Forest Regression

Random forest regression is an ensemble method that combines multiple decision trees.

**Key Points:**

- Reduces overfitting.
- Handles missing data well.

---

## Gradient Boosting Regression

Gradient boosting regression builds models sequentially to minimize errors.

**Key Points:**

- Highly accurate.
- Computationally expensive.

---
