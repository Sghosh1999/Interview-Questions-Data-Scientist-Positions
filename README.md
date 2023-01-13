



![data_science_topics](https://user-images.githubusercontent.com/44112345/211920662-d87c5754-b60e-4eb4-9d93-5b7324421b1e.JPG)

# Interview Questions - Data Scientist Positions ( Entry, Mid & Senior)

> This repository contains the curated list of topic wise questions for Data Scientist Positions in various companies. <br />

  ⭐     - Entry Level positions <br />
  ⭐⭐   - Mid Level positions <br />
  ⭐⭐⭐ - Senior positionsLevel


| Topics (Ongoing) | No Of Questions |
|--|--|
| [Data Science & ML - General Topics](#data-science--ml---general-topics) | 29 |
| [Regression Techniques](#regression-techniques--concepts-) | 20 |
| [Classification Techniques](#classification-techniques--concepts-) | 10* |


### Data Science & ML - General Topics

1. **What is the basic difference between AI, Machine Learning(ML) & Deep Learning(DL)?**. ⭐


`Ans:` Artificial Intelligence (AI) is a broad field that encompasses many different techniques and technologies, including machine learning (ML) and deep learning (DL). <br />
- **Artificial Intelligence (AI)** refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. It is a broad field that includes many different approaches and techniques, such as rule-based systems, and expert systems. <br />
- **Machine Learning (ML)** is a subfield of AI that is focused on the development of algorithms and statistical models that enable machines to learn from data and make predictions or decisions without being explicitly programmed. <br />
1- **Deep Learning (DL)** is a type of machine learning that is inspired by the structure and function of the brain's neural networks. It uses multiple layers of artificial neural networks to learn representations of data with multiple levels of abstraction. DL algorithms can be used for tasks such as image and speech recognition, natural language processing, and decision-making.
   
![al_ml_dl](https://user-images.githubusercontent.com/44112345/212048865-51478738-282a-46b5-b6ad-952fdb3abebd.JPG)

---
   
2. **Can you explain the difference between supervised and unsupervised learning?**. ⭐


`Ans:` The main difference between them is the type and amount of input provided to the algorithms.
- **Supervised learning** is a type of machine learning where the model is trained on a labeled dataset, i.e., the model is provided with input-output pairs, and the goal is to learn a mapping from inputs to outputs. This mapping can then be used to make predictions on new, unseen data. Examples of supervised learning include regression, classification and prediction tasks. <br />
- **Unsupervised learning**, on the other hand, is a type of machine learning where the model is not provided with labeled data. Instead, the algorithm is given a dataset without any output labels, and the goal is to find patterns or structure within the data. Examples of unsupervised learning include clustering, dimensionality reduction, and anomaly detection tasks.

---

3.  **Can you explain the difference between supervised and unsupervised learning?**. ⭐


`Ans:` There are several techniques for handling missing data in a dataset, some of the most common include: <br />
- **Mean/Median Imputation:** This method replaces the missing value with the mean or median of the non-missing values in the same column. (Numerical)
- **Random Sample Imputation:** This method replaces the missing value with a random sample from the non-missing values in the same column. (Numerical)
- **Most Frequent Imputation** Most Frequent is another statistical strategy to impute missing values which work with categorical features (strings or numerical representations) by replacing missing data with the most frequent values within each column. (Numerical/Categorical)
- **Zero or Constant Imputation** By this method, the missing values are replaced by zero/constant values.(Numerical)
- **Regression Imputation:** By this method, the missing values are predicted using Regression techniques such as KNN Imputation, and Logistic Regression for Categorical missing values.(Numerical/Categorical)
-  **Multivariate Imputation by Chained Equation (MICE):** This type of imputation works by filling in the missing data multiple times. Multiple Imputations (MIs) are much better than a single imputation as it measures the uncertainty of the missing values in a better way.

---

4.  **How do you select the appropriate evaluation metric for a given problem, and what are the trade-offs between different metrics such as precision, recall, and F1-score?**. ⭐


`Ans:` Selecting the appropriate evaluation metric for a given problem depends on the characteristics of the data and the goals of the model. Here are some common evaluation metrics and the situations in which they are typically used:

- **Accuracy:** This metric measures the proportion of correct predictions made by the model. It is commonly used when the classes are well balanced and the goal is to have a high overall performance of the model.
    
- **Precision:** This metric measures the proportion of true positive predictions to all positive predictions made by the model. It is commonly used when the goal is to minimize false positives. (Ex- Financial Transactions/ Spam Mail)
    
- **Recall:** This metric measures the proportion of true positive predictions to all actual positive cases in the dataset. It is commonly used when the goal is to minimize false negatives. (Ex-Medical diseases/Stock Market breakdown)
    
- **F1-Score:** This metric is the harmonic mean of precision and recall. It balances the trade-off between precision and recall and is commonly used when the goal is to have a balance between both.
    
- **ROC-AUC:** This metric measures the area under the Receiver Operating Characteristic curve and is commonly used when the classes are imbalanced and the goal is to have a high overall performance of the model.

---

5. **What is the beta value implies in the F-beta score? What is the optimum beta value?**. ⭐⭐


   `Ans:` The F-beta score is a variant of the F1-score, where the beta value controls the trade-off between precision and recall. The F1-score is a harmonic mean of precision and recall and is calculated as `(2 * (precision * recall)) / (precision + recall).`
- A beta value of 1 is equivalent to the F1 score, which means it gives equal weight to precision and recall. 
- A beta value less than 1 gives more weight to precision, which means it will increase the importance of precision over recall. 
- A beta value greater than 1 gives more weight to recall, which means it will increase the importance of recall over precision.
		![fbeta](https://user-images.githubusercontent.com/44112345/212004531-62545b14-f6cc-4de2-9e63-184e4e223c3c.JPG)

---


6.  **What are the advantages & disadvantages of Linear Regression?**.⭐


   `Ans:`
- Advantages:
	- Linear regression is a simple and interpretable algorithm that is easy to understand and implement.
	- It is a well-established technique that has been widely used for many years and is a good benchmark for other more complex algorithms.
	- Linear regression can handle large datasets with many input features and it's relatively fast and efficient to train.
	- It's easy to identify and measure the relationship between input features and target variables by looking at the coefficients.

- Disadvantages:
	- Linear regression assumes that the relationship between the input features and the target variable is linear, which may not always be the case in real-world problems.
	- It can't capture non-linear relationships between variables.
	- Linear regression is sensitive to outliers, as a single outlier can greatly impact the line.
	- Linear regression does not account for errors in the observations.
	- Linear regression can be affected by multicollinearity, which occurs when input features are highly correlated with each other.

---

7.  **How do you handle categorical variables in a dataset?**.⭐


`Ans:`Handling categorical variables in a dataset is an important step in the preprocessing of data before applying machine learning models. Here are some common techniques for handling categorical variables:
- **One-Hot Encoding:** This method creates a new binary column for each unique category in a categorical variable. Each row is then encoded with a 1 or 0 in the corresponding column, depending on the category. *This method is useful when there is no ordinal relationship between categories.*
- **Ordinal Encoding:** This method assigns an integer value to each category in a categorical variable. This method is useful when there is an ordinal relationship between categories.
- **Binary Encoding:** This method assigns a binary code to each category. *This method is useful when the number of categories is high and one-hot encoding creates too many columns.*
- **Count Encoding:** This method replaces each category with the number of times it appears in the dataset.
- **Target Encoding:** This method replaces each category with the mean of the target variable for that category.
- **Frequency Encoding:** This method replaces each category with the frequency of the category in the dataset.
    It's important to note that some of the techniques, like One-Hot Encoding, Ordinal Encoding, and Binary Encoding, have the potential to introduce a new feature, which could affect the model performance. *Additionally, target encoding and count encoding could introduce a leakage from the target variable, which could lead to overfitting.

---
  
8.  **What is the curse of dimensionality and how does it affect machine learning?**.⭐


`Ans:`The curse of dimensionality refers to the problem of increasing complexity and computational cost in high-dimensional spaces. In machine learning, the curse of dimensionality arises when the number of features in a dataset is large relative to the number of observations. This can cause problems for several reasons:
- **Sparsity:** With a high number of features, most observations will have many missing or zero values for many of the features. This can make it difficult for models to learn from the data.
- **Overfitting:** With a high number of features, models are more likely to fit the noise in the data rather than the underlying patterns. This can lead to poor performance on new, unseen data.
- **Computational cost:** High-dimensional spaces require more memory and computational power to store and process the data. This can make it difficult to train models and make predictions in real-world applications.

---
9.  **What are the approaches to mitigate Dimensionality reduction?**.⭐


`Ans:`These are some mechanisms to deal with Dimensionality reduction,
- Techniques like **principal component analysis (PCA), linear discriminant analysis (LDA), or t-distributed stochastic neighbor embedding (t-SNE)** can be used to reduce the number of features by combining or selecting a subset of the original features.
- **Regularization:** Techniques like L1 or L2 regularization can help prevent overfitting by adding a penalty term to the model's objective function that discourages the model from fitting to noise in the data.
- **Sampling:** With high-dimensional data, it is often infeasible to use all the data. In such cases, random sampling could be used to reduce the size of the data to work with.
- **Ensemble methods:** Ensemble methods like random forests and gradient boosting machines can be used to reduce overfitting and improve generalization performance in high-dimensional data.

---

10. **Can you explain the bias-variance tradeoff?**.⭐


`Ans:`The bias-variance tradeoff is a fundamental concept in machine learning that describes the trade-off between how well a model fits the training data (bias) and how well the model generalizes to new, unseen data (variance). <br/>
- Bias refers to the error introduced by approximating a real-world problem, which may be incredibly complex, with a much simpler model. High-bias models are typically considered to be "oversimplified" and will have a high error on the training set.
- On the other hand, variance refers to the error introduced by the model's sensitivity to small fluctuations in the training data. High variance models are typically considered to be "overcomplicated" or "overfit" and will have a high error on the test set.
		![bias-varinace_trade_off](https://user-images.githubusercontent.com/44112345/211999865-304f95fe-852b-42f0-b826-d827ebfad906.JPG)

---
		
11.  **How do you prevent overfitting in a model?**.⭐


`Ans:` Overfitting occurs when a model is too complex and captures the noise in the training data, instead of the underlying patterns. This can lead to poor performance on new, unseen data. Here are some common techniques for preventing overfitting: 
- **Regularization:** Techniques *like L1, L2 regularization, or dropout,* add a penalty term to the model's objective function that discourages the model from fitting to noise in the data.
- **Early stopping:** This technique is used when training deep neural networks, it monitors the performance of the model on a validation set and stops the training process when the performance on the validation set starts to degrade.
- **Cross-validation:** This technique involves *dividing the data into several subsets and training the model on different subsets while evaluating the model performance on the remaining subsets.* This technique helps to get a better estimate of the model's performance on unseen data.
- **Ensemble methods:** Ensemble methods like random forests and gradient boosting machines can be used to reduce overfitting and improve generalization performance. Ensemble methods combine the outputs of multiple models to produce a more robust prediction.
- **Feature selection or dimensionality reduction:** By reducing the number of features, it can decrease the complexity of the model and prevent overfitting.
- **Simplifying the model:** By *simplifying the model architecture or reducing the number of parameters,* it can decrease the complexity of the model and prevent overfitting.

---

12.  **What is Hypothesis Testing. Explain with proper example.**.⭐


   `Ans:` 
Hypothesis testing is a statistical method used to determine whether a claim or hypothesis about a population parameter is true or false. The process of hypothesis testing involves making an initial assumption or hypothesis about the population parameter and then using sample data to test the validity of that assumption.
- There are two types of hypotheses in hypothesis testing: **the null hypothesis (H0)** and the **alternative hypothesis (H1).** The null hypothesis states that there is no difference or relationship between the population parameter of interest and the sample data, while the alternative hypothesis states that there is a difference or relationship.
- The process of hypothesis testing involves several steps:
	1.  State the null and alternative hypotheses
	2.  Choose a level of significance (alpha)
	3.  Collect and analyze sample data
	4.  Calculate a test statistic and its corresponding p-value
	5.  Compare the p-value to the level of significance (alpha)
	6.  Make a decision and interpret the results. <br/>
    `For example, let's say a company wants to know if the mean weight of their product is equal to 50 grams. The null hypothesis (H0) would be that the mean weight of the product is equal to 50 grams, and the alternative hypothesis (H1) would be that the mean weight of the product is not equal to 50 grams. The company would then take a sample of products and calculate the mean weight of the sample. Using statistical methods, they would determine if the sample mean is statistically significantly different from 50 grams. If the sample mean is statistically significantly different from 50 grams, the company would reject the null hypothesis and conclude that the mean weight of the product is not equal to 50 grams.` <br/>

---


13. **What is Type 1 & Type 2 error?**.⭐


   `Ans:` *Type 1 error, **also known as a false positive, occurs when the null hypothesis is rejected, but it is actually true.** In other words, **it is a mistake of rejecting a null hypothesis that is true.** The probability of making a Type 1 error is represented by the level of significance (alpha) chosen before the hypothesis test. A common level of significance is 0.05, which means that there is a 5% chance of making a Type 1 error.
- For example, in a medical test to detect a disease, a Type 1 error would be a false positive, where the test says a patient has the disease, but they actually do not.
Type 2 error, also known as a false negative, occurs ***when the null hypothesis is not rejected, but it is actually false.*** In other words, **it is a mistake of not rejecting a null hypothesis that is false.** The probability of making a Type 2 error is represented by the beta (beta). A common level of beta is 0.2, which means that there is a 20% chance of making a Type 2 error.
- For example, in a medical test to detect a disease, a Type 2 error would be a false negative, where the test says a patient does not have the disease, but they actually do.

---

14. **Explain some of the Statistical test's use cases (Ex- 2 Tail test, T-Test, Anona test, Chi-Squared test)**.⭐


`Ans:` The use cases of the tests are as follows,<br/>
- **t-test:** A t-test is used to determine if there is a significant difference between the means of two groups. There are several types of t-tests, including independent samples t-test, dependent samples t-test, and one-sample t-test. It is commonly used for comparing the means of two samples or for comparing a sample mean to a known population mean.
- **ANOVA (Analysis of Variance):** ANOVA is used to determine if there is a significant difference between the means of two or more groups. There are several types of ANOVA, including one-way ANOVA, two-way ANOVA, and repeated measures ANOVA. It is commonly used for comparing means of multiple samples or for comparing a sample mean to multiple known population means.
- **Chi-Square test:** A Chi-Square test is used to determine if there is a significant association between two categorical variables. It is commonly used for testing independence in contingency tables and for goodness of fit tests.

---

15. **What do you mean when the p-values are high and low?**.⭐


`Ans:`In hypothesis testing, the p-value is used to estimate the probability of obtaining a test statistic as extreme or more extreme than the one observed, assuming that the null hypothesis is true.
- A **low p-value (typically less than 0.05) indicates that the evidence against the null hypothesis is strong and that the null hypothesis is unlikely to be true given the data.** In other words, a low p-value suggests that the sample data is unlikely to have occurred by chance alone and that the observed difference is statistically significant.
- A **high p-value (typically greater than 0.05) indicates that the evidence against the null hypothesis is weak and that the null hypothesis is likely to be true given the data.** In other words, a high p-value suggests that the sample data is likely to have occurred by chance alone and that the observed difference is not statistically significant.

---

16. **What is the significance of KL Divergence in Machine Learning?**.⭐⭐


`Ans:` KL divergence (also known as Kullback-Leibler divergence) is a measure of the difference between two probability distributions. In machine learning, KL divergence is used to measure the similarity or dissimilarity between two probability distributions, usually between the estimated distribution and the true distribution.
- One of the **main uses of KL divergence in machine learning is in the field of unsupervised learning, particularly in the training of generative models.** For example, Variational Autoencoder(VAE) and Generative Adversarial Networks(GANs) use KL divergence as a loss function to measure the difference between the estimated distribution and the true distribution of the data.
- **KL divergence is a popular measure for comparing probability distributions because it is a non-symmetric measure, which means that the KL divergence between distribution P and Q is not the same as the KL divergence between Q and P.** This makes it useful for comparing distributions that have different support sets. Additionally, it is a computationally efficient measure, and it is easy to calculate the gradient of KL divergence, which makes it suitable for optimization.

---

17. **How could you deal with data skewness? What are the approaches to resolve the skewness in the data?**.⭐


`Ans:`Skewness is a measure of the asymmetry of a probability distribution. 
- When the *skewness is positive, it indicates that the distribution has a long tail on the right side and the mode is less than the mean, which is also less than the median.* 
- When *the skewness is negative, it indicates that the distribution has a long tail on the left side and the mode is greater than the mean, which is also greater than the median.*
-  **There are several ways to deal with skewness in data:**
	1. Data Transformation: Data transformations like logarithmic, square root, and reciprocal can help to reduce skewness in the data. Logarithmic transformation is generally used for data that is positively skewed, square root and reciprocal for data that is negatively skewed.
	2. Binning: Binning is a method of grouping continuous variables into a smaller number of categories. It can be used to reduce the effect of outliers and to make the data more symmetrical.

---

18. **What is IQR? How it is been used to detect Outliers?**.⭐


`Ans:` IQR stands for interquartile range. It is a measure of the spread of a dataset that is based on the difference between the 75th percentile (Q3) and the 25th percentile (Q1) of the data. To calculate IQR, you first need to find the median (Q2) of the data and then subtract Q1 from Q3.
	![IQR](https://user-images.githubusercontent.com/44112345/212009455-fe60d8b1-ed02-45d4-9ea0-da4f8e1baca8.JPG)
- Outliers can be detected using the IQR method by calculating the lower and upper bounds of the data. The lower bound is defined as Q1 - 1.5 * IQR, and the upper bound is defined as Q3 + 1.5 * IQR. Any data points that fall outside of this range are considered to be outliers.
- It is important to note that while the IQR method is a useful tool for identifying outliers, it is not always the best method to use, especially when the data has a non-normal distribution. Other methods such as the Z-score method should also be considered.

---

19. **What are the algorithms that are sensitive & Robust to Outliers?**.⭐


`Ans:` There are several algorithms that are considered to be robust to outliers:
 - **The Median Absolute Deviation (MAD) method:** This is a robust measure of statistical dispersion that is based on the median of the absolute deviations from the median, rather than the mean.
 - **The Huber loss function:** This is a robust cost function that is less sensitive to outliers than the mean squared error (MSE) function.
 - **The RANSAC algorithm:** This is a robust estimation method that is designed to fit a model to a dataset that contains outliers.
 - **Random Forest:** Random Forest creates multiple decision trees, and each tree is trained on different random subset of features and observations, it is less sensitive to outliers than a single decision tree.
 - **Gradient Boosting Machine (GBM):** GBM is also a collection of decision trees, it is less sensitive to outliers because it combines the decision of multiple weak learners.
 - **Support Vector Machines (SVMs):** SVMs are robust to outliers because they try to find the largest margin between the classes, it is less affected by the presence of outliers.

---

20. **Why Feature Scaling is important? What are the feature scaling techniques?**.⭐


`Ans:` Feature scaling is an important step in the preprocessing of data for machine learning algorithms because it helps to standardize the range of independent variables or features of the data.
 - MinMaxScaler: This class provides a transformer to scale the data set in a given range, usually between 0 and 1.
 - StandardScaler: This class provides a transformer to standardize the data set, by centering and scaling.
 - RobustScaler: This class provides a transformer to scale the data set using statistics that are robust to outliers.

---

21. **If we don't remove the highly correlated values in the dataset, how does it impact the model performance?**.⭐⭐

`Ans:` If you don't remove high correlated values from your dataset, it can have a negative impact on the performance of your model.
 - **Overfitting:** The model will try to fit the noise in the data and this will result in poor generalization performance.
 - **Instability:** The model's parameters and coefficients may change dramatically with small changes in the training data.
 - **Difficulty in interpreting the model:** If two or more variables are highly correlated, it becomes difficult to interpret which variable is more important for the model.
 - **Difficulty in model optimization:** High correlation can lead to slow convergence of the model's optimization process, making it difficult to find the optimal solution.
 - **Large variance and low bias:** High correlation can lead to large variance and low bias in the model, which may result in an over-complicated model that is prone to overfitting.

Removing high correlated variables before training the model can help to improve the model's performance by removing multicollinearity and reducing the complexity of the model.

---

22. **What is Spearman Correlation? What do you mean by positive and negative Correlation?**.⭐


`Ans:`Spearman correlation is a measure of the statistical dependence between two variables, it is also known as the Spearman rank correlation coefficient. *It is a non-parametric measure of correlation, which means that it does not assume that the underlying distribution of the data is normal. Instead, it calculates the correlation between the ranks of the data points.*
 - Spearman correlation coefficient ranges from -1 to 1. **A coefficient of 1 indicates a perfect positive correlation, meaning that as one variable increases, the other variable also increases.** **A coefficient of -1 indicates a perfect negative correlation, meaning that as one variable increases, the other variable decreases.** A coefficient of 0 indicates no correlation between the two variables.

---

23.  **What is the difference between Co-Variance & Correlation?**.⭐⭐


`Ans:` Covariance `is a measure of the degree to which two random variables change together. It can be positive, negative or zero.` A positive covariance means that the variables increase or decrease together, a negative covariance means that as one variable increases, the other variable decreases and a zero covariance means that there is no relationship between the two variables. The formula for covariance is:
														Cov(X, Y) = (1/n) * Σ(x - x̄) * (y - ȳ)
where X and Y are the two random variables, x̄ and ȳ are the means of X and Y, respectively, and n is the number of observations.
- **Correlation, on the other hand,** is a standardized version of covariance. It is a value between -1 and 1 that indicates the strength and direction of the linear relationship between two variables.The formula for correlation is:
														Corr(X, Y) = Cov(X, Y) / (σX * σY)
where σX and σY are the standard deviations of X and Y, respectively.
In summary, covariance is a measure of how two variables change together while correlation is a standardized version of covariance that describes the strength and direction of the linear relationship between two variables.

---

24. **What is the difference between Multiclass Classification Models & Multilabel Classification Models?.**⭐


`Ans:` In multiclass classification, the goal is to classify instances into one of several predefined classes. For example, classifying images of animals into dog, cat, and horse classes. Each instance can only belong to one class, and the classes are mutually exclusive.

Multilabel classification, on the other hand, is a problem where each instance can belong to multiple classes simultaneously. For example, classifying news articles into multiple topics, such as sports, politics, and technology. In this case, an article can belong to the sports and politics class simultaneously.

---

25. **Can you explain the concept of ensemble learning?**.⭐


`Ans:` Ensemble learning is a technique in machine learning where multiple models are combined to create a more powerful model. The idea behind ensemble learning is to combine the predictions of multiple models to create a final prediction that is more accurate and robust than the predictions of any individual model.

---

26. **What are the different ensembling Modeling Strategies in ML?**.⭐


`Ans:` There are several ensemble learning techniques, including:
  - **Bagging:** This technique *involves training multiple models independently on different subsets of the data* and then averaging or voting on their predictions. This can be useful for reducing the variance of a model.
  - **Boosting:** This technique *involves training multiple models sequentially, where each model is trained to correct the mistakes of the previous model.* This can be useful for reducing the bias of a model.
  - **Stacking:** This technique involves training multiple models independently on the same data and then combining their predictions in a second-level model.
  - **Blending:** This technique is similar to stacking, *but it involves training the second-level model on a subset of the data rather than on the predictions of the first-level models.*

---
27. **How do you select features for a machine-learning model?**.⭐


`Ans:` There are several feature selection algorithms used in machine learning, including:
- **Filter Methods:** This approach is based on the correlation between the features and the target variable. It includes methods like *correlation coefficient, mutual information and Chi-Squared.*
-  **Wrapper Methods:** This approach is based on evaluating the performance of a subset of features in a model. It includes methods like *forward selection, backward elimination, recursive feature elimination and genetic algorithms.*
- **Embedded Methods:** This approach is based on training the model with all the features and then selecting the relevant features based on their contribution to the performance of the model. It includes methods like Lasso regularization and Ridge regularization.
- **Hybrid Methods:** This approach combines the strengths of filter and wrapper methods.
- **Mutual Information Feature Selection:** This approach selects feature by maximizing the mutual information between the feature and the target variable.

---
28.  **What are the dimensionality Reduction techniques in Machine Learning?**.⭐⭐


`Ans:` There are several dimensionality reduction techniques in machine learning, including:
- Principal Component Analysis (PCA)
- Singular Value Decomposition (SVD)
- Linear Discriminant Analysis (LDA)
- Independent Component Analysis (ICA)
- t-Distributed Stochastic Neighbor Embedding (t-SNE)
- UMAP (Uniform Manifold Approximation and Projection)
- Autoencoder (AE)
		8.  Variational Autoencoder (VAE)
These techniques are used to reduce the number of features in a dataset while preserving as much information as possible.

---
29. **Explain the PCA steps in machine learning.**.⭐⭐


`Ans:`Principal Component Analysis (PCA) is a widely used dimensionality reduction technique in machine learning. It is a linear technique that transforms the original dataset into a new set of uncorrelated variables called principal components. The goal of PCA is to find the directions (principal components) that capture the most variation in the data. The following are the steps for performing PCA:
-  Data Preprocessing: PCA is performed on the centered data, thus the first step is to center the data by subtracting the mean from each feature.
- Covariance Matrix: The next step is to calculate the covariance matrix of the centered data. The covariance matrix is a square matrix that contains the variances and covariances of the features.
- Eigenvalues and Eigenvectors: Calculate the eigenvalues and eigenvectors of the covariance matrix. Eigenvectors are the directions of maximum variation in the data, and eigenvalues are the corresponding magnitudes of this variation.
- Principal Components: Select the top k eigenvectors that have the highest eigenvalues, where k is the number of dimensions you want to reduce to. These eigenvectors are the principal components of the data.
- Dimensionality Reduction: The last step is to project the centered data onto the principal components using a dot product. This results in a new dataset with reduced dimensions (k dimensions) that captures the most variation in the original data.`
---

📙 [Back to Top Section ](#interview-questions---data-scientist-positions--entry-mid--senior)

## Regression Techniques ( Concepts )

1. **Can you explain the difference between simple linear regression and multiple linear regression? How do you decide which one to use for a given problem?**.⭐⭐


`Ans:`Simple linear regression is a type of linear regression *where the **target variable is predicted using a single predictor variable.** The relationship between the predictor variable and the target variable is represented by a straight line.* The equation for a simple linear regression model is:

    y = b0 + b1*x

Where y is the target variable, x is the predictor variable, b0 is the y-intercept, and b1 is the slope of the line.

Multiple linear regression, on the other hand, *is a type of linear regression where the target variable **is predicted using multiple predictor variables.** The relationship between the predictor variables and the target variable is represented by a hyperplane.* The equation for a multiple linear regression model is:

    y = b0 + b1_x1 + b2_x2 + ... + bn*xn

Where y is the target variable, x1, x2, ..., xn are the predictor variables, b0 is the y-intercept, and b1, b2, ..., bn are the coefficients of the predictor variables.
- When deciding which type of linear regression to use, you should consider **the number of predictor variables and the relationship between the predictor variables and the target variable.** If there is only one predictor variable and the relationship between the predictor variable and the target variable is approximately linear, then simple linear regression is a good choice. If there are multiple predictor variables and the relationship between the predictor variables and the target variable is approximately linear, then multiple linear regression is a good choice.
---

2. **What are the assumptions of linear regression and how do you check if they hold for a given dataset?**.⭐


`Ans:`The assumptions of linear regression are:
- **Linearity:** The relationship between the predictor variables and the target variable is linear. This means that a straight line can be used to approximate the relationship.
- **Independence of errors:** The *errors (residuals) of the model are independent and identically distributed.* This means that the errors are not correlated and have the same distribution.
- **Homoscedasticity:** The *variance of the errors is constant across all levels of the predictor variables.* This means that the spread of the errors is the same for all levels of the predictor variables.
- **Normality:** *The errors are normally distributed.* This means that the distribution of the errors follows a normal (or Gaussian) distribution.
- **No multicollinearity:** *The predictor variables are not highly correlated with each other.* This means that the predictor variables are independent of each other.
---
3. **How do you handle categorical variables in linear regression?**.⭐


`Ans:`Categorical imputations are the techniques that can be performed before fitting into the model.
- **OneHot Encoding:** One-hot encoding is a technique that converts a categorical variable into multiple binary variables, one for each category.
- **Ordinal Encoding:** Ordinal encoding is a technique that assigns an integer value to each category of a categorical variable. This method is useful when the categories have an ordinal relationship.
- **Effect Encoding:** Effect Encoding is a technique that represents each categorical variable as a set of contrasts between the different levels of that variable and a reference level.
---

4. **What is the difference between Lasso & Ridge regression?**.⭐


`Ans:` Both Lasso and Ridge regression are types of linear regression, but they have different approaches to solving the problem of overfitting, which occurs when a model is too complex and captures the noise in the data as well as the underlying relationship.
- **Lasso (Least Absolute Shrinkage and Selection Operator) regression** also known as L1 regularization, adds *a penalty term to the cost function equal to the absolute value of the magnitude of the coefficients.* Lasso Regression is useful to select important features of a dataset, *it will shrink the less important feature's coefficient to zero and it's generally used where the number of features is high.*
![lasso](https://user-images.githubusercontent.com/44112345/212061356-9f6af866-a433-4200-888d-9ca9f4fd7b88.JPG)

- **Ridge regression, also known as L2 regularization,** is another type of linear regression that uses a penalty term *equal to the square of the magnitude of the coefficients.* Instead of shrinking the coefficients to zero, this penalty term helps to *shrink the coefficients of less important features towards zero, but it doesn't make them exactly zero.* Ridge Regression is useful to handle multicollinearity problems by reducing the variance of the coefficients.

![ridge](https://user-images.githubusercontent.com/44112345/212061376-6b221eec-8281-4a69-9781-d2a4a14701c2.JPG)

---
5. **How do we select the right regularization parameters?**.⭐

`Ans:` Regularization is a technique used to prevent overfitting by adding a penalty term to the cost function of a model. The regularization term is controlled by a parameter, called the regularization parameter or hyperparameter. The value of this parameter determines the strength of the regularization.
- There are several ways to select the right regularization parameter:
	- **Grid Search:** Grid search is a technique that involves training the model with a range of different regularization parameter values and evaluating the performance of each one using a validation set. The regularization parameter value that results in the best performance is chosen as the final value.
	- **LassoCV:** LassoCV is a variant of Lasso Regression that uses cross-validation to select the regularization parameter. It performs the grid search and cross-validation internally and returns the best regularization parameter.
	- **RidgeCV:** RidgeCV is a variant of Ridge Regression that uses cross-validation to select the regularization parameter. It performs the grid search and cross-validation internally and returns the best regularization parameter.

---
6. **Why is Logistic Regression called a Linear Model?**.⭐

`Ans:` Logistic Regression is called a linear model because the relationship between the input features and the output variable is linear. The model is represented by a linear equation of the form:

    y = b0 + b1_x1 + b2_x2 + ... + bn*xn
where y is the output variable (a probability between 0 and 1), x1, x2, ..., xn are the input features, and b0, b1, b2, ..., bn are the model coefficients. The coefficients represent the contribution of each feature to the output variable.
However, it's important to note that the logistic regression model is used for classification problem, the output variable (y) is the probability of the input belongs to a certain class, and this probability is modeled by a logistic function (sigmoid function) which is non-linear.

---
7. **Can Logistic regression can be used in an Imbalanced dataset Problem?**.⭐⭐

`Ans:` Logistic regression can be used in an imbalanced dataset problem, but it is important to be aware of the limitations and potential issues that can arise.
- One potential issue with using logistic regression on imbalanced datasets is *that the model may be biased towards the majority class.* **This means that the model will have a higher accuracy, but it will not perform well on the minority class.** This can be addressed by using techniques such as oversampling the minority class, undersampling the majority class, or using a combination of both.
- Another **potential issue with using logistic regression on imbalanced datasets is that the model may not be sensitive to the minority class.** This means that the model will have a low recall for the minority class. This can be addressed by adjusting the threshold for classifying an example as the minority class.

Additionally, it's important to use different evaluation metrics other than accuracy such as precision, recall, F1-score, AUC, etc.

---
8. **Can you explain the concept of polynomial regression? How does it differ from linear regression?**.⭐⭐

`Ans:` **Polynomial regression** is a type of regression analysis in which the relationship between the *independent variable x and the dependent variable y is modeled as an nth degree polynomial.*
A polynomial regression model can be represented by an equation of the form:

    y = b0 + b1x + b2x^2 + b3x^3 + ... + bnx^n

where y is the output variable, x is the input variable, and b0, b1, b2, ..., bn are the model coefficients. The coefficients represent the contribution of each degree of x to the output variable.
- *The main difference between polynomial regression and linear regression is the form of the equation. Linear regression uses a linear equation of the form y = b0 + b1*x*, while polynomial regression uses a polynomial equation that can model non-linear relationships between the input and output variables.

---
9. **How do you deal with outliers in linear regression? What are the most common techniques for identifying and dealing with outliers?**.⭐⭐

`Ans:` Outliers in linear regression can have a significant impact on the model parameters and can lead to poor model performance. Here are some common techniques for identifying and dealing with outliers:
- **Visualization:** One of the simplest ways to identify outliers is by visualizing the data using scatter plots, box plots, or histograms. Outliers will typically appear as points that are far away from the main cluster of data points.
- **Z-score:** The Z-score is a measure of how many standard deviations an observation is from the mean. Observations with a Z-score greater than a certain threshold (usually 3) can be considered outliers.
- **Cook's distance: Cook's distance is a measure of the influence of each observation on the model parameters. Observations with a high Cook's distance can be considered outliers.**
- **Outlier removal:** Once the outliers are identified, they can be removed from the dataset. However, this method should be used with caution, as it can lead to loss of important information and can bias the model.
- **Robust regression:** Robust regression is a technique that is less sensitive to outliers. The most common method of robust regression is the least median of squares (LMS) method.
- **Winsorization:** Winsorization is a technique that replaces the outliers with a certain value. For example, replacing all the observations greater than 3 standard deviations from the mean with the value of 3 standard deviations from the mean.

---
10. **What are the most common evaluation metrics in Regression Problems?**.⭐

`Ans:` There are several common evaluation metrics used in regression problems, some of the most popular include:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE):
- R-squared (R2):
- Mean Absolute Percentage Error (MAPE):
- Adjust R-squared (Adjusted R²)

---

11. **How R-squared is different with Adjusted R²? What is the main difference**.⭐

`Ans:` R-squared (R²) and adjusted R-squared (adjusted R²) are both measures of the goodness of fit of a regression model, but they are slightly different.
- **R-squared is a measure of the proportion of variance in the dependent variable that can be explained by the independent variable(s) in the model.** It ranges from 0 to 1, where a value of 1 indicates that the model perfectly explains the variance in the dependent variable. **R-squared is calculated as the ratio of the explained variance to the total variance.**
- **Adjusted R-squared is an extension of R-squared that adjusts for the number of predictor variables in the model.** The main difference between R-squared and adjusted R-squared is that adjusted R-squared penalizes the model for the number of predictor variables that are not useful in explaining the variance in the dependent variable.
- This makes *adjusted R-squared a more conservative measure of the goodness of fit of a regression model, as it penalizes models with a large number of predictor variables* that do not explain a significant amount of variance in the dependent variable.
![r2](https://user-images.githubusercontent.com/44112345/212057791-5c0993d2-53d0-414b-b1e1-30b972ea888a.JPG) 
![Adjustedr2](https://user-images.githubusercontent.com/44112345/212057510-cdcd90a9-a378-492b-b783-7e8c1cea1657.JPG)


---
12. **Can you explain the concept of Elastic Net regression? How does it differ from Ridge and Lasso regression?**.⭐⭐

`Ans:`Elastic Net is a linear regression model that combines the properties of both Ridge and Lasso regression. Like Ridge, it adds a L2 regularization term to the cost function to prevent overfitting. Like Lasso, it adds a L1 regularization term to the cost function to perform feature selection. *The trade-off between the L1 and L2 regularization terms is controlled by a parameter, alpha, that ranges from 0 to 1. **When alpha = 0, the model becomes a Ridge regression, and when alpha = 1, it becomes a Lasso regression.***

---
13. **In a sparse dataset where most of the values are 0, which supervised classification algorithm we should use?**.⭐⭐

`Ans:`When dealing with a sparse dataset where most of the values are zero, a suitable supervised classification algorithm to use could be the **Naive Bayes Classifier, especially the variant called Multinomial Naive Bayes,** because **it can handle the large number of zero values and it relies on counting the occurrences of the features, this method can work well when a dataset is sparse** and the classifier can learn useful information from the occurrences of the features.
It uses multinomial distributions for the likelihood estimates of the features, which models the occurrences which are robust to the sparse data.

---

14. **What are the regularization techniques other than L1 and l2 in machine learning?**.⭐

`Ans:`**Elastic Net Regularization:** This is a combination of *L1 and L2 regularization, where a linear combination of the L1 and L2 penalties is used in the cost function.* It helps to balance the trade-off between sparsity and shrinkage.
- **Bayesian Regularization:** This method is a Bayesian version of regularization, this method uses the probabilistic approach to regularize the model by assuming a prior distribution on the parameters, this will help to prevent overfitting by reducing the variance in the parameters.

---

15.   **Which evaluation metrics are sensitive to the Outliers?**.

`Ans:`Some evaluation metrics that are sensitive to outliers include **mean absolute error, mean squared error.** These metrics can be greatly affected by the presence of outliers in the data, as they take into account the individual differences between the predicted and actual values. **On the other hand, median absolute error and coefficient of determination (R-squared) are resistant to the outliers.**

---
16.   **What is the difference between bagging and boosting?**.⭐

`Ans:`Bagging and Boosting are two ensemble methods that are used to improve the performance of machine learning models.
- **Bagging (Bootstrap Aggregating)** is a method that involves *training multiple models on different subsets of the training data, and then averaging (for regression) or voting (for classification) the predictions made by each model.* The subsets of the training data are created by randomly sampling the data with replacement. Bagging helps to reduce the variance of the model, making it less prone to overfitting.
- **Boosting** is a method that involves training multiple models *in a sequential manner, where each model tries to correct the errors made by the previous model.* The idea is to give more weight to the data points that are misclassified by the previous models, so that the next model can focus on those points. Boosting helps to reduce the bias of the model, making it more accurate.

---

17.   **What is the difference between bagging and boosting?**.⭐

`Ans:`Bagging and Boosting are two ensemble methods that are used to improve the performance of machine learning models.
- **Bagging (Bootstrap Aggregating)** is a method that involves *training multiple models on different subsets of the training data, and then averaging (for regression) or voting (for classification) the predictions made by each model.* The subsets of the training data are created by randomly sampling the data with replacement. Bagging helps to reduce the variance of the model, making it less prone to overfitting.
- **Boosting** is a method that involves training multiple models *in a sequential manner, where each model tries to correct the errors made by the previous model.* The idea is to give more weight to the data points that are misclassified by the previous models, so that the next model can focus on those points. Boosting helps to reduce the bias of the model, making it more accurate.

---
18.   **How does Decision Tree regressor works?**.⭐

`Ans:`A Decision Tree Regressor is a supervised learning algorithm that is used to *predict a continuous target variable based on several input features.* The algorithm works by recursively splitting the data into subsets based on the values of the input features. **Each split is made in such a way as to maximize the reduction in impurity of the target variable.**

---
19.   **Why don’t we use Mean Squared Error as a cost function in Logistic Regression?**.⭐

`Ans:`In summary, MSE is a cost function for linear regression and it's not a suitable cost function for logistic regression because *it's not a good measure of the difference between predicted probabilities (between 0 & 1) and true class labels.* The log loss or cross-entropy loss is more appropriate cost function for logistic regression because it penalizes predictions that are confident but incorrect.

---
20. **How can we avoid Over-fitting in Logistic Regression models?**

`Ans:`Regularization, Pruning, Cross-validation, Early stopping, and Ensemble methods are some of the techniques that can be used to avoid overfitting in logistic regression models.

---

📙 [Back to Top Section ](#interview-questions---data-scientist-positions--entry-mid--senior)

## Classification Techniques ( Concepts )

1.   **When would you use SVM vs Logistic regression?**.⭐⭐⭐

`Ans:`Support Vector Machine (SVM) and Logistic Regression are both supervised learning algorithms that can be used for classification tasks. However, there are some key differences between the two that may influence which one you choose to use for a given problem.
-   **Linear vs Non-linear decision boundary:** *SVM can learn non-linear decision boundaries by using the kernel trick, while logistic regression can only learn linear decision boundaries.* This means that SVM is more suitable for problems where the data is not linearly separable, while logistic regression is more suitable for problems where the data is linearly separable. 
-   **Outliers:** *SVM is less sensitive to outliers,* as it only depends on the support vectors (i.e., the data points that are closest to the decision boundary). *Logistic regression, on the other hand, is more sensitive to outliers,* as the model parameters are estimated based on the entire dataset.
    
-   **Large dataset:** *SVM can be less efficient when dealing with large dataset* because of the high computational cost of the algorithm. *Logistic regression is computationally cheaper and can handle large dataset.*
    
-   **Multi-class classification:** *Logistic Regression can be easily extended to multi-class classification problems* by training multiple binary classifiers and combining them, while SVM requires a different approach for multi-class classification.

---
