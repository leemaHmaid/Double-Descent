#Double Descent Phenomenon in Linear Regression

Overview

This repository provides an implementation of the double descent phenomenon in the context of linear regression. It includes experiments to understand how different model complexities and regularization parameters affect model performance.
 
#Experiments

Model Complexity
The experiment varies the size of the training set to study how increasing model complexity affects the performance of linear regression models. Different sizes of training sets are used to simulate underfitting and overfitting regimes.

Regularization
The experiment also studies the effect of different regularization parameters (
α
α) on the model's performance. Ridge regression (L2 regularization) is applied to understand how regularization can mitigate overfitting and affect the double descent curve.

Results

The results are visualized using plots that show the test error as a function of training set size for different regularization strengths. These plots help to illustrate the double descent phenomenon and the impact of regularization on model performance.

Contributing

Contributions are welcome! Please follow these steps:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature-name).
Commit your changes (git commit -am 'Add new feature').
Push to the branch (git push origin feature/your-feature-name).
Create a new Pull Request.
