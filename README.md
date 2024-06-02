# Double Descent Phenomenon in Linear Regression

## Overview

This repository provides an implementation of the double descent phenomenon in the context of linear regression. It includes experiments to understand how different model complexities and regularization parameters affect model performance.

# What is Double Descent Phenomenon

The double descent phenomenon refers to the behavior of the test error as a function of model complexity in certain machine learning models. Unlike the traditional U-shaped bias-variance trade-off curve, the double descent curve shows two descent phases:

1.Initial Descent: As model complexity increases, the test error initially decreases due to reduced bias.

2.Peak (Interpolation Threshold): Beyond a certain point, the model becomes overly complex, and the test error increases due to overfitting.

3.Second Descent: With further increase in complexity, the test error decreases again, even below the levels observed in the initial descent phase. This happens because highly complex models, often with a high number of parameters, can capture the underlying data structure more effectively in high-dimensional settings.

# Experiments

Model Complexity
The experiment varies the size of the training set to study how increasing model complexity affects the performance of linear regression models. Different sizes of training sets are used to simulate underfitting and overfitting regimes.

## Regularization
The experiment also studies the effect of different regularization parameters (
α
α) on the model's performance. Ridge regression (L2 regularization) is applied to understand how regularization can mitigate overfitting and affect the double descent curve.

## Results

The results are visualized using plots that show the test error as a function of training set size for different regularization strengths. These plots help to illustrate the double descent phenomenon and the impact of regularization on model performance.

 
