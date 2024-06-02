 
 # Expeerimenr 2: Regularization effect
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

 
# Set up the parameters
np.random.seed(42)
num_feat = 600
n_repeats = 5
train_n_values = np.arange(10,  2*num_feat + 1, 10)  # Different values of training set size to use
alpha_values = [0, 0.1, 0.01, 1, 100, 1000,10000]  # Different values of regularization parameter to use
test_errors = np.zeros((len(alpha_values), len(train_n_values), n_repeats))


def generate_data(n, num_feat):
    return np.random.normal(0, 1, size=(n, num_feat))

def generate_labels(X, a_true):
    return np.sign(X.dot(a_true))

def compute_regression(X, y, alpha):
    A = np.dot(X.T, X) + alpha * np.eye(num_feat)
    B = np.dot(X.T, y)
    return np.linalg.solve(A, B)

def compute_test_error(X_test, y_test, a):
    y_pred = np.sign(X_test.dot(a))
    return np.mean(y_pred != y_test)

for i, alpha in enumerate(alpha_values):
    for j, train_n in enumerate(train_n_values):
        for k in range(n_repeats):
            X_train = generate_data(train_n, num_feat)
            a_true = generate_data(num_feat, 1)
            y_train = generate_labels(X_train, a_true)

            a = compute_regression(X_train, y_train, alpha)

            test_n = 1000
            X_test = generate_data(test_n, num_feat)
            y_test = generate_labels(X_test, a_true)

            test_errors[i, j, k] = compute_test_error(X_test, y_test, a)
            

mean_test_errors = np.mean(test_errors, axis=2)
std_test_errors = np.std(test_errors, axis=2)


# Iterate over each alpha value and plot separately
for i, alpha in enumerate(alpha_values):
    # Create a new figure and axis for each alpha value
    fig, ax = plt.subplots()

    # Plot the test errors with error bars for the current alpha value
    ax.errorbar(train_n_values, mean_test_errors[i], yerr=std_test_errors[i], label=r'$\alpha$ = {}'.format(alpha))

    # Set the x-axis and y-axis labels
    ax.set_xlabel('Training set size')
    ax.set_ylabel('Test error')

    # Set the title of the plot
    ax.set_title('Regularization(alpha = {})'.format(alpha))

    # Add a grid
    ax.grid(True)

    # Add legend
    ax.legend()

    # Show the plot for the current alpha value
    plt.show()
 
