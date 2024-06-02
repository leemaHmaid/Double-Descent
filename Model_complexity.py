
#Experiment1:Model Complexity and Dataset size:
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#Generate random data for demonstration

np.random.seed(42)
num_samples = 200
num_features = 2000
X = np.random.rand(num_samples, num_features)
y = np.random.rand(num_samples)

#Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Create and train the linear regression model for different model complexities

complexities = [1,10,20,30,40,100,1000,2000]  # Different model complexities
train_loss = []
test_loss = []
for complexity in complexities:
    # linear regression model
    model = LinearRegression()
    model.fit(X_train[:, :complexity], y_train)

    # Predict on the training set and calculate the training loss
    y_train_pred = model.predict(X_train[:, :complexity])
    train_loss.append(mean_squared_error(y_train, y_train_pred))

    # Predict on the test set and calculate the test loss
    y_test_pred = model.predict(X_test[:, :complexity])
    test_loss.append(mean_squared_error(y_test, y_test_pred))
  
#Plot the train loss and test loss against model complexity
plt.plot(complexities, train_loss, marker='o', label='Train Loss')
plt.plot(complexities, test_loss, marker='o', label='Test Loss')
plt.xlabel('Model Complexity')
plt.ylabel('Mean Squared Error')
plt.title('Train Loss and Test Loss vs. Model Complexity')
plt.legend()
plt.show()
