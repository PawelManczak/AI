import numpy as np
import matplotlib.pyplot as plt
from numpy import square

from data import get_data, inspect_data, split_data

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 andtheta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()

# TODO: calculate closed-form solution
yVector = np.array(y_train).reshape((-1, 1))  # zamiana z pionowego na poziomy wektor
xVector = np.array(x_train).reshape((-1, 1))  # analogicznie
xVector = np.append(np.ones([xVector.shape[0], 1], dtype=np.float64), xVector,
                    axis=1)  # dopisujemy na początek nasze jedynki

theta = xVector.T.dot(xVector)
theta = np.linalg.inv(theta)
theta = theta.dot(xVector.T).dot(yVector)

print(theta)

theta_best = theta

# TODO: calculate error
yCalculated = np.array(x_train).reshape((-1, 1))

myFun = lambda x: theta[0] + theta[1] * x

sum = 0
for yCalc, yRes in zip(yCalculated, yVector):
    yCalc = myFun(yCalc)
    sum += square(np.subtract(yRes, yCalc))

MSE = sum / (np.size(yCalculated))

print('MSE: {MSE}')


# plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
# plt.show()

# TODO: standardization

# TODO: calculate theta using Batch Gradient Descent

# TODO: calculate error

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
# plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
# plt.show()
