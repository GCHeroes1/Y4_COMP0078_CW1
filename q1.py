import matplotlib.pyplot as plt
import numpy as np
from numpy import arange
from polynomial_regression import PolynomialRegression

x_data = np.array([[1, 2, 3, 4]])
y_data = np.array([3, 2, 0, 5])


def constant(a):
    return a


def linear(x, a, b):
    return a + (b * x)


def quadratic(x, a, b, c):
    return a + (b * x) + (c * x ** 2)


def cubic(x, a, b, c, d):
    return a + (b * x) + (c * x ** 2) + (d * x ** 3)


def get_y_hats(x_data, objective_function, *args):
    y_hats = []
    for x in x_data:
        y_hats.append(objective_function(x, *args))
    y_hats = np.array(y_hats)
    return y_hats


def get_mse(y, y_hat):
    errors = y - y_hat
    squared_errors = np.square(errors)
    mse = sum(squared_errors) / len(squared_errors)
    return round(mse, 2)


if __name__ == '__main__':
    plt.scatter(x=[1, 2, 3, 4], y=[3, 2, 0, 5])

    ########
    # Fi_1 #
    ########
    print("Fi_1")
    model_fi_1 = PolynomialRegression(degree=0)
    model_fi_1.fit(x_data.T, y_data)
    weights = model_fi_1.weights
    equation_text = f'y = %.2f' % (weights[0])
    print(f'Fitted Curve: {equation_text}')

    # calculate MSE
    y_hats = [weights[0] for _ in range(len(x_data[0]))]
    mse = get_mse(y_data, y_hats)
    print(f"MSE: {mse}")

    # Plotting
    plt.plot(x_data[0], y_hats, label=equation_text + f" ; MSE: {mse}",
             color='black', linestyle='dashed')

    print("=========")

    ########
    # Fi_2 #
    ########
    print("Fi_2")
    model_fi_2 = PolynomialRegression(degree=1)
    model_fi_2.fit(x_data.T, y_data)
    weights = model_fi_2.weights
    a, b = weights
    equation_text = f'%.2f + %.2f X' % (a, b)
    print(f'Fitted Curve: {equation_text}')

    # calculate MSE
    y_hats = get_y_hats(x_data[0], linear, a, b)
    mse = get_mse(y_data, y_hats)
    print(f"MSE: {mse}")

    # Plotting
    x_line = arange(1, 4.5, 0.1)
    y_line = linear(x_line, a, b)
    plt.plot(x_line, y_line, label=equation_text + f" ; MSE: {mse}",
             color='blue', linestyle='dashed')

    print("=========")

    ########
    # Fi_3 #
    ########
    print("Fi_3")
    model_fi_3 = PolynomialRegression(degree=2)
    model_fi_3.fit(x_data.T, y_data)
    weights = model_fi_3.weights
    a, b, c = weights
    equation_text = f'%.2f + %.2f X + %.2f X^2' % (a, b, c)
    print(f'Fitted Curve: {equation_text}')

    # calculate MSE
    y_hats = get_y_hats(x_data[0], quadratic, a, b, c)
    mse = get_mse(y_data, y_hats)
    print(f"MSE: {mse}")

    # Plotting
    x_line = arange(1, 4.5, 0.1)
    y_line = quadratic(x_line, a, b, c)
    plt.plot(x_line, y_line, label=equation_text + f" ; MSE: {mse}",
             color='green', linestyle='dashed')

    print("=========")

    ########
    # Fi_4 #
    ########
    print("Fi_4")
    model_fi_4 = PolynomialRegression(degree=3)
    model_fi_4.fit(x_data.T, y_data)
    weights = model_fi_4.weights
    a, b, c, d = weights
    equation_text = f'%.2f + %.2f X + %.2f X^2 + %.2f X^3' % (a, b, c, d)
    print(f'Fitted Curve: {equation_text}')

    # calculate MSE
    y_hats = get_y_hats(x_data[0], cubic, a, b, c, d)
    mse = get_mse(y_data, y_hats)
    print(f"MSE: {mse}")

    # Plotting
    x_line = arange(1, 4.5, 0.1)
    y_line = cubic(x_line, a, b, c, d)
    plt.plot(x_line, y_line, label=equation_text + f" ; MSE: {mse}",
             color='red', linestyle='dashed')

    plt.title("polynomial regression for different base of dimension")
    plt.legend()
    plt.savefig('./plots/q1a.png')
