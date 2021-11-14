import matplotlib.pyplot as plt
import numpy as np
from numpy import arange
from polynomial_regression import PolynomialRegression

x_data = np.array([[1, 2, 3, 4]])
y_data = np.array([3, 2, 0, 5])


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
    y_hats = model_fi_1.predict(x_data.T)
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
    y_hats = model_fi_2.predict(x_data.T)
    mse = get_mse(y_data, y_hats)
    print(f"MSE: {mse}")

    # Plotting
    x_line = arange(1, 4.5, 0.1)
    y_line = model_fi_2.predict(np.array([x_line]).T)
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
    y_hats = model_fi_3.predict(x_data.T)
    mse = get_mse(y_data, y_hats)
    print(f"MSE: {mse}")

    # Plotting
    x_line = arange(1, 4.5, 0.1)
    y_line = model_fi_3.predict(np.array([x_line]).T)
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
    y_hats = model_fi_4.predict(x_data.T)
    mse = get_mse(y_data, y_hats)
    print(f"MSE: {mse}")

    # Plotting
    x_line = arange(1, 4.5, 0.1)
    y_line = model_fi_4.predict(np.array([x_line]).T)
    plt.plot(x_line, y_line, label=equation_text + f" ; MSE: {mse}",
             color='red', linestyle='dashed')

    plt.title("polynomial regression for different base of dimension")
    plt.legend()
    plt.savefig('./plots/q1a.png')
