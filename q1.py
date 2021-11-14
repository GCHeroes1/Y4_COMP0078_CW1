from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
from numpy import arange
import seaborn as sns

# data = {(1, 3), (2, 2), (3, 0), (4, 5)}
x_data = [1, 2, 3, 4]
y_data = [3, 2, 0, 5]
sns.set_theme()


def constant(x, a, b):
    return 1


def linear(x, a, b):
    return a * x + b


def quadratic(x, a, b, c):
    return (a * x) + (b * x ** 2) + c


def cubic(x, a, b, c, d):
    return (a * x) + (b * x ** 2) + (c * x ** 3) + d


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
    return mse


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Fi_1
    sns.scatterplot(x=x_data, y=y_data)
    pars, _ = curve_fit(f=constant, xdata=x_data, ydata=y_data)
    sns.lineplot(x_data, constant(x_data, *pars), linestyle='--', linewidth=2, color='black')

    mse = get_mse(y_data, np.array([1, 1, 1, 1]))
    print(f"MSE for Fi_1: {mse}")

    # Fi_2
    popt, _ = curve_fit(linear, x_data, y_data)
    a, b = popt
    print('y = %.5f * x + %.5f' % (a, b))
    x_line = arange(1, 4, 0.1)
    y_line = linear(x_line, a, b)
    sns.lineplot(x=x_line, y=y_line, linestyle='--', linewidth=2, color='blue')

    y_hats = get_y_hats(x_data, linear, a, b)
    mse = get_mse(y_data, y_hats)
    print(f"MSE for Fi_2: {mse}")

    # Fi_3
    popt, _ = curve_fit(quadratic, x_data, y_data)
    a, b, c = popt
    print('y = %.5f * x + %.5f * x^2 + %.5f' % (a, b, c))
    x_line = arange(1, 4.5, 0.1)
    y_line = quadratic(x_line, a, b, c)
    sns.lineplot(x=x_line, y=y_line, linestyle='--', linewidth=2, color='red')

    y_hats = get_y_hats(x_data, quadratic, a, b, c)
    mse = get_mse(y_data, y_hats)
    print(f"MSE for Fi_3: {mse}")

    # Fi_4
    popt, _ = curve_fit(cubic, x_data, y_data)
    a, b, c, d = popt
    print('y = %.5f * x + %.5f * x^2 + %.5f + x^3 + %.5f' % (a, b, c, d))  # THIS IS THE SAME AS IN THE BRIEF :D
    x_line = arange(1, 4.5, 0.1)
    y_line = cubic(x_line, a, b, c, d)
    sns.lineplot(x=x_line, y=y_line, linestyle='--', linewidth=2, color='green', legend='auto')

    y_hats = get_y_hats(x_data, cubic, a, b, c, d)
    mse = get_mse(y_data, y_hats)
    print(f"MSE for Fi_4: {mse}")

    plt.title("polynomial fitting for each base of dimension")
    # TODO: lengend with sns...
    plt.savefig('./plots/q1a.png')
