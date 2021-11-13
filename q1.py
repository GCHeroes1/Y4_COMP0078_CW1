from scipy.optimize import curve_fit
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy import arange
from pylab import cm
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
    return (a * x) + (b * x**2) + c

def cubic(x, a, b, c, d):
    return (a * x) + (b * x**2) + (c * x**3) + d

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    plot1 = plt.figure(1)
    sns.scatterplot(x=x_data, y=y_data)
    pars, cov = curve_fit(f=constant, xdata=x_data, ydata=y_data, bounds=(-np.inf, np.inf))
    sns.lineplot(x_data, constant(x_data, *pars), linestyle='--', linewidth=2, color='black')

    plot2 = plt.figure(2)
    sns.scatterplot(x=x_data, y=y_data)
    # pars, cov = curve_fit(f=linear, xdata=x_data, ydata=y_data, bounds=(-np.inf, np.inf))
    # pars, cov = curve_fit(f=exponential, xdata=x_dummy, ydata=y_dummy, p0=[0, 0], bounds=(-np.inf, np.inf))
    # sns.lineplot(x_data, linear(x_data, *pars), linestyle='--', linewidth=2, color='black')
    # popt, _ = curve_fit(linear, x_data, y_data)
    popt, _ = curve_fit(linear, x_data, y_data)
    a, b = popt
    print('y = %.5f * x + %.5f' % (a, b))
    x_line = arange(1, 5, 0.1)
    y_line = linear(x_line, a, b)
    sns.lineplot(x=x_line, y=y_line, linestyle='--', linewidth=2, color='black')

    plot3 = plt.figure(3)
    sns.scatterplot(x=x_data, y=y_data)
    popt, _ = curve_fit(quadratic, x_data, y_data)
    a, b, c = popt
    print('y = %.5f * x + %.5f * x^2 + %.5f' % (a, b, c))
    x_line = arange(1, 5, 0.1)
    y_line = quadratic(x_line, a, b, c)
    sns.lineplot(x=x_line, y=y_line, linestyle='--', linewidth=2, color='black')

    plot4 = plt.figure(4)
    sns.scatterplot(x=x_data, y=y_data)
    popt, _ = curve_fit(cubic, x_data, y_data)
    a, b, c, d = popt
    print('y = %.5f * x + %.5f * x^2 + %.5f + x^3 + %.5f' % (a, b, c, d)) # THIS IS THE SAME AS IN THE BRIEF :D
    x_line = arange(1, 5, 0.1)
    y_line = cubic(x_line, a, b, c, d)
    sns.lineplot(x=x_line, y=y_line, linestyle='--', linewidth=2, color='black')



    plt.show()
