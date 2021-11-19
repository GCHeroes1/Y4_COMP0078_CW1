import numpy
import numpy as np
import q4
from numpy.linalg import inv
import csv
import random
import math
import polynomial_regression as regression
import matplotlib.pyplot as plt

# def argmin(array):
# argmin returns the indicies of the minimum element
# return np.argmin(array)

# def sum(array):
# 	return np.sum(array)

# def summation(start, end, data):
# 	sum = 0
# 	for i in range (start, end+1):
# 		sum += data[i]
# 	return sum
gamma_values = [2 ** (-40), 2 ** (-39), 2 ** (-38), 2 ** (-37), 2 ** (-36), 2 ** (-35), 2 ** (-34), 2 ** (-33),
                2 ** (-32), 2 ** (-31), 2 ** (-30), 2 ** (-29), 2 ** (-28), 2 ** (-27), 2 ** (-26)]
sigma_values = [2 ** 7, 2 ** 7.5, 2 ** 8, 2 ** 8.5, 2 ** 9, 2 ** 9.5, 2 ** 10, 2 ** 10.5, 2 ** 11, 2 ** 11.5, 2 ** 12,
                2 ** 12.5, 2 ** 13]


def nested_summation_alphastar(l, alpha, x, y):
	sum = 0
	inner_sum = 0
	for i in range(l + 1):
		for j in range(l + 1):
			inner_sum += alpha[j - 1] * kernelK(x, i, j) - y[i]
		sum += inner_sum ** 2
		inner_sum = 0
	return sum


def kernelK(x, i, j):
	# request for k_ij is the inner product of x_i and x_j
	return np.inner(x[i - 1], x[j - 1])


def kernelKConstruction(x):
	size = len(x)
	kernel = []
	for i in range(size):
		for j in range(size):
			kernel.append(kernelK(x, i, j))
	return np.reshape(kernel, (size, size))


def equation_10(n, data, weight, predictions, gamma):
	# w* = argmin _ (for all w in real) * 1/l *
	sum = 0
	for i in range(n):
		math = data.T[i] * weight - predictions[i]
		sum += math ** 2
	part_2 = gamma * np.matmul(weight.T, weight)
	return np.argmin(1 / n * sum + part_2)


# return np.argmin(1/n * np.sum(np.matmul(data.T, weight) - predictions)**2 + lamb * np.matmul(weight.T, weight))

def equation_11(n, gamma, alpha, x, y): # alpha star dual optimisation
	sum_1 = np.argmin(1 / n * nested_summation_alphastar(n, alpha, x, y))
	sum_2 = gamma * np.matmul(np.matmul(alpha.T, kernelKConstruction(x)), alpha)
	return sum_1 + sum_2


def calc_alpha_star(x, gamma, n, y): #alpha star, solved dual
	kernel = kernelKConstruction(x)
	identity = np.identity(n)
	part_1 = (kernel + gamma * n * identity).inv
	return part_1.dot(y)
	# return np.matmul(part_1, y)


def calc_y_test(x, y, gamma, test, n):  # generating y_test
	y_test = 0
	for i in range(n):
		math = calc_alpha_star(x, gamma, n, y)[i] * kernelK(x, i, test)
		y_test += math
	return y_test


def gaussian_kernel(x, i, j, sigma): #eq 14, the gaussian kernel
	numerator = numpy.abs(x[i] - x[j]) ** 2
	denominator = 2 * (sigma ** 2)
	return np.exp(-numerator / denominator)

def five_fold_dataset(data):
	random.shuffle(data)
	split = math.floor(len(data)/5)
	one, two, three, four, five = [], [], [], [], []
	one.append(data[0: split])
	two.append(data[split:2*split])
	three.append(data[2*split:3*split])
	four.append(data[3*split:4*split])
	five.append(data[4*split:5*split])
	return one, two, three, four, five

def five_fold_validation(data, gamma_values, sigma_values):
	one, two, three, four, five = five_fold_dataset(data)
	five_fold_data = [one, two, three, four, five]
	# need to make predictions with each gamma and sigma value and calculate MSE for each permutation
	for gamma in gamma_values:
		for sigma in sigma_values:
			MSE = 0
			for i in range (5): #5 fold validation
				testing_data = five_fold_data[i]
				training_data = []
				for j in range (5):
					if j != i:
						for data in five_fold_data[j]:
							training_data.append(data)
			# now we have the training data and testing data, but need to also split them to get the y value (13th value)





if __name__ == '__main__':
	data = np.arange(9, 18).reshape((3, 3))
	weight = np.arange(3).reshape((3, 1))
	predictions = np.arange(6, 9).reshape((3, 1))
	print(data)
	print(data[1][1])
# lamb = 5
# print(data)
# print(weight)
# print(predictions)
# print(equation_10(3, data, weight, predictions, lamb))
