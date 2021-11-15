import numpy as np
from numpy.linalg import inv
import csv
import random
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

def nested_summation_alphastar(l, alpha, x, y):
	sum = 0
	inner_sum = 0
	for i in range (l+1):
		for j in range (l+1):
			inner_sum+= alpha[j-1] * kernelK(x, i, j) - y[i]
		sum += inner_sum ** 2
		inner_sum = 0
	return sum

def kernelK(x, i, j):
	#request for k_ij is the inner product of x_i and x_j
	return np.inner(x[i-1], x[j-1])

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
	return np.argmin(1/n * sum + part_2)
	# return np.argmin(1/n * np.sum(np.matmul(data.T, weight) - predictions)**2 + lamb * np.matmul(weight.T, weight))

def equation_11(n, gamma, alpha, x, y):
	sum_1 = np.argmin(1/n * nested_summation_alphastar(n, alpha, x, y))
	sum_2 = gamma * np.matmul(np.matmul(alpha.T, kernelKConstruction(x)), alpha)
	return sum_1 + sum_2

def equation_12(x, gamma, n, y):
	kernel = kernelKConstruction(x)
	identity = np.identity(n)
	part_1 = (kernel + gamma * n * identity).inv
	return np.matmul(part_1, y)

def equation_13(x, y, gamma, test, n):
	sum = 0
	for i in range(n):
		math = equation_12(x, gamma, n, y)[i] * kernelK(x, i, test)
		sum += math
	return sum

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
