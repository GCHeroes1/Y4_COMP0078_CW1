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

def equation_10(n, data, weight, predictions, lamb):
	# w* = argmin _ (for all w in real) * 1/l *
	return (1/n * np.sum(np.matmul(data.T, weight) - predictions)**2 + lamb * np.matmul(weight.T, weight))

if __name__ == '__main__':
	data = np.arange(9, 18).reshape((3, 3))
	weight = np.arange(3).reshape((3, 1))
	predictions = np.arange(6, 9).reshape((3, 1))
	lamb = 5
	print(data)
	print(weight)
	print(predictions)
	print(equation_10(3, data, weight, predictions, lamb))
