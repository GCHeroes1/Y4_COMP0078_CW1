import numpy as np
import pandas as pd
from numpy.linalg import inv
import random
from polynomial_regression import PolynomialRegression
from q1 import get_mse
import matplotlib.pyplot as plt
from sys import exit
import os


def get_raw_data() -> np.ndarray:
	try:
		data = pd.read_csv('./Boston-filtered.csv')
	except FileNotFoundError:
		data = pd.read_csv('http://www.cs.ucl.ac.uk/staff/M.Herbster/boston-filter/Boston-filtered.csv')

	return data.values


# 4a - Naive Regression

def sample_training(dataset, train_size: int):
	"""
    Randomly sample training set and test set
    :param dataset: numpy.ndarray dataset
    :param train_size: int valued size of training dataset
    :return: numpy.ndarray typed train set and test set
    """
	sample_start = random.randint(0, len(dataset))

	training_set = []
	testing_set = []
	if sample_start < len(dataset) - train_size:
		for i in range(len(dataset)):
			if sample_start <= i < (sample_start + train_size):
				training_set.append(dataset[i])
			else:
				testing_set.append(dataset[i])
	else:
		# this is when the starting point would take u out of bounds
		for i in range(len(dataset)):
			if i < (train_size - (len(dataset) - sample_start)) or i >= sample_start:
				training_set.append(dataset[i])
			else:
				testing_set.append(dataset[i])

	return np.array(training_set), np.array(testing_set)


def constant_predictor(X, y):
	# print(X)
	# return np.average(X)
	X_transpose = np.transpose(X)
	X_inverse = inv(X_transpose * X)
	return X_inverse


if __name__ == '__main__':

	data = get_raw_data()

	# get number of train and test set
	train_size = int(data.shape[0] * 0.66)
	test_size = data.shape[0] - train_size
	print(f"{train_size} Train sets and {test_size} Test sets")

	##################
	# Q4a: Naive Regression
	###################
	naive_regression_model = PolynomialRegression(need_transform=False)

	train_errors = []
	test_errors = []
	for run in range(20):
		# each run is based on a different (0.66, 0.33) random split
		train, test = sample_training(data, train_size)

		# Split vertically for X and Y
		X_train = np.ones((train_size, 1))
		Y_train = train[:, -1]
		X_test = np.ones((test_size, 1))
		Y_test = test[:, -1]

		# Fit Naive Regression model without transformation
		naive_regression_model.fit(X_train, Y_train)

		# get predicted result from both train data and test data
		y_hat_train = naive_regression_model.predict(X_train)  # (333, 1) -> (333,)
		y_hat_test = naive_regression_model.predict(X_test)  # (173, 1) -> (173,)

		# get MSE for each train and test
		mse_train = get_mse(Y_train, y_hat_train)
		mse_test = get_mse(Y_test, y_hat_test)

		# log the error for each run
		train_errors.append(mse_train)
		test_errors.append(mse_test)

	if not os.path.exists('logs'):
		os.makedirs('logs')

	with open('./logs/q4a.txt', 'w') as f:
		f.write(f"Train errors (MSE): {str(train_errors)}\n")
		f.write(f"Average Train error: {np.mean(train_errors)}\n")
		f.write("\n")
		f.write(f"Test errors (MSE): {str(test_errors)}\n")
		f.write(f"Average Test error: {np.mean(test_errors)}\n")
		f.close()

	del naive_regression_model

	##################
	# Q4c: Linear Regression with single attributes
	###################
	single_attribute_regression_model = PolynomialRegression(need_transform=False)

	train_errors_attributes = np.zeros((data.shape[1] - 1, 20))
	test_errors_attributes = np.zeros((data.shape[1] - 1, 20))
	for attribute in range(data.shape[1] - 1):
		train_errors = []
		test_errors = []
		for run in range(20):
			# each run is based on a different (0.66, 0.33) random split
			train, test = sample_training(data, train_size)

			# Split vertically for X and Y
			X_train = train[:, attribute, np.newaxis]
			X_test = test[:, attribute, np.newaxis]
			Y_train = train[:, -1]
			Y_test = test[:, -1]

			# augment with an additional bias term
			X_train = np.hstack((X_train, np.ones((train_size, 1))))
			X_test = np.hstack((X_test, np.ones((test_size, 1))))

			# print(X_train.shape, X_test.shape)  # (333, 2) (173, 2)
			# print(Y_train.shape, Y_test.shape)  # (333,) (173,)

			# Fit single attribute Regression model
			single_attribute_regression_model.fit(X_train, Y_train)

			# get predicted result from both train data and test data
			y_hat_train = single_attribute_regression_model.predict(X_train)
			y_hat_test = single_attribute_regression_model.predict(X_test)

			# get MSE for each train and test
			mse_train = get_mse(Y_train, y_hat_train)
			mse_test = get_mse(Y_test, y_hat_test)

			# log the error for each run
			train_errors.append(mse_train)
			test_errors.append(mse_test)

		# log the 20 errors for each attribute
		train_errors_attributes[attribute] = train_errors
		test_errors_attributes[attribute] = test_errors

	with open('./logs/q4c.txt', 'w') as f:
		# iterate through attributes
		for i in range(data.shape[1] - 1):
			train_errors_att_i = train_errors_attributes[i]
			train_error_att_i_mean = np.mean(train_errors_att_i)

			test_errors_att_i = test_errors_attributes[i]
			test_error_att_i_mean = np.mean(test_errors_att_i)

			f.write(f"Attribute {i+1}\n")
			f.write(f"Average train error: {train_error_att_i_mean}\n")
			f.write(f"Average test error: {test_error_att_i_mean}\n")
			f.write("\n")

	del single_attribute_regression_model

	##################
	# Q4d: Linear Regression using all attributes
	###################
	all_att_regressor = PolynomialRegression(need_transform=False)

	train_errors = []
	test_errors = []
	for run in range(20):
		# each run is based on a different (0.66, 0.33) random split
		train, test = sample_training(data, train_size)

		# Split vertically for X and Y
		X_train = train[:, :-1]
		Y_train = train[:, -1]
		X_test = test[:, :-1]
		Y_test = test[:, -1]

		# augment with an additional bias term
		X_train = np.hstack((X_train, np.ones((train_size, 1))))
		X_test = np.hstack((X_test, np.ones((test_size, 1))))

		# Fit all_att_regressor without transformation
		all_att_regressor.fit(X_train, Y_train)

		# get predicted result from both train data and test data
		y_hat_train = all_att_regressor.predict(X_train)
		y_hat_test = all_att_regressor.predict(X_test)

		# get MSE for each train and test
		mse_train = get_mse(Y_train, y_hat_train)
		mse_test = get_mse(Y_test, y_hat_test)

		# log the error for each run
		train_errors.append(mse_train)
		test_errors.append(mse_test)

	with open('./logs/q4d.txt', 'w') as f:
		f.write(f"Train errors (MSE): {str(train_errors)}\n")
		f.write(f"Average Train error: {np.mean(train_errors)}\n")
		f.write("\n")
		f.write(f"Test errors (MSE): {str(test_errors)}\n")
		f.write(f"Average Test error: {np.mean(test_errors)}\n")
		f.close()

	del all_att_regressor
