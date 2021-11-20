import numpy as np
from numpy.linalg import inv
import csv
import random
import polynomial_regression as regression
import matplotlib.pyplot as plt


# 4a - Naive Regression

def sample_training(dataset, size):
    sample_start = random.randint(0, len(dataset))
    # sample_start = 100
    # print(sample_start)
    training_set = []
    testing_set = []
    if sample_start < len(dataset) - size:
        for i in range(len(dataset)):
            if sample_start <= i < (sample_start + size):
                training_set.append(dataset[i])
            # print("training set" + str(len(testing_set)))
            else:
                # print(i)
                testing_set.append(dataset[i])
    else:
        # this is when the starting point would take u out of bounds, so 506-337 (169)
        for i in range(len(dataset)):
            if i < (size - (len(dataset) - sample_start)) or i >= sample_start:
                training_set.append(dataset[i])
            else:
                testing_set.append(dataset[i])

    # print("training set" + str(len(training_set)))
    # print("testing set" + str(len(testing_set)))
    return training_set, testing_set


def constant_predictor(X, y):
    # print(X)
    # return np.average(X)
    X_transpose = np.transpose(X)
    X_inverse = inv(X_transpose * X)
    return X_inverse


# print(X)
# return np.invert(np.matmul(X_transpose, X))
# return np.matmul(np.matmul(np.invert(np.matmul(X_transpose, X)), X_transpose), y)

if __name__ == '__main__':
    rows = []
    with open("Boston-filtered.txt", 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            rows.append(row)

    training_set_size = round(len(rows) * 2 / 3)
    testing_set_size = len(rows) - training_set_size

    # print(training_set_size)
    # print(testing_set_size)

    # training_ones = np.ones(training_set_size)
    # training_ones_matrix = np.mat(training_ones).transpose()
    # testing_ones = np.ones(testing_set_size)
    # testing_ones_matrix = np.mat(testing_ones).transpose()
    # print(len(training_ones_matrix))

    # for i in range (0, 19):
    training_set, testing_set = sample_training(rows, training_set_size)
    x_training_set = np.array(np.delete(training_set, np.s_[-1:], axis=1)).astype(float)
    y_training_set = np.array(np.delete(training_set, np.s_[:-1], axis=1)).astype(float)

    x_testing_set = np.array(np.delete(testing_set, np.s_[-1:], axis=1)).astype(float)
    y_testing_set = np.array(np.delete(testing_set, np.s_[:-1], axis=1)).astype(float)

    x_training_set = x_training_set.T
    # y_training_set = y_training_set.T

    # model training
    x_training_set = x_training_set.T
    model = regression.PolynomialRegression(degree=12)
    X_transform = model.transform(x_training_set)
    # print(X_transform)

    model.fit(x_training_set, y_training_set)
    # print(model.W)

    # Prediction on training set
    Y_pred = model.predict(x_training_set)
    print(Y_pred[-1][0])

# print(np.shape(x_training_set[0]))
# print(np.shape(np.transpose(y_training_set[0:12][0])))
# plt.scatter(x_training_set[1], np.transpose(y_training_set[0:12]), color='blue')
# plt.plot(x_training_set[0], Y_pred, color='orange')
# plt.title('X vs Y')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()
# print(constant_predictor(x_training_set.T, y_training_set))
# trying to do linear regression against y=a
