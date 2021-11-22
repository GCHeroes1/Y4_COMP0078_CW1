import numpy as np
import q4
from numpy.linalg import inv
import csv
import random
import math
import matplotlib.pyplot as plt
from polynomial_regression import PolynomialRegression


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


def gaussian_KernelK_Construction(x, sigma):
    size = len(x)
    # print(size)
    # kernel = []
    kernel = np.zeros(shape=(size, size))
    for i in range(size):
        for j in range(size):
            kernel[i][j] = gaussian_kernel_calc(x[i], x[j], sigma)
    return kernel


# def equation_10(n, data, weight, predictions, gamma):
#     # w* = argmin _ (for all w in real) * 1/l *
#     sum = 0
#     for i in range(n):
#         math = data.T[i] * weight - predictions[i]
#         sum += math ** 2
#     part_2 = gamma * np.matmul(weight.T, weight)
#     return np.argmin(1 / n * sum + part_2)


# return np.argmin(1/n * np.sum(np.matmul(data.T, weight) - predictions)**2 + lamb * np.matmul(weight.T, weight))

# def equation_11(n, gamma, alpha, x, y):  # alpha star dual optimisation
#     sum_1 = np.argmin(1 / n * nested_summation_alphastar(n, alpha, x, y))
#     sum_2 = gamma * np.matmul(np.matmul(alpha.T, kernelKConstruction(x)), alpha)
#     return sum_1 + sum_2


def calc_alpha_star(kernel, gamma, n, y):  # alpha star, solved dual
    # kernel = gaussianKernelKConstruction(x, sigma)
    identity = np.identity(n)
    part_1 = inv(kernel + gamma * n * identity)
    return part_1.dot(y)


def calc_y_test(alpha_star, training_data, testing_data, n, sigma):  # generating y_test
    y_test = 0
    for i in range(n):
        y_test += alpha_star[i] * gaussian_kernel_calc(training_data[i], testing_data, sigma)
        # y_test += math
    return y_test


def gaussian_kernel_calc(x_i, x_j, sigma):  # eq 14, the gaussian kernel
    # print(type(x))
    # print(np.size(x))
    # print(x[i])
    # print("done")
    x_ = x_i - x_j
    normal = 0
    # numerator = numpy.abs(x_) ** 2
    for i in x_:
        normal += i ** 2
    denominator = 2 * (sigma ** 2)
    gaussian = np.exp((-1 * normal) / denominator)
    # print("gaussian is ")
    # print(gaussian)
    return gaussian


def five_fold_dataset(data):
    random.shuffle(data)
    split = math.floor(len(data) / 5)
    split_data = [data[0: split], data[split:2 * split], data[2 * split:3 * split], data[3 * split:4 * split],
                  data[4 * split:5 * split]]
    return split_data


def five_fold_validation(data, gamma_values, sigma_values):
    five_fold_data = five_fold_dataset(data)
    predictor = [-1, -1, -1]
    plotting_data = []    # sigma, gamma, MSE to be plotted
    # five_fold_data = [one, two, three, four, five]
    # need to make predictions with each gamma and sigma value and calculate MSE for each permutation
    for gamma in gamma_values:
        for sigma in sigma_values:
            average_MSE = 0
            for i in range(5):  # 5 fold validation
                training_data = []
                for j in range(5):
                    if j != i:
                        for data_ in five_fold_data[j]:
                            training_data.append(data_)
        # now we have the training data and testing data, but need to also split them to get the y value (13th value)
        #         training_data_y_array = []
        #         training_data_x_array = []
        #         for z in training_data:
        #             training_data_y_array.append(z[:1])
        #             training_data_x_array.append(z[:-1])
        #
        #         training_data_x = np.array(training_data_x_array).astype(float)
        #         training_data_y = np.array(training_data_y_array).astype(float)
                training_data_x, training_data_y = split_data(training_data)

                n = len(training_data_x)
                gaussian_kernel = gaussian_KernelK_Construction(training_data_x, sigma)
                alpha_star = calc_alpha_star(gaussian_kernel, gamma, n, training_data_y)
                # print(np.shape(alpha_star))

                testing_data = five_fold_data[i]
                # testing_data_y_array = []
                # testing_data_x_array = []
                # for z in testing_data:
                #     testing_data_y_array.append(z[:1])
                #     testing_data_x_array.append(z[:-1])
                #
                # testing_data_x = np.array(testing_data_x_array).astype(float)
                # testing_data_y = np.array(testing_data_y_array).astype(float)
                testing_data_x, testing_data_y = split_data(testing_data)

                # MSE = SSE / m
                m = len(testing_data_x)
                # print(len(testing_data_x))
                SSE = 0
                for i in range(m):
                    y_test = calc_y_test(alpha_star, training_data_x, testing_data_x[i], n, sigma)
                    # print(y_test)
                    SSE += (testing_data_y[i] - y_test) ** 2
                MSE = SSE / m
                # print(MSE)
                average_MSE += MSE
            average_MSE = average_MSE / 5
            # print("calculated average MSE of", average_MSE, "with sigma", sigma, "and gamma", gamma)
            plotting_data.append([sigma, gamma, average_MSE[0]])
            # print(plotting_data)
            if predictor[2] > average_MSE or predictor[2] == -1:
                predictor[0], predictor[1], predictor[2] = sigma, gamma, average_MSE[0]
    # print(predictor)
    # print(plotting_data)
    return plotting_data, predictor


def training_MSE(x_data, y_data, var, gamma):
    n = len(x_data)
    gaussian_kernel = gaussian_KernelK_Construction(x_data, var)
    alpha_star = calc_alpha_star(gaussian_kernel, gamma, n, y_data)
    SSE = 0
    for i in range(n):
        y_test = calc_y_test(alpha_star, x_data, x_data[i], n, var)
        SSE += (y_data[i] - y_test) ** 2
    MSE = SSE / n
    return MSE


def testing_MSE(x_training, y_training, var, gamma, x_testing, y_testing):
    n = len(x_training)
    m = len(x_testing)
    gaussian_kernel = gaussian_KernelK_Construction(x_training, var)
    alpha_star = calc_alpha_star(gaussian_kernel, gamma, n, y_training)
    SSE = 0
    for i in range(m):
        y_test = calc_y_test(alpha_star, x_training, x_testing[i], n,  var)
        SSE += (y_testing[i] - y_test) ** 2
    MSE = SSE / m
    return MSE


def split_data(data):
    data_y_array = []
    data_x_array = []
    for x in data:
        data_y_array.append(x[:1])
        data_x_array.append(x[:-1])

    data_x = np.array(data_x_array).astype(float)
    data_y = np.array(data_y_array).astype(float)
    return data_x, data_y


def naive_regression(training_data, testing_data, train_size, test_size):
    naive_regression_model = PolynomialRegression(need_transform=False)

    # Split vertically for X and Y
    X_train = np.ones((train_size, 1))
    Y_train = training_data[:, -1]
    X_test = np.ones((test_size, 1))
    Y_test = testing_data[:, -1]

    # Fit Naive Regression model without transformation
    naive_regression_model.fit(X_train, Y_train)

    # get predicted result from both train data and test data
    y_hat_train = naive_regression_model.predict(X_train)  # (333, 1) -> (333,)
    y_hat_test = naive_regression_model.predict(X_test)  # (173, 1) -> (173,)

    # get MSE for each train and test
    mse_train = q4.get_mse(Y_train, y_hat_train)
    mse_test = q4.get_mse(Y_test, y_hat_test)

    del naive_regression_model
    return mse_train, mse_test  # these are the MSEs that I need

def single_attribute_regression(training_data, testing_data, train_size, test_size, attribute):    # attribute must go from 0 to 11
    single_attribute_regression_model = PolynomialRegression(need_transform=False)

    # Split vertically for X and Y
    X_train = training_data[:, attribute, np.newaxis]
    X_test = testing_data[:, attribute, np.newaxis]
    Y_train = training_data[:, -1]
    Y_test = testing_data[:, -1]

    # augment with an additional bias term
    X_train = np.hstack((X_train, np.ones((train_size, 1))))
    X_test = np.hstack((X_test, np.ones((test_size, 1))))


    # Fit single attribute Regression model
    single_attribute_regression_model.fit(X_train, Y_train)

    # get predicted result from both train data and test data
    y_hat_train = single_attribute_regression_model.predict(X_train)
    y_hat_test = single_attribute_regression_model.predict(X_test)

    # get MSE for each train and test
    mse_train = q4.get_mse(Y_train, y_hat_train)
    mse_test = q4.get_mse(Y_test, y_hat_test)
    del single_attribute_regression_model

    return mse_train, mse_test

def all_attribute_regression(training_data, testing_data, train_size, test_size):
    all_att_regressor = PolynomialRegression(need_transform=False)

    # Split vertically for X and Y
    X_train = training_data[:, :-1]
    Y_train = training_data[:, -1]
    X_test = testing_data[:, :-1]
    Y_test = testing_data[:, -1]

    # augment with an additional bias term
    X_train = np.hstack((X_train, np.ones((train_size, 1))))
    X_test = np.hstack((X_test, np.ones((test_size, 1))))

    # Fit all_att_regressor without transformation
    all_att_regressor.fit(X_train, Y_train)

    # get predicted result from both train data and test data
    y_hat_train = all_att_regressor.predict(X_train)
    y_hat_test = all_att_regressor.predict(X_test)

    # get MSE for each train and test
    mse_train = q4.get_mse(Y_train, y_hat_train)
    mse_test = q4.get_mse(Y_test, y_hat_test)

    del all_att_regressor
    return mse_train, mse_test


if __name__ == '__main__':
    # data = np.arange(9, 18).reshape((3, 3))
    # weight = np.arange(3).reshape((3, 1))
    # predictions = np.arange(6, 9).reshape((3, 1))
    # print(data)
    # print(data[1][1])
    #
    # gamma_values = [2 ** (-40), 2 ** (-39), 2 ** (-38), 2 ** (-37), 2 ** (-36), 2 ** (-35), 2 ** (-34), 2 ** (-33),
    #                 2 ** (-32), 2 ** (-31), 2 ** (-30), 2 ** (-29), 2 ** (-28), 2 ** (-27), 2 ** (-26)]
    # sigma_values = [2 ** 7, 2 ** 7.5, 2 ** 8, 2 ** 8.5, 2 ** 9, 2 ** 9.5, 2 ** 10, 2 ** 10.5, 2 ** 11, 2 ** 11.5,
    #                 2 ** 12, 2 ** 12.5, 2 ** 13]
    #
    # rows = []
    # with open("Boston-filtered.txt", 'r') as file:
    #     csvreader = csv.reader(file)
    #     header = next(csvreader)
    #     for row in csvreader:
    #         rows.append(row)
    #
    # # print(np.array(rows))
    # plotting, predictor_ = five_fold_validation(rows[:12], gamma_values, sigma_values)
    # sigma_points, gamma_points, MSE_points = [], [], []
    # for plots in plotting:
    #     sigma_points.append(plots[0])
    #     gamma_points.append(plots[1])
    #     MSE_points.append(plots[2])
    # # sigma_points, gamma_points, MSE_points = plotting[0], plotting[1], plotting[2]
    # # print(sigma_points)
    # # print(gamma_points)
    # # print(MSE_points)
    # # print(predictor_)
    #
    # ################
    # # Question 5b  #
    # ################
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.set_xlabel('Variance')
    # ax.set_ylabel('Gamma')
    # ax.set_zlabel('MSE')
    # plt.title("Q5b: Cross-validation Error")
    # ax.scatter3D(sigma_points, gamma_points, MSE_points, label='Cross-validation error')
    # plt.legend()
    # plt.savefig('./plots/q5b.png')
    # # plt.show()
    #
    # ################
    # # Question 5c  #
    # ################
    # data = q4.get_raw_data()
    # train_size = int(data.shape[0] * 0.66)
    # test_size = data.shape[0] - train_size
    # # print(f"{train_size} Train sets and {test_size} Test sets")
    # train, test = q4.sample_training(data, train_size)
    # training_data_x, training_data_y = split_data(train)
    # testing_data_x, testing_data_y = split_data(test)
    #
    # trainingMSE = training_MSE(training_data_x, training_data_y, predictor_[0], predictor_[1])
    # testingMSE = testing_MSE(training_data_x, training_data_y, predictor_[0], predictor_[1], testing_data_x, testing_data_y)
    # # print(trainingMSE)
    # # print(testingMSE)
    # # print("MSE on training set with predictor:", predictor_, "is", trainingMSE)
    # # print("MSE on testing set with predictor:", predictor_, "is,", testingMSE)
    #
    # with open('./logs/q5c.txt', 'w') as f:
    #     f.write(f"MSE on training set with predictor: {str(predictor_)} is {str(trainingMSE)}\n")
    #     f.write("\n")
    #     f.write(f"MSE on testing set with predictor: {str(predictor_)} is {str(testingMSE)}\n")
    #     f.close()

    ################
    # Question 5d  #
    ################

    gamma_values = [2 ** (-40), 2 ** (-39), 2 ** (-38), 2 ** (-37), 2 ** (-36), 2 ** (-35), 2 ** (-34), 2 ** (-33),
                    2 ** (-32), 2 ** (-31), 2 ** (-30), 2 ** (-29), 2 ** (-28), 2 ** (-27), 2 ** (-26)]
    sigma_values = [2 ** 7, 2 ** 7.5, 2 ** 8, 2 ** 8.5, 2 ** 9, 2 ** 9.5, 2 ** 10, 2 ** 10.5, 2 ** 11, 2 ** 11.5,
                    2 ** 12, 2 ** 12.5, 2 ** 13]
    training_results = np.zeros((15, 20)) # store the results for each trial in here, for each method
    testing_results = np.zeros((15, 20))
    for trial in range(20):
        # print(trials)
        data = q4.get_raw_data()
        train_size = int(data.shape[0] * 0.66)
        test_size = data.shape[0] - train_size
        train, test = q4.sample_training(data, train_size)

        # naive regression
        training_error, testing_error = naive_regression(train, test, train_size, test_size)
        training_results[0][trial] = training_error
        testing_results[0][trial] = testing_error

        # each single attribute
        # print(training_error)
        for i in range(1, 13):
            training_error, testing_error = single_attribute_regression(train, test, train_size, test_size, i - 1)
            training_results[i][trial] = training_error
            testing_results[i][trial] = testing_error
            # print(i)
        # every attribute
        training_error, testing_error = all_attribute_regression(train, test, train_size, test_size)
        training_results[13][trial] = training_error
        testing_results[13][trial] = testing_error

        #kernel ridge regression

        plotting, predictor_2 = five_fold_validation(data, gamma_values, sigma_values) # only care to get the predictor, best gamma and sigma values
        training_data_x, training_data_y = split_data(train)
        testing_data_x, testing_data_y = split_data(test)

        trainingMSE = training_MSE(training_data_x, training_data_y, predictor_2[0], predictor_2[1])
        testingMSE = testing_MSE(training_data_x, training_data_y, predictor_2[0], predictor_2[1], testing_data_x,
                                 testing_data_y)
        training_results[14][trial] = trainingMSE
        testing_results[14][trial] = testingMSE
        print("done loop #", trial)
    print(testing_results)

    # work out average MSE across trials

    for i in range(len(training_results)):
        print(i)
        training_MSE_final = np.sum(training_results[i]) / 20
        training_SD_final = np.std(training_results[i])
        testing_MSE_final = np.sum(testing_results[i]) / 20
        testing_SD_final = np.std(testing_results[i])
        with open('./logs/q5d.txt', 'w') as f:
            f.write(f"Method {str(i)} with MSE train {str(training_MSE_final)} with SD {str(training_SD_final)} and MSE test {str(testing_MSE_final)} with SD {str(testing_SD_final)}\n")
            f.write("\n")
            f.close()
