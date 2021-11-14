import numpy as np
import matplotlib.pyplot as plt


class PolynomialRegression:

    def __init__(self, degree, learning_rate=0.01, iterations=10000):
        self.degree = degree
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.X = None
        self.Y = None
        self.m = None
        self.n = None
        self.weights = None

    # function to transform X
    def transform(self, X):
        self.m, self.n = X.shape
        # initialize X_transform
        X_transform = np.ones((self.m, 1))

        j = 0
        for j in range(self.degree + 1):
            if j != 0:
                x_pow = np.power(X, j)
                # append x_pow to X_transform 2-D array
                X_transform = np.append(X_transform, x_pow.reshape(-1, 1), axis=1)

        return X_transform

    # model training
    def fit(self, X, Y, method='ne'):
        self.X = X
        self.Y = Y

        # transform X for polynomial h( x ) = w0 * x^0 + w1 * x^1 + w2 * x^2 + ........+ wn * x^n
        X_transform = self.transform(self.X)

        if method == 'gd':
            # gradient descent with MSE cost function
            self.weights = np.zeros(self.degree + 1)
            for i in range(self.iterations):
                y_hat = self.predict(self.X)
                error = y_hat - self.Y

                # update weights by derivative of MSE
                self.weights = self.weights - self.learning_rate * (2 / self.m) * np.dot(X_transform.T, error)

        elif method == 'ne':
            self.weights = np.linalg.inv(X_transform.T.dot(X_transform)).dot(X_transform.T).dot(self.Y)

        else:
            raise NotImplementedError

        return self

    def predict(self, X):
        # transform X for polynomial h( x ) = w0 * x^0 + w1 * x^1 + w2 * x^2 + ........+ wn * x^n
        X_transform = self.transform(X)
        return np.dot(X_transform, self.weights)


if __name__ == "__main__":
    X = np.array([[1, 2, 3, 4]])
    X = X.T
    Y = np.array([3, 2, 0, 5])

    # model training
    model = PolynomialRegression(degree=3)

    model.fit(X, Y)
    print(model.weights)

    # Prediction on training set
    Y_pred = model.predict(X)

    # Visualization
    plt.scatter(X, Y, color='blue')
    plt.plot(X, Y_pred, color='orange')
    plt.title('X vs Y')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
