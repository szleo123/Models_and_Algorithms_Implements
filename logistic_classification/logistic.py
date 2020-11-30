import numpy as np
import pandas as pd
import math

def sigmoid(x):
    return 1 /(1 + np.exp(-x))

def standardize(X):
    """ Standardize the dataset X """
    X_std = X
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    for col in range(np.shape(X)[1]):
        if std[col]:
            X_std[:, col] = (X_std[:, col] - mean[col]) / std[col]
    # X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    return X_std

class LogReg():
    '''
    Logistic Regression given a evidence matrix and
    corresponding labels

    - learning_rate is for gradient descent
    '''
    def __init__(self, learning_rate = 0.05):
        self.beta = None
        self.learning_rate = learning_rate

    def train(self, X, y, iterations):
        X = np.hstack((np.ones((X.shape[0], 1)), X)) # prepend a col of 1 to the data matrix

        # initialize the parameters
        num_para = X.shape[1]
        limit = 1 / math.sqrt(num_para)
        self.beta = np.random.uniform(-limit, limit, (num_para,))

        # apply gradient descent of given number of iterations
        for i in range(iterations):
            y_pred = sigmoid(X.dot(self.beta)) # probabilities of positive case
            grad = - (y - y_pred).dot(X)
            self.beta = self.beta - self.learning_rate * grad
            if i % 10 == 0:
                cost = np.sum(-y*X.dot(self.beta) + np.log(1 + np.exp(X.dot(self.beta))))
                print('{}th iteration, cost is {}'.format(i, cost))

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))  # prepend a col of 1 to the data matrix
        y_pred = np.round(sigmoid(X.dot(self.beta))).astype(float)
        return y_pred

if __name__ == "__main__":
    path = "data/watermelon_3a.csv"
    data = pd.read_csv(path, header = None).sample(frac=1).values
    X = standardize(data[:,1:3])
    y = data[:, 3]
    X_train = X[:-3,:]
    y_train = y[:-3]
    X_test = X[-3:,:]
    y_test = y[-3:]
    logreg = LogReg()
    logreg.train(X_train, y_train, 100)
    print(logreg.predict(X_test)) # predicted class
    print(y_test) # real class

