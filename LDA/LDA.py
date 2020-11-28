import numpy as np
import pandas as pd

class LDA():
    """
    The Linear Discriminant Analysis classifier
    Also can be treated as a dimension reduction tool
    """
    def __init__(self):
        self.w = None
        self.u0 = None
        self.u1 = None

    def fit(self, X, y):
        # Separate data by class
        X0 = X[y == 0]
        X1 = X[y == 1]

        # calculate means of datasets
        self.u0 = X0.mean(0) # size of (1, p), where p is the num of parameters
        self.u1 = X1.mean(0)

        # Calculate the covariance matrices of the two datasets
        cov0 = np.dot((X0 - self.u0).T, X0 - self.u0) # size of (p, p)
        cov1 = np.dot((X1 - self.u1).T, X1 - self.u1)
        Sw = cov0 + cov1

        # calculate weight matrix by the formula derived with Lagrange multiplier
        self.w = np.dot(np.linalg.inv(Sw), (self.u0 - self.u1).T).reshape(1, -1)  # (1, p)

    def predict(self, X):
        project = np.dot(X, self.w.T)

        wu0 = np.dot(self.w, self.u0.T)
        wu1 = np.dot(self.w, self.u1.T)
        return (np.abs(project - wu1) < np.abs(project - wu0)).astype(float).flatten()

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

if __name__ == "__main__":
    path = "data/watermelon_3a.csv"
    data = pd.read_csv(path, header = None).sample(frac=1).values
    X = standardize(data[:,1:3])
    y = data[:, 3]
    X_train = X[:-3,:]
    y_train = y[:-3]
    X_test = X[-3:,:]
    y_test = y[-3:]
    lda = LDA()
    lda.fit(X_train, y_train)
    print(lda.predict(X_test)) # predicted class
    print(y_test) # real class