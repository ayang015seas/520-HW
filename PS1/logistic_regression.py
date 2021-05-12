import numpy as np
import utils
from scipy.optimize import minimize
from scipy.special import logsumexp
import math
import matplotlib.pyplot as plt

class LogisticRegression():
    """ Logistic regression
    """
    def __init__(self, lam):
        """ Construct a least square regression object
        Don't modify this

        Args: 
            lam (float):
                regularization constant
                Default = 0 (ordinary logistic regression)
                If lam > 0 (regularized logistic regression)
        """
        self.lam = lam
        self.w = None
    
    def logistic_loss(self, w, X, y):
        """ Compute the logistic loss

        Args: 
            w (numpy.array)
                Weight vector
            
            X (numpy.array)
                Input feature matrix

            y (numpy.array)
                Label vector

        Returns:
            Logistic loss
        
        """
        # Your solution goes here
        yparam = (X @ w) * y * -1
        l = len(w)
        sub = w[0:l - 1:1]
        wsum = sub.dot(sub)
        return np.sum(np.log(1 + (np.exp(yparam)))) / len(y) + (self.lam * wsum)
    
        
    def fit(self, X, y, w0=None):
        """ Learn the weight vector of the logistic regressor via 
        iterative optimization using scipy.optimize.minimize

        Args:
            X (numpy.array)
                Input feature matrix with shape (N,d)
            y (numpy.array)
                Label vector with shape (N,)
            w0 (numpy.array) 
                (Optional) initial estimate of weight vector

        Returns:
            (w, b)
                w (numpy.array):
                    Learned weight vector with shape (d,)
                    where d is the number of features of data X
                b (float)
                    Learned bias term

        """
        (N, d) = X.shape
        assert(y.shape == (N,))
        
        # You should initialize w0 here
        if not w0:
            # Do something instead of return
            w0 = np.zeros(shape=(d + 1),)

        X = utils.augment_bias(X)
        # Hint:
        res = minimize(self.logistic_loss, w0, args=(X, y))
        w  = res["x"]
        self.w = w
        # bias is included in the w, so no need to return separately
        return w

        
    def predict(self, X):
        """ Predict the label {+1, -1} of data points
        
        Args:
            X (numpy.array)
                Input feature matrix with shape (N, d)


        Returns:
            y (numpy.array)
                Predicted label vector with shape (N,)
                Each entry is in {+1, -1}

        """
        res = [] #  empty regular list
        X = utils.augment_bias(X)
        for i in range(0, len(X)):
            if (self.w.dot(X[i]) > 0):
                res.append(1)
            else:
                res.append(-1)
        return np.array(res)
        

        