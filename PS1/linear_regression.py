import numpy as np
import utils
from numpy.linalg import inv

class LeastSquareRegression():
    """ Least square regression
    """
    def __init__(self, lam=0):
        """ Construct a least square regression object
        Don't modify this
        
        Args: 
            lam (float):
                regularization constant
                Default = 0 (ordinary least square regression)
                If lam > 0 (regularized least square regression)
        """
        self.lam = lam
        self.w = None
        self.b = 0
    
    def fit(self, X, y):
        """ Learn the weights of the linear regression model
        by solving for w in closed form

        Args:
            X (numpy.array)
                Input feature data with shape (N, d)
                
            y (numpy.array)
                Label vector with shape (N,)

        Returns:
            (w, b)
                w (numpy.array):
                    Learned weight vector with shape (d,)
                    where d is the number of features of data X
                b (float)
                    Learned bias term
        
        """
        N, d = X.shape
        X = utils.augment_bias(X)
        
        assert(y.shape == (N,))
        xt = X.transpose()
        iden = np.identity(d + 1)
        iden[d][d] = 0
        prod1 = inv((xt @ X) + (self.lam * N * iden))
        prod2 = xt @ y
        
        wvec = prod1 @ prod2
        # Your solution goes here
        wlen = len(wvec) - 1
        self.w = wvec[0: wlen]
        self.b = wvec[wlen]
        return wvec
    
    def predict(self, X):
        """ Do prediction on given input feature matrix

        Args:
            X (numpy.array)
                Input feature matrix with shape (N, d)
        Returns:
            Predicted output vector with shape (N,)
        
        """
        # Your solution goes here

        res = [] #  empty regular list
        for i in range(0, len(X)):
            res.append(self.w.dot(X[i]) + self.b)
        return np.array(res)
    

