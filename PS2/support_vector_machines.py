# support_vector_machines.py starts here

import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers


class SVM():
    def __init__(self, kernel, C=1):
        """
        Constructor for SVM classifier
        NOTE: do NOT modify this function
        
        Args:
            kernel (function):
                The kernel function, it needs to take
                TWO vector input and returns a float

            C (float): Regularization parameter

        Returns:
            An initialized SVM object

        Example usage:
            >>> kernel = lambda x, y: numpy.sum(x * y)
            >>> C = 0.1
            >>> model = SVM(kernel, C) 

        ---
        """
        self.kernel = kernel 
        self.C = C
        self.alpha = None
        self.w = None
        self.b = None
        self.svec = None
        self.y = None
        self.x = None
    
    def fit(self, X, y):
        """
        Learns the support vectors by solving the dual optimization
        problem of kernel SVM.

        Args:
            X (numpy.array[float]):
                Input feature matrix with shape (N, d)
            y (numpy.array[float]):
                Binary label vector {-1, +1} with shape (N,)

        Returns:
            None

        """

        # Your code goes here
        N, d = X.shape
        H = np.zeros((N, N))
        
        # default kernel ops
        if (self.kernel == None):
            self.kernel = lambda x1, x2: np.dot(x1, x2)
            
        # perform kernel operations
        for i in range(0, N):
            for j in range(0, N):
                H[i,j] = self.kernel(X[i], X[j]) * y[i] * y[j]
        
        P = cvxopt_matrix(H)
        # setup vector of -1
        qvec = -np.ones((N, 1))
        q = cvxopt_matrix(qvec)
        # setup stacked identity matrices
        gmat = np.vstack((np.eye(N)*-1, np.eye(N)))
        G = cvxopt_matrix(gmat)
        # 0s followed by c 
        hvec = np.hstack((np.zeros(N), np.ones(N) * self.C))
        h = cvxopt_matrix(hvec)
        # avec is y vec
        A = cvxopt_matrix(y.reshape(1, -1))
        # b is just zeroes
        b = cvxopt_matrix(np.zeros(1))
        
        cvxopt_solvers.options['show_progress'] = False
        
        opt = cvxopt_solvers.qp(P, q, G, h, A, b)
        alphas = np.array(opt['x'])
        
        support_vecs = np.greater(alphas, self.C*(10**-6))
        upper_bound = np.greater(self.C*(1 - (10**-6)), alphas)
        ltc = np.logical_and(support_vecs, upper_bound)
        ltc = ltc.flatten()
        support_vecs = support_vecs.flatten()
        
        alphasmod = np.zeros(len(ltc))
        sv1num = 0
        
        # threshold all alpha values
        for i in range(0, len(ltc)):
            if (alphas[i] > self.C*(1 - (10**-6))):
                alphasmod[i] = self.C*(1 - (10**-6))
                ltc[i] = False
            elif (alphas[i] < self.C*(10**-6)):
                alphasmod[i] = self.C*(10**-6)
                ltc[i] = False
            else:
                alphasmod[i] = alphas[i]
                sv1num += 1
                ltc[i] = True
        
        # set up bias calculation 
        w = ((y * alphasmod).transpose() @ X)
        x_sv = X[support_vecs]
        y_sv = y[support_vecs]
        x_sv1 = X[ltc]
        y_sv1 = y[ltc]
        sv_alphas = alphas[support_vecs]
        
        self.w = w
        self.alpha = alphasmod
        self.svec = ltc
        self.y = y
        self.x = X
        
        ymod = y[ltc]
        xmod = X[ltc]
        ywx = np.zeros(N)
        
        # compute bias using kernel 
        bias = 0
        svnum = np.count_nonzero(support_vecs)
        for i in range(sv1num):
            w = 0
            for j in range(svnum):
                w += sv_alphas[j] * y_sv[j] * self.kernel(x_sv1[i], x_sv[j])
            bias += (y_sv1[i] - w)
        bias = bias / sv1num
        self.b = bias

        return w, bias
    
    def predict(self, X):
        """
        Predict the label {-1, +1} of input data points using the learned
        support vectors

        Args:
            X (numpy.array):
                The data feature matrix of shape (N,d)
        Returns:
            y_hat (numpy.array):
                The {-1, +1} label vector of shape (N,)
        
        """
        xtx = np.zeros(len(X))
        
        for i in range(0, len(X)):
            total = 0            
            for j in range(0, len(self.x)):
                total += self.kernel(self.x[j], X[i]) * self.alpha[j] * self.y[j]
            total += self.b
            xtx[i] = total
        
        ans = np.sign(xtx)
        return ans

        