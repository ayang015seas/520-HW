import numpy as np
from scipy.stats import multivariate_normal


class GaussianMixtureModel():
    def __init__(self, K, init_mean, covariances, mixing_coeff=None):
        """ Gaussian Mixture Model

        NOTE: You should NOT modify this constructor or the signature of any class functions

        Args:
            k (int): Number of mixture components

            init_mean (numpy.array):
                The initial mean parameter for the mixture components.
                It should have shape (k, d) where d is the dimension of the
                data
            
            covariances (numpy.array):
                The initial covariance parameter of the mixture components
                It should have shape (k, d, d)

            mixing_coef (numpy.array):
                The initial mixing coefficient. Default is None
                If provided, it should have shape (k,)

        """
        # Some housekeeping things to make sure the dimensions agree
        (num_k, d) = init_mean.shape
        assert(num_k == K)
        assert(init_mean.shape == (K, d))
        assert(covariances.shape == (K, d, d))
#         assert(mixing_coeff.shape == (K,))
        
        self.K = K
        # If mixing coefficient is not specified, initialize to uniform distribution
        if mixing_coeff is None:
            mixing_coeff = np.ones((K,)) * 1./K
        self.mixing_coeff = mixing_coeff
        self.mus = np.copy(init_mean)
        self.covariances = np.copy(covariances)
        self.d = d
    
    def fit(self, X, num_iters=1000, eps=1e-6):
        """ Learn the GMM via Expectation-Maximization
        
        NOTE: you should NOT modify this function

        Args:
            X (numpy.array): Input data matrix

            num_iters (int): Number of EM iterations

        Returns:
            None

        """
        (N, d) = X.shape
        assert(d == self.d)

        self.llh = []
                
        for it in range(num_iters):
            p = self.E_step(X)
            self.M_step(X, p)

            log_ll = self.compute_llh(X)
            # Early stopping
            self.llh.append(log_ll)
            if it > 0 and np.abs(log_ll - self.llh[-2]) < eps:
                print("Hit Log Constraint")
                break
                    
    def E_step(self, X):
        """ Do the estimation step of the EM algorithm
        
        Arg:
            X (numpy.array):
                The input feature matrix of shape (N, d)

        Returns 
            p (numpy.array):
                The "membership" matrix of shape (N, k)
                Where p[i, j] is the "membership proportion" of instance
                `i` in mixture component `j` 
                with `0 <= i <= N-1`
                     `0 <= j <= k-1`

        """
        # Your code goes here
#         print(X.shape)
#         print(self.mus.shape)
        scores = []
        
        for i in range(0, self.K):
            chance = multivariate_normal.pdf(X, mean=self.mus[i], cov=self.covariances[i])
            chance = chance * self.mixing_coeff[i]
            scores.append(chance)
        scores = np.asarray(scores).T
#         print("E scores", scores.shape)
        for i in range(0, scores.shape[0]):
            total = np.sum(scores[i])
            scores[i] = scores[i] / total
#         print(scores)
        return scores
    
    def covar(self, X, mean, r_weights, mc):
        cov = np.zeros((2, 2))
        for i in range(0, X.shape[0]):
            outer = np.outer(X[i] - mean, X[i] - mean) * r_weights[i]
            cov = cov + outer
#         print("COV", cov.shape)
        return cov / mc
        
    def M_step(self, X, p):
        """  Do the maximization step of the EM algorithm
        Update the mean and covariance matrix of the component mixtures

        Arg:
            X (numpy.array):
                The input feature matrix of shape (N, d)

            p (numpy.array):
                The "membership" matrix of shape (N, k)
                Where p[i, j] is the "membership proportion" of instance
                `i` in mixture component `j` 
                with `0 <= i <= N-1`
                     `0 <= j <= k-1`

        Returns:
            None

        """
        # Your code goes here
        total_sum = X.shape[0]
        p_transpose = p.T 
        coeff_array = []
        mean_array = []
        cov_array = []
        
        for i in range(0, self.K):
            mc = np.sum(p_transpose[i])
            coeff_array.append(mc / total_sum)
            r_matrix = np.tile(p_transpose[i] ,(X.shape[1],1))
#             print((r_matrix.T * X).shape)
            
            mean = np.sum(r_matrix.T * X, axis=0) / mc
#             print("mean", mean)
            mean_array.append(mean)
            meantile = np.tile(mean, (X.shape[0],1))
            cov = ((r_matrix * (X - meantile).T) @ (X - meantile)) / mc
#             cov = self.covar(X, mean, p_transpose[i], mc)
            cov_array.append(cov)

        coeff_array = np.asarray(coeff_array)
        mean_array = np.asarray(mean_array)
        cov_array = np.asarray(cov_array)
        self.mixing_coeff = coeff_array
        self.mus = mean_array
        self.covariances = cov_array
        
#         print(self.mixing_coeff.shape, coeff_array.shape)
#         print(self.mus.shape, mean_array.shape)
#         print(self.covariances.shape, cov_array.shape)
        return 
                
    def compute_llh(self, X):
        """ Compute the log likelihood under the GMM
        Arg:
            X (numpy.array): Input feature matrix of shape (N, d)

        Returns:
            llh (float): Log likelihood of the given data 
                under the learned GMM

        """
        # Your code goes here
        llh = 0
        for i in range(0, X.shape[0]):
            aggregate = 0
            for j in range(0, self.K):
                chance = multivariate_normal.pdf(X[i], mean=self.mus[j], cov=self.covariances[j])
                chance = chance * self.mixing_coeff[j]
                aggregate = aggregate + chance
#             print(aggregate)
            llh = llh + np.log(aggregate)
        return llh / len(X)

