#neural_network.py starts here

import numpy as np

"""Some helper functions"""

def sigmoid(g):
    return 1./(1. + np.exp(-g))

def sigmoid_derivative(g):
    return sigmoid(g) * (1 - sigmoid(g))

def relu(g):
    return np.maximum(0, g)

def relu_derivative(g):
    derivatives = np.heaviside(g, 0)
    return derivatives


class NeuralNetworkBase():
    def __init__(self, input_dim, num_hidden=10, activation="sigmoid", W1=None, b1=None, W2=None, b2=None):
        """ Base Neural Network class
        You are to implement NeuralNetworkClassification which
        inherit from NeuralNetworkBase.

        NOTE: do NOT modify this class

        Args:
            input_dim (int): Number of input features

            num_hidden (int): Hidden dimension (number of hidden nodes)
                Default value is 10

            activation (str): Activation function name. Default is "sigmoid"

            W1 (numpy.array): 
                Initial value for the weight matrix of the 1st layer
                W1 should have shape (num_hidden, input_dim)
                Default is None

            b1 (numpy.array): 
                Initial value for the bias of the 1st layer
                b1 should have shape (num_hidden, )
                Default is None

            W2 (numpy.array): 
                Initial value for the weight matrix of the 2nd layer
                W2 should have shape (1, num_hidden)
                Default is None

            b2 (numpy.float): 
                Initial value for the bias of the 1st layer
                b2 should be a scalar
                Default is None

        Example usage:
            >>> init_param = utils.load_initial_weights("P3/Synthetic-Dataset/InitParams/relu/5")
            >>> nnet__classification = NeuralNetworkClassification(input_dim=10, num_hidden=20)
            >>> nnet__classification.fit(X_train, y_train, step_size=0.01)

        """
        self.d  = input_dim # Remember the input dimension and hidden dimension
        self.d1 = num_hidden
        
        assert(activation in ["sigmoid", "relu"])
        self.g = sigmoid if activation == "sigmoid" else relu
        self.g_prime = sigmoid_derivative if activation == "sigmoid" else relu_derivative
        
        # If the parameters are not specified, initialize them randomly
        self.layers = {
            "W1": W1 if W1 is not None else np.random.rand(self.d1, self.d),
            "b1": b1 if b1 is not None else np.random.rand(self.d1),
            "W2": W2 if W2 is not None else np.random.rand(1, self.d1),
            "b2": b2 if b2 is not None else np.random.rand(), 
        }
        
        # NOTE the shape of the vectors
        assert(self.layers["b1"].shape == (self.d1,))
        assert(self.layers["W1"].shape == (self.d1, self.d))
        assert(self.layers["W2"].shape == (1, self.d1))
    
    def fit(self, X, y, num_iters=8000, step_size=0.1):
        """ Train the neural network on given data via gradient descent

        Args:
            X (numpy.array):
                Input data matrix. X should have shape (num_instances, input_dim)
            
            y (numpy.array):
                Input label vector. y should have shape (num_instances,)

        Returns:
            None

        """
        for it in range(num_iters):
            derivatives = self.back_propagate(X, y)
            max_deriv_norm = 0.
            if (it % 1000 == 0):
                print(it, " Iterations")
            for parameter, value in derivatives.items():
                self.layers[parameter] = self.layers[parameter] - step_size * value
                max_deriv_norm = max(max_deriv_norm, np.sum(value ** 2))

    def back_propagate(self, X, y):
        """ Compute the derivative of the loss function with respect to
        the parameters of the neural networks

        Args:
            X (numpy.array):
                Input data matrix. X should have shape (num_instances, input_dim)
            
            y (numpy.array):
                Input label vector. y should have shape (num_instances,)

        Returns:
            derivative (dictionary):
                A dictionary that contains the derivative of the loss function
                with respect to the parameters of the neural networks.
                For example,
                derivative["W1"] should contain a numpy array that is the derivative
                of the loss function with respect to W1. It should have the same shape
                as W1

        """
        raise NotImplementedError

    def predict(self, X):
        """ Do prediction on given input data
        Arg:
            X (numpy.array):
                Input data matrix. X should have shape (num_instances, input_dim)
        
        Returns:
            y (numpy.array):
                Label vector. y should have shape (num_instances,)

        """
        raise NotImplementedError


class NeuralNetworkClassification(NeuralNetworkBase):
    def __init__(self, input_dim, num_hidden=10, activation="sigmoid", W1=None, b1=None, W2=None, b2=None):
        """ Neural network for classification.
        This simply calls the constructor for NeuralNetworkBase

        NOTE: do NOT modify this constructor
        NOTE: do NOT modify the signature of any class functions

        Args:
            input_dim (int): Number of input features

            num_hidden (int): Hidden dimension (number of hidden nodes)
                Default value is 10

            activation (str): Activation function name. Default is "sigmoid"

            W1 (numpy.array): 
                Initial value for the weight matrix of the 1st layer
                W1 should have shape (num_hidden, input_dim)
                Default is None

            b1 (numpy.array): 
                Initial value for the bias of the 1st layer
                b1 should have shape (num_hidden, )
                Default is None

            W2 (numpy.array): 
                Initial value for the weight matrix of the 2nd layer
                W2 should have shape (1, num_hidden)
                Default is None

            b2 (numpy.float): 
                Initial value for the bias of the 1st layer
                b2 should be a scalar
                Default is None

        Example usage:
            >>> init_param = utils.load_initial_weights("P3/Synthetic-Dataset/InitParams/relu/5")
            >>> nnet_classification = NeuralNetworkClassification(input_dim=10, num_hidden=5)
            >>> nnet_classification.fit(X_train, y_train, step_size=0.01)

        """
        super(NeuralNetworkClassification, self).__init__(
            input_dim, num_hidden, activation, W1, b1, W2, b2
        )

    def forwardProp(self, Xi):
        # xi should be dimension d 
        self.z = np.dot(Xi, self.layers['W1'].T) + self.layers['b1']
        self.z2 = self.g(self.z)  # activation function
        self.z3 = np.dot(self.z2, self.layers['W2'].T) + self.layers['b2']
        result = sigmoid(self.z3) 
        return result
        
    def back_propagate(self, X, y):
        """ Perform back propagation

        Args:
            X (numpy.array):
                Input data matrix. X should have shape (num_instances, input_dim)
            
            y (numpy.array):
                Binary {-1, +1} label vector. y should have shape (num_instances,)

        Returns:
            derivative (dictionary):
                A dictionary that contains the derivative of the loss function
                with respect to the parameters of the neural networks.
                For example,
                derivative["W1"] should contain a numpy array that is the derivative
                of the loss function with respect to W1. It should have the same shape
                as W1
        """
        
        # Your code should go here
        # first use existing weights to calculate output 
        res = self.forwardProp(X)
        numinstances = X.shape[0]
        o = res.flatten() - ((y + 1) / 2)
        
        # first compute w2
        w2temp = np.zeros(self.d1)
        b2temp = 0
        w1temp = np.zeros((self.d1, self.d))
        b1temp = np.zeros(self.d1)

        for i in range(numinstances):
            #w2 calculation
            w2i = o[i] * self.z2[i]
            w2temp += w2i
            
            #bias2 calculation
            b2i = o[i]
            b2temp += b2i
            
            #w1 calculation
            w1i = self.layers['W2'] * self.g_prime(self.z[i]) * o[i]
            w1i = np.outer(w1i, X[i])
            w1temp += w1i
            
            #bias1 calculation
            b1i = self.layers['W2'] * self.g_prime(self.z[i])
            b1i = b1i * o[i]
            b1temp += b1i.flatten()
            
        # compute averages
        w2temp /= numinstances
        w2_deriv = w2temp
        
        b2temp /= numinstances
        b2_deriv = b2temp
        
        w1temp /= numinstances
        w1_deriv = w1temp 
        
        b1temp /= numinstances
        b1_deriv = b1temp
        
        return {
                # the keys here are selected to match those in self.layers
                # (initialized on line 70)
                "W1": w1_deriv,
                "b1": b1_deriv,
                "W2": w2_deriv,
                "b2": b2_deriv
                }
    
    def predict(self, X):
        """ Do prediction on given input data
        Arg:
            X (numpy.array):
                Input data matrix. X should have shape (num_instances, input_dim)
        
        Returns:
            y (numpy.array):
                Binary {-1, +1} label vector. y should have shape (num_instances,)

        """
        y = self.forwardProp(X).flatten()
        y = np.sign(y - 0.5)
        # Your code should go here
        return y

