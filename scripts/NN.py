# Imports
import numpy as np
import torch


def act_logistic(x):
    '''
    Returns result of the logistic function.
    '''

    s = 1 / (1 + np.exp(-x))

    return s


def act_logistic_prime(x):
    '''
    Returns result of the derivative of the logistic function.
    '''

    a = act_logistic(x)*(1-act_logistic(x))

    return a


def loss_ss(yhat, y):
    '''
    Returns sum of squares loss function.
    '''

    a = (np.sum(np.power((yhat - y), 2)))

    return a


def loss_ss_prime(yhat, y):
    '''
    Returns the derivative of the sum of squares loss function.
    '''

    a = (yhat - y)

    return a


class NeuralNetwork():
    '''
    Arbitrary sized neural network.

    Parameters
    ----------
    dimensions: list
        List of dimensions of neural network to be constructed.
    
    Attributes
    ----------
    dimensions : list
        List of dimensions of neural network to be constructed.
    train_array : np.array
        Training data.
    train_labels : np.array
        Labels for training data.
    n_epochs : int
        Number of training iterations.
    learning_rate : int
        Rate of minimizing loss function per epoch.
    nlayers : int
        Number of layers in neural network.
    weights : list of np.arrays
        Neural network weights.
    biases : list of np.arrays
        Neural network biases.
    activations : list of np.arrays
        Neural network outputs of activation functions.
    z_inputs : list of np.arrays
        Neural network outputs of weight and bias functions.
    loss_list : list
        Model loss per epoch.
    grad_biases : list of np.arrays
        Neural network outputs of the gradients of the bias functions.
    grad_weights : list of np.arrays
        Neural network outputs of the gradients of the weight functions.
    pred: np.array
        Neural network model predictions.

    Examples
    --------
    >>> net = nn.NeuralNetwork([8, 3, 8])
    >>> net.initialize_model()
    >>> ...
    ...
    ...
    '''

    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.train_array = np.array([])
        self.train_labels = np.array([])
        self.n_epochs = int()
        self.learning_rate = int()
        self.nlayers = int()
        self.weights = []
        self.biases = []
        self.activations = []
        self.z_inputs = []
        self.loss_list = []
        self.grad_biases = []
        self.grad_weights = []
        self.test_array = np.array([])
        self.test_labels = np.array([])
        self.pred = np.array([])

    def _initialize_model(self):
        '''
        Given a list of positive integers which represent the network 
        dimensions, randomly generate model wieghts and biases.
        '''

        self.nlayers = len(self.dimensions)

        for i in range(self.nlayers-1):
            w = np.random.randn(self.dimensions[i+1], self.dimensions[i])
            b = np.random.randn(self.dimensions[i+1], 1)
            self.weights.append(w)
            self.biases.append(b)


    def _forward_prop(self, train_array, train_labels):
        '''
        Given an input layer of training data, the model is forwared 
        propagated. Weighted sums and the activations are calculated.

        Parameters
        ----------
        train_array : np.array
            Training data.
        train_labels : np.array
            Labels for training data.
        '''

        a = train_array
        self.train_array, self.train_labels = train_array, train_labels
        self.activations, self.z_inputs = [a], [] # Initialize lists

        for i in range(self.nlayers - 1):
            z = (self.weights[i] @ a) + self.biases[i]
            a = act_logistic(z)
            self.z_inputs.append(z)
            self.activations.append(a)
        
        # Calculate loss per epoch
        err = loss_ss(self.activations[-1], self.train_labels)
        loss = err/train_array.shape[1]

        self.loss_list.append(loss)


    def _backward_prop(self):
        '''
        Given the labels of the neural network output layer, 
        the model is backword propagated. Gradients of the bias
        functions and the gradients of the weight functions are calculated.
        '''

        Nbatch = self.train_labels.shape[1]
        yhat = self.activations[-1]

        for i in reversed(range(self.nlayers - 1)):
            if i == (self.nlayers - 2):
                w = self.weights[i]
                z = self.z_inputs[i]
                a = self.activations[i]
                loss = np.multiply(loss_ss_prime(yhat, self.train_labels), act_logistic_prime(z))
            else:
                z = self.z_inputs[i]
                a = self.activations[i]
                loss = np.multiply(np.transpose(w) @ loss, act_logistic_prime(z))
                w = self.weights[i]

            self.grad_biases.insert(0, loss @ np.ones((Nbatch, 1)))
            self.grad_weights.insert(0, loss @ np.transpose(a))


    def _update(self, learning_rate):
        '''
        Given a scaler learning rate the model is updated.
        
        Model weights and biases are updated using the gradients
        of both the bias functions and the weight functions that 
        were previously calculated.
        
        Weighted inputs, activations, gradients of the bias functions, 
        and gradients of the weight functions are reset.

        Parameters
        ----------
        learning_rate : int
            Rate of minimizing loss function per epoch.
        '''

        self.learning_rate = learning_rate

        for i in range(self.nlayers - 1):
            self.biases[i] += -self.learning_rate*self.grad_biases[i]
            self.weights[i] += -self.learning_rate*self.grad_weights[i]

        self.z_inputs, self.activations, \
        self.grad_biases, self.grad_weights = [], [], [], []


    def fit(self, train_array, train_labels, n_epochs, learning_rate):
        '''
        Fit the neural network given training data and labels.
        
        Parameters
        ----------
        train_array : np.array
            Training data.
        train_labels : np.array
            Labels for training data.
        n_epochs : int
            Number of training iterations.
        learning_rate : int
            Rate of minimizing loss function per epoch.
        '''

        self.n_epochs = n_epochs

        # Training loop
        for _ in range(self.n_epochs):
            self._initialize_model()
            self._forward_prop(train_array, train_labels)
            self._backward_prop()
            self._update(learning_rate)

    def predict(self, test_array, test_labels):
        '''
        Given an input layer of testing data, the trained model is forwared 
        propagated. Weighted sums and the activations are calculated
        and the model prediction is recorded.

        Parameters
        ----------
        test_array : np.array
            Testing data.
        test_labels : np.array
            Labels for test data.
            
        Returns
        ----------
        pred: np.array
            Neural network model predictions.
        '''

        a = test_array
        self.test_array, self.test_labels = test_array, test_labels
        self.activations, self.z_inputs = [a], []

        for i in range(self.nlayers - 1):
            z = (self.weights[i] @ a) + self.biases[i]
            a = act_logistic(z)
            self.z_inputs.append(z)
            self.activations.append(a)
        
        self.pred = self.activations[-1]

        return self.pred
