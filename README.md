# Project 3 - Neural Networks
## Due 03/19/2021

[![HW3](https://github.com/zagerpatrick/Project3/actions/workflows/test.yml/badge.svg)](https://github.com/zagerpatrick/Project3/actions/workflows/test.yml)

def act(x, d):
    '''
    Returns network activation functions.
    '''


def act_prime(x, d):
    '''
    Returns the derivatives of the network activation functions. 
    '''


def loss_ss(yhat, y):
    '''
    Returns sum of squares loss function.
    '''


def loss_ss_prime(yhat, y):
    '''
    Returns the derivative of the sum of squares loss function.
    '''


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
    break_lim = float
        loss limit to stop model training.
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
    self.act_type : string
        Network activation type.
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
    '''

    def _initialize_model(self):
        '''
        Given a list of positive integers which represent the network 
        dimensions, randomly generate model wieghts and biases.
        '''


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


    def _backward_prop(self):
        '''
        Given the labels of the neural network output layer, 
        the model is backword propagated. Gradients of the bias
        functions and the gradients of the weight functions are calculated.
        '''


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


    def predict(self, test_array):
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


def txt2str(file):
    '''
    Returns strings of sequences from a text file.
    
    Parameters
    ----------
    file : str
        A path to a .txt file containing nucleotide or amino acid sequences.
    Returns
    ----------
    seq: list
        A list of nucleotide sequence strings.
    '''


def fa2str(file):
    '''
    Returns strings of sequences from a fasta file.
    
    Parameters
    ----------
    file : str
        A path to a fasta (.fa) file containing 
        nucleotide or amino acid sequences.
    Returns
    ----------
    seq: list
        A list of nucleotide sequence strings.
    '''


def one_hot_dna(seq, exp_len):
    '''
    One-hot encodes DNA sequence data.
    
    Parameters
    ----------
    seq : list
        Input list of DNA sequences (str).
    exp_len : int
        Expected length of output sequences.
    Returns
    ----------
    encode: list
        List of one-hot encoded DNA sequences.
    '''


def gen_label_array(s):
    '''
    Generate a label array of size (m, n), where each column contains 
    m-1 zeros and a single one value.
    
    Parameters
    ----------
    s : tuple
        Tuple of label array size (m, n).
    Returns
    ----------
    lab: np.array
        Array where each column is a single label array.
    '''


def sample_array(array, samp_size, freq):
    '''
    Sample an array continuously along the rows.
    
    Parameters
    ----------
    array : np.array
        Array to be sampled from.
    samp_size : int
        Length of range of values to be samples continuously.
    freq : int
        frequency of sampling.
    Returns
    ----------
    sample : np.array
        Samples array.
    '''


def train_test_split(array, train_num):
    '''
    Split an array randomly along columns into training and testing arrays.
    
    Parameters
    ----------
    array : np.array
        Array of data to be split along columns.
    train_num : 
        Number of columns to be in training array.
    Returns
    ----------
    train_array : np.array
        Array of training data.
    test_array : np.array
        Array of testing data.
    '''


def split(a, n):
    '''
    Split a list or 1D array into approximately equal sized lists or 1D arrays.
    
    Parameters
    ----------
    a : np.array or list
        Array or list of data to be split.
    n : int
        Number of sub lists or arrays to output.
    Returns
    ----------
    train_array : np.array
        Array of training data.
    test_array : np.array
        Array of testing data.    
    '''


def pred_gen(scores):
    '''
    Generates list of binary predictions for all possible threshold values.
    
    Parameters
    ----------
    scores : np.array
        Array of predicted values.
    Returns
    ----------
    pred_list : list
        Lists of arrays of binary predictions.
    '''


def pr_calc(actual, prediction_list):
    '''
    Calculates true positive rate and false positive rate for lists of binary
    predictions.
    
    Parameters
    ----------
    actual : np.array
        Array of ground truth binra values.
    prediction_list : list
        Lists of arrays of binary predictions.
    Returns
    ----------
    tpr : list
        List of true positive rate values.
    fpr : list
        List of false positive rate values.
    '''
