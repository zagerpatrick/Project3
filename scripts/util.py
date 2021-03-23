# Imports
import random
import numpy as np
from sklearn.metrics import confusion_matrix, auc


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

    d = {'A': 0, 'T':1, 'G':2, 'C':3}
    
    encode = []
    for dna in seq:
        one_hot_list = []
        for nuc in dna:
            c = d[nuc]
            m = np.zeros([4, 1])
            m[c] = 1
            one_hot_list.append(m)
        if len(one_hot_list) != exp_len:
            continue
        one_hot_array = np.vstack(one_hot_list)
        encode.append(one_hot_array)
    
    return encode


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

    m, n = s[0], s[1]

    values = np.random.choice(list(range(0, m)), size=(1, n))
    n_values = np.max(values) + 1
    value_array = np.eye(n_values)[values]
    
    lab = value_array[0, :, :].T

    return lab


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

    t = array.shape[0]/freq
    r = samp_size*freq
    
    sample_list = []
    for i in range(0, array.shape[1]):
        n = random.randint(0, t-samp_size)*freq
        sample_list.append(array[n:n+r, i:i+1])
    
    sample = np.hstack(sample_list)
    
    return sample


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

    full_ind = list(range(0, array.shape[1]))
    train_ind = np.random.choice(array.T.shape[0], train_num, replace=False)
    test_ind =  np.array([x for x in full_ind if x not in train_ind])
    
    train_array = array[:, tuple(train_ind)]
    test_array = array[:, tuple(test_ind)]
    
    return train_array, test_array


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

    k, m = divmod(len(a), n)
    s = (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
    
    return s 


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

    pred_list = []

    for thresh in np.sort(scores):
        pred = []
        for value in scores:
            if value >= thresh:
                pred.append(1)
            else:
                pred.append(0)
        pred_list.append(pred)

    return pred_list


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

    tpr, fpr = [], []

    for prediction in prediction_list:
        cm = confusion_matrix(actual, prediction)
        tn, fp, fn, tp = cm.ravel()
        tpr.append(tp/(tp + fn))
        fpr.append(fp/(fp + tn))

    return tpr, fpr
