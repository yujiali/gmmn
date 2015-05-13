"""
Data I/O for mnist dataset.

Yujia Li, 01/2015
"""

import cPickle as pickle
import numpy as np

# Fill in the path to your mnist data here
#
# This data file is supposed to be a pickled dictionary containing numpy arrays
# including train_data, test_data, train_labels, test_labels.  For train_data
# and test_data, they are matrices of size NxD, where N is the number of data
# points and D=784 (28x28) is the size of the image.  Each row is a data point,
# assumed to be already normalized to [0,1].  For train_label and test_label, 
# they are matrices of size Nx1, each label is an integer from 0 to 9.
_DATA_FILE_PATH = 'path/to/your/mnist/data'

def load_raw_data():
    """
    Return the original train/test split.
    """
    with open(_DATA_FILE_PATH) as f:
        d = pickle.load(f)

    return d['train_data'], d['test_data']

def load_data():
    """
    Split part of training data to be used as validation data.
    """
    with open(_DATA_FILE_PATH) as f:
        d = pickle.load(f)

    x_train = d['train_data']
    x_test  = d['test_data']

    # keep current state of random number generator
    rand_state = np.random.get_state()

    np.random.seed(0)
    idx = np.random.permutation(x_train.shape[0])

    n_val = 5000
    x_val = x_train[idx[:n_val]]
    x_train = x_train[idx[n_val:]]

    # restore the state of random number generator
    np.random.set_state(rand_state)

    return x_train, x_val, x_test

def load_labeled_data(n_val=5000):
    """
    Load both the data and the labels.
    """
    with open(_DATA_FILE_PATH) as f:
        d = pickle.load(f)

    x_train = d['train_data']
    t_train = d['train_label']
    
    x_test = d['test_data']
    t_test = d['test_label']

    rand_state = np.random.get_state()

    np.random.seed(0)
    idx = np.random.permutation(x_train.shape[0])

    x_val = x_train[idx[:n_val]]
    t_val = t_train[idx[:n_val]]
    x_train = x_train[idx[n_val:]]
    t_train = t_train[idx[n_val:]]

    np.random.set_state(rand_state)

    return x_train, t_train, x_val, t_val, x_test, t_test


