"""
Some helpful utility functions.

Yujia Li, 09/2014
"""

import gnumpy as gnp
import numpy as np

def to_garray(x):
    return x if isinstance(x, gnp.garray) else gnp.garray(x)

def to_nparray(x):
    return x if isinstance(x, np.ndarray) else x.asarray()

def to_one_of_K(t, K=None):
    n_cases = t.size
    if K is None:
        K = t.max() + 1
    if len(t.shape) > 0:
        t = t.ravel()

    t_mat = np.zeros((n_cases, K))
    t_mat[np.arange(n_cases), t] = 1
    return gnp.garray(t_mat)

def to_plus_minus_of_K(t, K=None):
    """
    Convert the 1-D label vector into a matrix where the t[i]th element on the
    ith row is 1 and all others on that row is -1.
    """
    n_cases = t.size
    if K is None:
        K = t.max() + 1
    if len(t.shape) > 0:
        t = t.ravel()

    t_mat = -np.ones((n_cases, K))
    t_mat[np.arange(n_cases), t] = 1
    return gnp.garray(t_mat)

