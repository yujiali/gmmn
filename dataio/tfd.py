"""
The Toronto Face Database, Charlie/Marc'Aurelio's version

Yujia Li, 01/2015
"""

import scipy.io as sio
import numpy as np

# Fill in your TFD path here
_TFD_DATA_PATH_FORMAT = 'path/to/your/TFD_ranzato_%dx%d.mat'

def _load_raw_data(image_size=48):
    d = sio.loadmat(_TFD_DATA_PATH_FORMAT % (image_size, image_size))
    return d['images'], d['folds'], d['labs_id'].squeeze(), d['labs_ex'].squeeze()

def get_fixed_rand_permutation(size, seed=1):
    rand_state = np.random.get_state()
    np.random.seed(seed)
    idx = np.random.permutation(size)
    np.random.set_state(rand_state)

    return idx

class TFD(object):
    def __init__(self, image_size=48):
        self.images, self.folds, self.labs_id, self.labs_ex = \
                _load_raw_data(image_size)

        self._val_sizes = [(self.folds[:,fold] == 2).sum() for fold in range(5)]
        self._val_idx_start = np.array([0] + self._val_sizes).cumsum()

    def get_fold(self, fold, set_name, center=False, scale=False):
        """
        0 <= fold < 5
        set_name should be one of {train, val, test, unlabeled}

        Return images, labs_id, and labs_ex.

        There are two labels available: identity and expression.  For 
        unsupervised learning tasks these labels are not useful though.  The
        quality of these labels are also not very high.
        """
        set_map = {'unlabeled': 0, 'train' : 1, 'val': 2, 'test': 3}
        set_id = set_map[set_name]
        data_mask = (self.folds[:,fold] == set_id)

        images = self.images[data_mask].astype(np.float32)
        labs_id = self.labs_id[data_mask]
        labs_ex = self.labs_ex[data_mask]

        if center and scale:
            images -= 127.5
            images /= 127.5
        elif center:
            images -= 127.5
        elif scale:
            images /= 255.0

        return images, labs_id, labs_ex

    def get_proper_fold(self, fold, set_name, center=False, scale=False):
        """
        Same as get_fold, except that the validation sets across folds will be
        disjoint from test sets and training sets - so validation is proper.
        """
        set_map = {'unlabeled': 0, 'train' : 1, 'val': 2, 'test': 3}
        set_id = set_map[set_name]

        if set_id == 0 or set_id == 2:
            data_mask = (self.folds[:,fold] == 0)
            unlabeled_idx = np.arange(self.folds.shape[0])[data_mask]
            idx = get_fixed_rand_permutation(unlabeled_idx.size)
            data_mask = np.zeros(data_mask.size, dtype=np.bool)
            if set_id == 2:
                data_mask[idx[self._val_idx_start[fold]:self._val_idx_start[fold+1]]] = True
            else:
                data_mask[idx[self._val_idx_start[-1]:]] = True
        else:
            data_mask = (self.folds[:,fold] == set_id)

        images = self.images[data_mask].astype(np.float32)
        labs_id = self.labs_id[data_mask]
        labs_ex = self.labs_ex[data_mask]

        if center and scale:
            images -= 127.5
            images /= 127.5
        elif center:
            images -= 127.5
        elif scale:
            images /= 255.0

        return images, labs_id, labs_ex
        


_tfd = {48: None, 96: None}

def load_fold(fold, set_name, center=False, scale=False, image_size=48):
    if image_size != 48 and image_size != 96:
        raise Exception('image_size has to be either 48 or 96!')

    if _tfd[image_size] is None:
        _tfd[image_size] = TFD(image_size) # load data the first time we use it

    return _tfd[image_size].get_fold(fold, set_name, center, scale)

def load_proper_fold(fold, set_name, center=False, scale=False, image_size=48):
    if image_size != 48 and image_size != 96:
        raise Exception('image_size has to be either 48 or 96!')

    if _tfd[image_size] is None:
        _tfd[image_size] = TFD(image_size) # load data the first time we use it

    return _tfd[image_size].get_proper_fold(fold, set_name, center, scale)

