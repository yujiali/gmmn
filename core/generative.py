"""
Generative model using MMD objective.

Yujia Li, 09/2014
"""

import pynn.nn as nn
import pynn.loss as ls
import pynn.learner as learner
import gnumpy as gnp
import numpy as np
import math
import util
import time
import scipy.optimize as spopt

class UnsupervisedMmdLoss(ls.Loss):
    """
    MMD loss for unsupervised learning.

    This loss measures the discrepancy between a distribution given by a 
    neural net model with a data distribution.
    """
    def __init__(self, **kwargs):
        super(UnsupervisedMmdLoss, self).__init__(**kwargs)
        self.sigma = kwargs.get('sigma', 1)

    def load_target(self, target, **kwargs):
        """
        target is the target data distribution, n_cases * n_dims matrix.
        """
        if isinstance(target, gnp.garray):
            self.target = target
        else:
            self.target = gnp.garray(target)

        self.n_target = target.shape[0]

    def _make_s_mat(self, n_pred, n_target):
        """
        Create the S matrix that will be used in loss computation.
        """
        s = gnp.zeros((n_pred + n_target, 2))
        s[:n_pred, 0] = 1.0 / n_pred
        s[n_pred:, 1] = 1.0 / n_target
        s -= 1.0 / (n_pred + n_target)
        return s

    def compute_not_weighted_loss_and_grad(self, pred, compute_grad=False):
        if not isinstance(pred, gnp.garray):
            pred = gnp.garray(pred)

        n_pred = pred.shape[0]
        W = self._make_s_mat(n_pred, self.n_target)
        X = gnp.concatenate((pred, self.target), axis=0)

        XX = X.dot(X.T)
        if XX.shape[0] > 4000:  # this special case is due to a weird bug in gnumpy
            x = gnp.garray(np.diag(XX.asarray()))
        else:
            x = XX.diag()

        K = gnp.exp(-1.0 / (2 * self.sigma) * (-2 * XX + x + x[:,gnp.newaxis]))
        A = W.dot(W.T) * K

        loss = A.sum()
        a = A.sum(axis=1)
        grad = 2.0 / self.sigma * (A.dot(X) - X * a[:,gnp.newaxis])

        return loss, grad[:n_pred,:]

    def get_name(self):
        return 'mmdgen'

    def get_id(self):
        return 201

    def __repr__(self):
        return 'Loss <%s> w=%g, sigma=%g' % (
                self.get_name(), self.weight, self.sigma)

ls.register_loss(UnsupervisedMmdLoss())

class UnsupervisedMmdLossMultiScale(ls.Loss):
    """
    Multi-scale MMD loss for unsupervised learning.

    This loss measures the discrepancy between a distribution given by a 
    neural net model with a data distribution.
    """
    def __init__(self, sigma=[1.0], scale_weight=None, **kwargs):
        super(UnsupervisedMmdLossMultiScale, self).__init__(**kwargs)
        self.sigma = [float(s) for s in sigma]
        self.n_scales = len(sigma)

        if scale_weight is None:
            self.scale_weight = [1.0] * self.n_scales
        else:
            assert(len(scale_weight) == len(sigma))
            self.scale_weight = [float(w) for w in scale_weight]

    def load_target(self, target, **kwargs):
        """
        target is the target data distribution, n_cases * n_dims matrix.
        """
        if isinstance(target, gnp.garray):
            self.target = target
        else:
            self.target = gnp.garray(target)

        self.n_target = target.shape[0]

    def _make_s_mat(self, n_pred, n_target):
        """
        Create the S matrix that will be used in loss computation.
        """
        s = gnp.zeros((n_pred + n_target, 2))
        s[:n_pred, 0] = 1.0 / n_pred
        s[n_pred:, 1] = 1.0 / n_target
        s -= 1.0 / (n_pred + n_target)
        return s

    def compute_not_weighted_loss_and_grad(self, pred, compute_grad=False):
        if not isinstance(pred, gnp.garray):
            pred = gnp.garray(pred)

        n_pred = pred.shape[0]
        W = self._make_s_mat(n_pred, self.n_target)
        X = gnp.concatenate((pred, self.target), axis=0)

        XX = X.dot(X.T)
        if XX.shape[0] > 4000:  # this special case is due to a weird bug in gnumpy
            x = gnp.garray(np.diag(XX.asarray()))
        else:
            x = XX.diag()

        prod_mat = XX - 0.5 * x - 0.5 * x[:,gnp.newaxis]
        ww = W.dot(W.T)

        loss = 0
        grad = None
        for i in range(self.n_scales):
            K = gnp.exp(1.0 / self.sigma[i] * prod_mat)
            A = self.scale_weight[i] * ww * K
            loss += A.sum()
            a = A.sum(axis=1)
            if grad is None:
                grad = 2.0 / self.sigma[i] * (A.dot(X) - X * a[:,gnp.newaxis])
            else:
                grad += 2.0 / self.sigma[i] * (A.dot(X) - X * a[:,gnp.newaxis])

        return loss, grad[:n_pred,:]

    def get_name(self):
        return 'mmdgen_multiscale'

    def get_id(self):
        return 202

    def __repr__(self):
        return 'Loss <%s> w=%g, sigma=%s, scale_weight=%s' % (
                self.get_name(), self.weight, str(self.sigma), str(self.scale_weight))

ls.register_loss(UnsupervisedMmdLossMultiScale())

class LinearTimeUnsupervisedMmdLoss(ls.Loss):
    """
    MMD loss for unsupervised learning.

    This loss measures the discrepancy between a distribution given by a 
    neural net model with a data distribution.

    This is the linear time estimator proposed by Gretton et al.
    """
    def __init__(self, **kwargs):
        super(LinearTimeUnsupervisedMmdLoss, self).__init__(**kwargs)
        self.use_modified_loss = kwargs.get('use_modified_loss', False)
        self.use_absolute_value = kwargs.get('use_absolute_value', True)
        self.sigma = kwargs.get('sigma', 1)

    def load_target(self, target, **kwargs):
        """
        target is the target data distribution, n_cases * n_dims matrix.
        """
        if isinstance(target, gnp.garray):
            self.target = target
        else:
            self.target = gnp.garray(target)

        self.n_target = target.shape[0]

    def compute_not_weighted_loss_and_grad(self, pred, compute_grad=False):
        if not isinstance(pred, gnp.garray):
            pred = gnp.garray(pred)

        n_pred = pred.shape[0]
        assert n_pred == self.n_target
        assert n_pred % 2 == 0

        n_half = n_pred / 2

        X = pred[:n_half]
        X_N = pred[n_half:]
        Y = self.target[:n_half]
        Y_N = self.target[n_half:]

        diff_x_xn = X - X_N
        diff_x_yn = X - Y_N
        diff_xn_y = X_N - Y
        diff_y_yn = Y - Y_N


        factor = -0.5 / self.sigma

        k_x_xn = gnp.exp(factor * (diff_x_xn**2).sum(axis=1))
        k_y_yn = gnp.exp(factor * (diff_y_yn**2).sum(axis=1))
        k_x_yn = gnp.exp(factor * (diff_x_yn**2).sum(axis=1))
        k_xn_y = gnp.exp(factor * (diff_xn_y**2).sum(axis=1))

        loss = 1.0 / n_pred * (k_x_xn.sum() + k_y_yn.sum() - k_x_yn.sum() - k_xn_y.sum())
        grad_x = 1.0 / (n_pred * self.sigma) * (k_x_yn[:,gnp.newaxis] * diff_x_yn - k_x_xn[:,gnp.newaxis] * diff_x_xn)
        grad_xn = 1.0 / (n_pred * self.sigma) * (k_xn_y[:,gnp.newaxis] * diff_xn_y + k_x_xn[:,gnp.newaxis] * diff_x_xn)

        if self.use_modified_loss:
            diff_x_y = X - Y
            diff_xn_yn = X_N - Y_N
            k_x_y = gnp.exp(factor * (diff_x_y**2).sum(axis=1))
            k_xn_yn = gnp.exp(factor * (diff_xn_yn**2).sum(axis=1))

            loss += 1.0 / n_pred * (k_x_xn.sum() + k_y_yn.sum() - k_x_y.sum() - k_xn_yn.sum())
            grad_x += 1.0 / (n_pred * self.sigma) * (k_x_y[:,gnp.newaxis] * diff_x_y - k_x_xn[:,gnp.newaxis] * diff_x_xn)
            grad_xn += 1.0 / (n_pred * self.sigma) * (k_xn_yn[:,gnp.newaxis] * diff_xn_yn + k_x_xn[:,gnp.newaxis] * diff_x_xn)

        grad = gnp.concatenate([grad_x, grad_xn], axis=0)

        if self.use_absolute_value and loss < 0:
            loss = -loss
            grad = -grad

        return loss, grad

    def get_name(self):
        return 'linear_time_mmdgen'

    def get_id(self):
        return 203

    def __repr__(self):
        return 'Loss <%s> w=%g, sigma=%g' % (
                self.get_name(), self.weight, self.sigma)

ls.register_loss(LinearTimeUnsupervisedMmdLoss())

class LinearTimeMinibatchUnsupervisedMmdLoss(ls.Loss):
    """
    MMD loss for unsupervised learning.

    This loss measures the discrepancy between a distribution given by a 
    neural net model with a data distribution.

    This is a version where the full MMD is only computed on minibatches,
    therefore the time complexity for a set of N pairs of data points and
    minibatch size M is O(N/M * M^2) = O(NM)
    """
    def __init__(self, **kwargs):
        super(LinearTimeMinibatchUnsupervisedMmdLoss, self).__init__(**kwargs)
        self.sigma = kwargs.get('sigma', 1)
        self.minibatch_size = kwargs.get('minibatch_size', 100)

    def load_target(self, target, **kwargs):
        """
        target is the target data distribution, n_cases * n_dims matrix.
        """
        if isinstance(target, gnp.garray):
            self.target = target
        else:
            self.target = gnp.garray(target)

        self.n_target = target.shape[0]

    def _make_s_mat(self, n_pred, n_target):
        """
        Create the S matrix that will be used in loss computation.
        """
        s = gnp.zeros((n_pred + n_target, 2))
        s[:n_pred, 0] = 1.0 / n_pred
        s[n_pred:, 1] = 1.0 / n_target
        s -= 1.0 / (n_pred + n_target)
        return s

    def compute_not_weighted_loss_and_grad(self, pred, compute_grad=False):
        if not isinstance(pred, gnp.garray):
            pred = gnp.garray(pred)

        n_pred = pred.shape[0]
        assert n_pred == self.n_target

        W_Full = self._make_s_mat(n_pred, self.n_target)

        loss = 0
        grad = []

        n_batches = (n_pred + self.minibatch_size - 1) / self.minibatch_size
        for i_batch in range(n_batches):
            i_start = i_batch * self.minibatch_size
            if i_batch < n_batches - 1:
                i_end = i_start + self.minibatch_size
            else:
                i_end = n_pred

            X = gnp.concatenate((pred[i_start:i_end], self.target[i_start:i_end]), axis=0)
            W = self._make_s_mat(i_end - i_start, i_end - i_start)
            
            XX = X.dot(X.T)
            if XX.shape[0] > 4000:  # this special case is due to a weird bug in gnumpy
                x = gnp.garray(np.diag(XX.asarray()))
            else:
                x = XX.diag()

            K = gnp.exp(-1.0 / (2 * self.sigma) * (-2 * XX + x + x[:,gnp.newaxis]))
            A = W.dot(W.T) * K

            loss += A.sum()
            a = A.sum(axis=1)
            grad.append((2.0 / self.sigma * (A.dot(X) - X * a[:,gnp.newaxis]))[:(i_end - i_start)])

        return loss / n_batches, gnp.concatenate(grad, axis=0) / n_batches

    def get_name(self):
        return 'linear_time_minibatch_mmdgen'

    def get_id(self):
        return 204

    def __repr__(self):
        return 'Loss <%s> w=%g, sigma=%g, minibatch_size=%d' % (
                self.get_name(), self.weight, self.sigma, self.minibatch_size)

ls.register_loss(LinearTimeMinibatchUnsupervisedMmdLoss())

class RandomFeatureMmdLoss(ls.Loss):
    """
    MMD loss for unsupervised learning.

    This loss measures the discrepancy between a distribution given by a 
    neural net model with a data distribution.

    This is a version where the kernel k(x,y) is estimated by product of random
    features.
    """
    def __init__(self, sigma=[1.0], scale_weight=None, n_features=1024, **kwargs):
        super(RandomFeatureMmdLoss, self).__init__(**kwargs)
        self.original_sigma = sigma
        self.sigma = [np.sqrt(float(s)) for s in sigma]
        self.n_scales = len(sigma)

        if scale_weight is None:
            self.scale_weight = [1.0] * self.n_scales
        else:
            assert(len(scale_weight) == len(sigma))
            self.scale_weight = [float(w) for w in scale_weight]

        self.n_features = n_features

    def _generate_random_matrix(self, n_features, n_dims, sigma):
        """
        return a list of random matrices each of size n_features x n_dims
        """
        w = []
        for i in range(len(sigma)):
            w.append(gnp.randn(n_features, n_dims) / sigma[i])
        return w

    def _generate_random_features(self, x, w):
        return gnp.cos(x.dot(w.T)) / np.sqrt(self.n_features), \
                gnp.sin(x.dot(w.T)) / np.sqrt(self.n_features)

    def load_target(self, target, **kwargs):
        """
        target is the target data distribution, n_cases * n_dims matrix.
        """
        # actually target does not need to be stored
        if isinstance(target, gnp.garray):
            self.target = target
        else:
            self.target = gnp.garray(target)

        self.n_target = target.shape[0]
        self.w = self._generate_random_matrix(self.n_features, target.shape[1], self.sigma)

        self.v_target = []
        for w in self.w:
            t_c, t_s = self._generate_random_features(target, w)
            self.v_target.append((t_c.mean(axis=0), t_s.mean(axis=0)))

    def compute_not_weighted_loss_and_grad(self, pred, compute_grad=False):
        if not isinstance(pred, gnp.garray):
            pred = gnp.garray(pred)

        loss = 0
        grad = None
        for i in range(self.n_scales):
            w = self.w[i]
            x_c, x_s = self._generate_random_features(pred, w)
            d_c = x_c.mean(axis=0) - self.v_target[i][0]
            d_s = x_s.mean(axis=0) - self.v_target[i][1]

            loss += ((d_c**2).sum() + (d_s**2).sum()) * self.scale_weight[i]
            s_c = 2.0 / pred.shape[0] * d_c 
            s_s = 2.0 / pred.shape[0] * d_s

            g = (-x_s * s_c + x_c * s_s).dot(w) * self.scale_weight[i]

            if grad is None:
                grad = g
            else:
                grad += g

        return loss, grad

    def get_name(self):
        return 'random_feature_mmdgen'

    def get_id(self):
        return 205

    def __repr__(self):
        return 'Loss <%s> w=%g, nf=%d, sigma=%s, scale_weight=%s' % (
                self.get_name(), self.weight, self.n_features, str(self.original_sigma),
                str(self.scale_weight))

ls.register_loss(RandomFeatureMmdLoss())

class PairMmdLossMultiScale(ls.Loss):
    """
    Multi-scale MMD loss for unsupervised learning.

    This loss measures the discrepancy between a distribution given by a 
    neural net model with a data distribution.

    This class considers only a pair of distributions, rather than a set of 
    multiple distributions.
    """
    def __init__(self, sigma=[1.0], scale_weight=None, **kwargs):
        super(PairMmdLossMultiScale, self).__init__(**kwargs)
        self.sigma = [float(s) for s in sigma]
        self.n_scales = len(sigma)

        if scale_weight is None:
            self.scale_weight = [1.0] * self.n_scales
        else:
            assert(len(scale_weight) == len(sigma))
            self.scale_weight = [float(w) for w in scale_weight]

    def load_target(self, target, **kwargs):
        """
        target is the target data distribution, n_cases * n_dims matrix.
        """
        if isinstance(target, gnp.garray):
            self.target = target
        else:
            self.target = gnp.garray(target)

        self.n_target = target.shape[0]

    def _make_s_mat(self, n_pred, n_target):
        """
        Create the S matrix that will be used in loss computation.
        """
        s = gnp.zeros((n_pred + n_target, 1))
        s[:n_pred] = 1
        s = s / n_pred - (1 - s) / n_target
        return s

    def compute_not_weighted_loss_and_grad(self, pred, compute_grad=False):
        if not isinstance(pred, gnp.garray):
            pred = gnp.garray(pred)

        n_pred = pred.shape[0]
        W = self._make_s_mat(n_pred, self.n_target)
        X = gnp.concatenate((pred, self.target), axis=0)

        XX = X.dot(X.T)
        if XX.shape[0] > 4000:  # this special case is due to a weird bug in gnumpy
            x = gnp.garray(np.diag(XX.asarray()))
        else:
            x = XX.diag()

        prod_mat = XX - 0.5 * x - 0.5 * x[:,gnp.newaxis]
        ww = W.dot(W.T)

        loss = 0
        grad = None
        for i in range(self.n_scales):
            K = gnp.exp(1.0 / self.sigma[i] * prod_mat)
            A = self.scale_weight[i] * ww * K
            loss += A.sum()
            a = A.sum(axis=1)
            if grad is None:
                grad = 2.0 / self.sigma[i] * (A.dot(X) - X * a[:,gnp.newaxis])
            else:
                grad += 2.0 / self.sigma[i] * (A.dot(X) - X * a[:,gnp.newaxis])

        return loss, grad[:n_pred,:]

    def get_name(self):
        return 'mmdgen_multiscale_pair'

    def get_id(self):
        return 206

    def __repr__(self):
        return 'Loss <%s> w=%g, sigma=%s, scale_weight=%s' % (
                self.get_name(), self.weight, str(self.sigma), str(self.scale_weight))

ls.register_loss(PairMmdLossMultiScale())

############################################################
# Some extensions to the loss
############################################################

class DifferentiableKernelMmdLoss(ls.Loss):
    """
    Base class for MMD loss with kernels that can be backpropagated through.
    """
    def __init__(self, **kwargs):
        super(DifferentiableKernelMmdLoss, self).__init__(**kwargs)

    def load_target(self, target, **kwargs):
        """
        target is the target data batch that we want our model to match.
        """
        self.target = util.to_garray(target)
        self.n_target = self.target.shape[0]

    def _make_s_mat(self, n_pred, n_target):
        """
        Make the S matrix. Here it is only a single vector as we have only two
        domains.

        The full set of data is always assumed to have the samples (pred) first
        and then the real data (target).
        """
        s = gnp.zeros((n_pred + n_target, 1))
        s[:n_pred] = 1
        s = s / n_pred - (1 - s) / n_target
        return s

        #s = gnp.zeros((n_pred + n_target, 2))
        #s[:n_pred, 0] = 1.0 / n_pred
        #s[n_pred:, 1] = 1.0 / n_target
        #s -= 1.0 / (n_pred + n_target)
        #return s

    def compute_not_weighted_loss_and_grad(self, pred, compute_grad=False):
        """
        Return loss and gradient
        """
        raise NotImplementedError()

class MultiScaleDifferentiableKernelMmdLoss(DifferentiableKernelMmdLoss):
    """
    Base class for MMD loss with kernels on multiple scales.
    """
    def __init__(self, sigma=[1.0], scale_weight=None, **kwargs):
        super(MultiScaleDifferentiableKernelMmdLoss, self).__init__(**kwargs)
        if not isinstance(sigma, list):
            sigma = [sigma]
        self.sigma = [float(s) for s in sigma]
        self.n_scales = len(sigma)

        if scale_weight is None:
            self.scale_weight = [1.0] * self.n_scales
        else:
            if not isinstance(scale_weight, list):
                scale_weight = [scale_weight]
            assert(len(scale_weight) == len(sigma))
            self.scale_weight = [float(w) for w in scale_weight]

class GaussianKernelMmdLoss(MultiScaleDifferentiableKernelMmdLoss):
    """
    k(x,y) = exp(-|x-y|^2 / (2 sigma))

    Multi-scale MMD loss with Gaussian kernels.  Essentially reimplementing 
    PairMmdLossMultiScale / UnsupervisedMmdLoss.
    """
    def __init__(self, sigma=[1.0], scale_weight=None, **kwargs):
        super(GaussianKernelMmdLoss, self).__init__(sigma=sigma, 
                scale_weight=scale_weight, **kwargs)

    def compute_not_weighted_loss_and_grad(self, pred, compute_grad=False):
        pred = util.to_garray(pred)
        n_pred = pred.shape[0]

        W = self._make_s_mat(n_pred, self.n_target)
        X = gnp.concatenate((pred, self.target), axis=0)

        XX = X.dot(X.T)
        if XX.shape[0] > 4000:  # this special case is due to a weird bug in gnumpy
            x = gnp.garray(np.diag(XX.asarray()))
        else:
            x = XX.diag()

        prod_mat = XX - 0.5 * x - 0.5 * x[:,gnp.newaxis]
        ww = W.dot(W.T)

        loss = 0
        grad = None
        for i in range(self.n_scales):
            K = gnp.exp(1.0 / self.sigma[i] * prod_mat)
            A = self.scale_weight[i] * ww * K
            loss += A.sum()
            a = A.sum(axis=1)
            if grad is None:
                grad = 2.0 / self.sigma[i] * (A.dot(X) - X * a[:,gnp.newaxis])
            else:
                grad += 2.0 / self.sigma[i] * (A.dot(X) - X * a[:,gnp.newaxis])

        return loss, grad[:n_pred,:]

    def get_name(self):
        return 'mmdgen_gaussian'

    def get_id(self):
        return 301

    def __repr__(self):
        return 'Loss <%s> w=%g, sigma=%s, scale_weight=%s' % (
                self.get_name(), self.weight, str(self.sigma), str(self.scale_weight))

ls.register_loss(GaussianKernelMmdLoss())

class LaplacianKernelMmdLoss(MultiScaleDifferentiableKernelMmdLoss):
    """
    k(x,y) = exp(-|x-y|_2/sigma)
    """
    def __init__(self, sigma=[1.0], scale_weight=None, **kwargs):
        super(LaplacianKernelMmdLoss, self).__init__(sigma=sigma, 
                scale_weight=scale_weight, **kwargs)

    def compute_not_weighted_loss_and_grad(self, pred, compute_grad=False):
        pred = util.to_garray(pred)
        n_pred = pred.shape[0]

        W = self._make_s_mat(n_pred, self.n_target)
        X = gnp.concatenate((pred, self.target), axis=0)

        ww = W.dot(W.T)

        XX = X.dot(X.T)
        if XX.shape[0] > 4000:  # this special case is due to a weird bug in gnumpy
            x = gnp.garray(np.diag(XX.asarray()))
        else:
            x = XX.diag()

        idx = np.arange(X.shape[0])
        zv = gnp.zeros(idx.size)

        # handle numeric problems
        _R = x + x[:,gnp.newaxis] - 2 * XX
        _R_min = _R.min()
        if _R_min < 1e-4:
            _R = _R - _R_min + 1e-4
            _R[idx,idx] = zv

        R = gnp.sqrt(_R)

        loss = 0
        grad = None

        for i in range(self.n_scales):
            K = gnp.exp(-1.0 / self.sigma[i] * R)
            L = self.scale_weight[i] * ww * K
            loss += L.sum()
            A = L / (R + gnp.eye(L.shape[0]))
            A[idx,idx] = zv
            a = A.sum(axis=1)
            if grad is None:
                grad = 2.0 / self.sigma[i] * (A.dot(X) - X * a[:,gnp.newaxis])
            else:
                grad += 2.0 / self.sigma[i] * (A.dot(X) - X * a[:,gnp.newaxis])
        
        return loss, grad[:n_pred,:]

    def get_name(self):
        return 'mmdgen_laplacian'

    def get_id(self):
        return 302

    def __repr__(self):
        return 'Loss <%s> w=%g, sigma=%s, scale_weight=%s' % (
                self.get_name(), self.weight, str(self.sigma), str(self.scale_weight))

ls.register_loss(LaplacianKernelMmdLoss())

class LaplacianL1KernelMmdLoss(MultiScaleDifferentiableKernelMmdLoss):
    """
    k(x,y) = exp(-|x-y|/sigma)
    """
    def __init__(self, sigma=[1.0], scale_weight=None, **kwargs):
        super(LaplacianL1KernelMmdLoss, self).__init__(sigma=sigma, 
                scale_weight=scale_weight, **kwargs)

    def compute_not_weighted_loss_and_grad(self, pred, compute_grad=False):
        pred = util.to_garray(pred)
        n_pred = pred.shape[0]

        W = self._make_s_mat(n_pred, self.n_target)
        X = gnp.concatenate((pred, self.target), axis=0)

        ww = W.dot(W.T)

        loss = 0
        grad = None
        
        for i in range(X.shape[0]):
            v = X[i]
            w = ww[i]
            diff = X - v

            a = diff.abs().sum(axis=1)
            for i_scale in range(self.n_scales):
                k = gnp.exp(-a / self.sigma[i_scale])
                loss += self.scale_weight[i_scale] * (w * k).sum()

                g = (self.scale_weight[i_scale] * w * k / self.sigma[i_scale])[:,gnp.newaxis] * ((diff < 0) - (diff > 0)) 
                g[i] = -g.sum(axis=0)
                if grad is None:
                    grad = g
                else:
                    grad += g

        return loss, grad[:n_pred,:]

    def get_name(self):
        return 'mmdgen_laplacian_l1'

    def get_id(self):
        return 303

    def __repr__(self):
        return 'Loss <%s> w=%g, sigma=%s, scale_weight=%s' % (
                self.get_name(), self.weight, str(self.sigma), str(self.scale_weight))

ls.register_loss(LaplacianL1KernelMmdLoss())

class SqrtGaussianKernelMmdLoss(GaussianKernelMmdLoss):
    """
    k(x,y) = sqrt{exp(-|x-y|^2 / (2 sigma))}

    Multi-scale MMD loss with Gaussian kernels.  Essentially reimplementing 
    PairMmdLossMultiScale / UnsupervisedMmdLoss.
    """
    def __init__(self, sigma=[1.0], scale_weight=None, **kwargs):
        super(SqrtGaussianKernelMmdLoss, self).__init__(sigma=sigma, 
                scale_weight=scale_weight, **kwargs)

    def compute_not_weighted_loss_and_grad(self, pred, compute_grad=False):
        loss, grad = super(SqrtGaussianKernelMmdLoss, self).compute_not_weighted_loss_and_grad(pred, compute_grad=compute_grad)
        sqrt_loss = math.sqrt(loss)
        return sqrt_loss, grad / (2 * sqrt_loss + 1e-10)

    def get_name(self):
        return 'mmdgen_sqrt_gaussian'

    def get_id(self):
        return 304

    def __repr__(self):
        return 'Loss <%s> w=%g, sigma=%s, scale_weight=%s' % (
                self.get_name(), self.weight, str(self.sigma), str(self.scale_weight))

ls.register_loss(SqrtGaussianKernelMmdLoss())

class CpuDifferentiableKernelMmdLoss(ls.Loss):
    """
    Base class for MMD loss with kernels that can be backpropagated through.
    """
    def __init__(self, **kwargs):
        super(CpuDifferentiableKernelMmdLoss, self).__init__(**kwargs)

    def load_target(self, target, **kwargs):
        """
        target is the target data batch that we want our model to match.
        """
        self.target = util.to_nparray(target)
        self.n_target = self.target.shape[0]

    def _make_s_mat(self, n_pred, n_target):
        """
        Make the S matrix. Here it is only a single vector as we have only two
        domains.

        The full set of data is always assumed to have the samples (pred) first
        and then the real data (target).
        """
        s = np.zeros((n_pred + n_target, 1), dtype=np.float32)
        s[:n_pred] = 1
        s = s / n_pred - (1 - s) / n_target
        return s

    def compute_not_weighted_loss_and_grad(self, pred, compute_grad=False):
        """
        Return loss and gradient
        """
        raise NotImplementedError()

class CpuMultiScaleDifferentiableKernelMmdLoss(CpuDifferentiableKernelMmdLoss):
    """
    Base class for MMD loss with kernels on multiple scales.
    """
    def __init__(self, sigma=[1.0], scale_weight=None, **kwargs):
        super(CpuMultiScaleDifferentiableKernelMmdLoss, self).__init__(**kwargs)
        if not isinstance(sigma, list):
            sigma = [sigma]
        self.sigma = [float(s) for s in sigma]
        self.n_scales = len(sigma)

        if scale_weight is None:
            self.scale_weight = [1.0] * self.n_scales
        else:
            if not isinstance(scale_weight, list):
                scale_weight = [scale_weight]
            assert(len(scale_weight) == len(sigma))
            self.scale_weight = [float(w) for w in scale_weight]

class CpuGaussianKernelMmdLoss(CpuMultiScaleDifferentiableKernelMmdLoss):
    """
    k(x,y) = exp(-|x-y|^2 / (2 sigma))

    Multi-scale MMD loss with Gaussian kernels.  Essentially reimplementing 
    PairMmdLossMultiScale / UnsupervisedMmdLoss.
    """
    def __init__(self, sigma=[1.0], scale_weight=None, **kwargs):
        super(CpuGaussianKernelMmdLoss, self).__init__(sigma=sigma, 
                scale_weight=scale_weight, **kwargs)

    def compute_not_weighted_loss_and_grad(self, pred, compute_grad=False):
        pred = util.to_nparray(pred)
        n_pred = pred.shape[0]

        W = self._make_s_mat(n_pred, self.n_target)
        X = np.concatenate((pred, self.target), axis=0)

        XX = X.dot(X.T)
        x = np.diag(XX)

        prod_mat = XX - 0.5 * x - 0.5 * x[:,np.newaxis]
        ww = W.dot(W.T)

        loss = 0
        grad = None

        K = self.scale_weight[0] * np.exp(1.0 / self.sigma[0] * prod_mat)
        scaled_K = K / self.sigma[0]
        for i in range(1, self.n_scales):
            T = self.scale_weight[i] * np.exp(1.0 / self.sigma[i] * prod_mat)
            K += T
            scaled_K += T / self.sigma[i]

        loss = (ww * K).sum()
        A = ww * scaled_K
        a = A.sum(axis=1)

        grad = 2.0 * (A[:n_pred,:].dot(X) - X[:n_pred,:] * a[:n_pred,np.newaxis])

        return loss, util.to_garray(grad)

    def get_name(self):
        return 'cpu_mmdgen_gaussian'

    def get_id(self):
        return 305

    def __repr__(self):
        return 'Loss <%s> w=%g, sigma=%s, scale_weight=%s' % (
                self.get_name(), self.weight, str(self.sigma), str(self.scale_weight))

ls.register_loss(CpuGaussianKernelMmdLoss())

class CpuSqrtGaussianKernelMmdLoss(CpuGaussianKernelMmdLoss):
    """
    k(x,y) = sqrt{exp(-|x-y|^2 / (2 sigma))}

    Multi-scale MMD loss with Gaussian kernels.  Essentially reimplementing 
    PairMmdLossMultiScale / UnsupervisedMmdLoss.
    """
    def __init__(self, sigma=[1.0], scale_weight=None, **kwargs):
        super(CpuSqrtGaussianKernelMmdLoss, self).__init__(sigma=sigma, 
                scale_weight=scale_weight, **kwargs)

    def compute_not_weighted_loss_and_grad(self, pred, compute_grad=False):
        loss, grad = super(CpuSqrtGaussianKernelMmdLoss, self).compute_not_weighted_loss_and_grad(pred, compute_grad=compute_grad)
        sqrt_loss = math.sqrt(loss)
        return sqrt_loss, grad / (2 * sqrt_loss + 1e-10)

    def get_name(self):
        return 'cpu_mmdgen_sqrt_gaussian'

    def get_id(self):
        return 306

    def __repr__(self):
        return 'Loss <%s> w=%g, sigma=%s, scale_weight=%s' % (
                self.get_name(), self.weight, str(self.sigma), str(self.scale_weight))

ls.register_loss(CpuSqrtGaussianKernelMmdLoss())

class CpuPerExampleSqrtGaussianKernelMmdLoss(ls.Loss):
    """
    Each batch contains multiple examples, MMD is applied on a per example basis.
    """
    def __init__(self, sigma=[1.0], scale_weight=None, pred_per_example=1, **kwargs):
        super(CpuPerExampleSqrtGaussianKernelMmdLoss, self).__init__(**kwargs)
        self.mmd_loss = CpuSqrtGaussianKernelMmdLoss(sigma=sigma, scale_weight=scale_weight, **kwargs)
        self.pred_per_example = pred_per_example

    def load_target(self, target, **kwargs):
        """
        target is the target data batch that we want our model to match.

        target is a list of target matrices, each correspond to the targets for
        one prediction/one group of predictions.
        """
        self.target = [util.to_nparray(t) if len(t.shape) > 1 else util.to_nparray(t)[np.newaxis,:] for t in target]

    def compute_not_weighted_loss_and_grad(self, pred, compute_grad=False):
        """
        Return loss and gradient
        """
        pred = util.to_nparray(pred)

        loss = 0
        grad = gnp.zeros(pred.shape)

        assert pred.shape[0] % self.pred_per_example == 0
        n_groups = pred.shape[0] / self.pred_per_example
        assert n_groups == len(self.target)

        for i_group in range(n_groups):
            i_start = i_group * self.pred_per_example
            i_end = i_start + self.pred_per_example

            self.mmd_loss.load_target(self.target[i_group])
            t_loss, grad[i_start:i_end] = self.mmd_loss.compute_not_weighted_loss_and_grad(pred[i_start:i_end], compute_grad=True)
            loss += t_loss

        return loss / n_groups, grad / n_groups

    def get_name(self):
        return 'cpu_per_example_mmdgen_sqrt_gaussian'

    def get_id(self):
        return 307

    def __repr__(self):
        return 'Loss <%s> w=%g, sigma=%s, scale_weight=%s, pred_per_example=%s' % (
                self.get_name(), self.mmd_loss.weight, str(self.mmd_loss.sigma), str(self.mmd_loss.scale_weight), str(self.pred_per_example))

ls.register_loss(CpuPerExampleSqrtGaussianKernelMmdLoss())

############################################################
# Learners, samplers, and others
############################################################

class StochasticGenerativeNet(nn.NeuralNet):
    """
    A generative feed-forward neural net with a layer of stochastic hidden 
    units at the top (or bottom depending on how you orient the network), and
    a deterministic top-down mapping given by the neural net.

    The hidden units are fixed to have a uniform distribution over the space
    [-1,1]^out_dim.
    """
    def __init__(self, in_dim=0, out_dim=0):
        super(StochasticGenerativeNet, self).__init__(in_dim, out_dim)

    def sample_hiddens(self, n_samples):
        """
        Generate specified number of samples of hidden units.
        """
        return 2 * gnp.rand(n_samples, self.in_dim) - 1
        # return gnp.randn(n_samples, self.in_dim)

    def generate_samples(self, z=None, n_samples=100, sample_batch_size=1000):
        """
        Generate samples of visibles units. The provided z will be used for
        propagating samples if given, otherwise new samples of z will be
        generated using sample_hiddens.
        """
        if z is not None:
            return self.forward_prop(z, add_noise=False, compute_loss=False)

        samples = gnp.zeros((n_samples, self.out_dim))
        n_batches = (n_samples + sample_batch_size - 1) / sample_batch_size

        for i_batch in range(n_batches):
            i_start = i_batch * sample_batch_size
            i_end = (i_start + sample_batch_size) if i_batch + 1 < n_batches else n_samples
            n_samples_in_batch = i_end - i_start

            z = self.sample_hiddens(n_samples_in_batch)
            samples[i_start:i_end] = self.forward_prop(z, add_noise=False, compute_loss=False)

        # return self.forward_prop(z, add_noise=False, compute_loss=False)
        return samples

class StochasticGenerativeNetWithAutoencoder(StochasticGenerativeNet):
    """
    A StochasticGenerativeNet together with an autoencoder. The stochastic 
    generative network is used in the code layer of the autoencoder.
    """
    def __init__(self, in_dim=0, out_dim=0, autoencoder=None):
        super(StochasticGenerativeNetWithAutoencoder, self).__init__(in_dim, out_dim)
        self.autoencoder = autoencoder

    def _generate_code_samples(self, z=None, n_samples=100, sample_batch_size=1000):
        return super(StochasticGenerativeNetWithAutoencoder, self).generate_samples(
                z=z, n_samples=n_samples, sample_batch_size=sample_batch_size)

    def generate_samples(self, z=None, n_samples=100, sample_batch_size=1000):
        return self.autoencoder.decoder.forward_prop(self._generate_code_samples(
                z=z, n_samples=n_samples, sample_batch_size=sample_batch_size))

    def load_target(self, target, *args, **kwargs):
        """
        Need to first transform target into the code space using encoder.
        """
        super(StochasticGenerativeNetWithAutoencoder, self).load_target(
                self.autoencoder.encode(target), *args, **kwargs)

class StochasticGenerativeNetWithAutoencoderContainer(object):
    """
    A container used to combine a net with an autoencoder after training - for
    generating samples.
    """
    def __init__(self, net, autoencoder):
        self.net = net
        self.autoencoder = autoencoder

    def generate_samples(self, z=None, n_samples=100, sample_batch_size=1000):
        return self.autoencoder.decoder.forward_prop(
                self.net.generate_samples(z, n_samples, sample_batch_size))

class SampleFilter(object):
    """
    Used to filter samples.
    """
    def __init__(self):
        pass

    def filter(self, x):
        """
        x: n x D is a matrix of examples

        Return a matrix n' x D, with n' <= n, such that it contains all 'good'
        samples.
        """
        raise NotImplementedError()

class BlankSampleFilter(SampleFilter):
    """
    Place holder for debugging, this class does nothing.
    """
    def filter(self, x):
        return x

class ClassifierSampleFilter(SampleFilter):
    """
    Applies a classifier to judge whether a sample is good.
    """
    def __init__(self, classifier, threshold, prev=None):
        """
        The classifier makes probabilistic predictions and has a function 
        predict_proba that outputs a prediction matrix with elements between 0
        and 1.  p[i][0] close to 1 indicates that a sample is good, close to 0
        indicates a sample is bad. p[i][1] should always be 1-p[i][0].

        prev: allows multiple filters to be chained together.
        """
        self.classifier = classifier
        self.threshold = threshold
        self.prev = prev

    def filter(self, x):
        if self.prev is not None:
            x = self.prev.filter(x)
        is_garray = isinstance(x, gnp.garray)
        if is_garray:
            x = x.asarray()
        p = self.classifier.predict_proba(x)
        idx = np.arange(p.shape[0])[p[:,0] > self.threshold]
        x = x[idx]
        if is_garray:
            x = gnp.garray(x)
        return x

class ClassifierSampleStochasticFilter(SampleFilter):
    """
    Same as above, but filter out samples probabilistically rather than
    deterministically using a hard threshold.
    """
    def __init__(self, classifier, prev=None):
        """
        The classifier should support probabilistic outputs.
        """
        self.classifier = classifier
        self.prev = prev

    def filter(self, x):
        if self.prev is not None:
            x = self.prev.filter(x)
        is_garray = isinstance(x, gnp.garray)
        if is_garray:
            x = x.asarray()
        p = self.classifier.predict_proba(x)

        # TODO: implement probabilistic filtering
        idx = np.arange(p.shape[0])[p[:,0] > np.random.rand(p.shape[0])]
        x = x[idx]
        if is_garray:
            x = gnp.garray(x)

        return x


class StochasticGenerativeNetWithFilter(object):
    """
    This is a class used purely for generating samples, it is required to have
    a method called generate_samples.

    StochasticGenerativeNet can be used as a subclass of this one.
    """
    def __init__(self, net, sample_filter):
        """
        net can be StochasticGenerativeNet, or StochasticGenerativeNetWithFilter,
        which allows multiple filtered nets to be chained together.
        """
        self.net = net
        self.sample_filter = sample_filter

    def generate_samples(self, z=None, n_samples=100):
        """
        Generate samples from the StochasticGenerativeNet and then filter out
        bad samples using the sample filter.
        """
        factor = 2
        x = self.sample_filter.filter(self.net.generate_samples(z, n_samples * factor))[:n_samples]
        gnp.free_reuse_cache()
        is_garray = isinstance(x, gnp.garray)
        while x.shape[0] < n_samples:
            # factor *= 2   # this will explode in high threshold settings
            y = self.sample_filter.filter(self.net.generate_samples(z, (n_samples - x.shape[0]) * factor))
            if is_garray:
                x = gnp.concatenate([x, y[:n_samples - x.shape[0]]], axis=0)
            else:
                x = np.r_[x, y[:n_samples - x.shape[0]]]
            gnp.free_reuse_cache()

        return x

class StochasticGenerativeNetLearner(learner.Learner):
    """
    Used for learning the StochasticGenerativeNet model.
    """
    def __init__(self, net):
        super(StochasticGenerativeNetLearner, self).__init__(net)
        self.n_samples_per_update = 100
        self.n_sample_update_iters = 1
        self.i_sample_update_iter = 0

        self.set_output_dir('.')

    def load_data(self, x_train):
        self.x_train = util.to_garray(x_train)

    def load_train_target(self):
        self.net.load_target(self.x_train)

    def sample_hiddens(self):
        self.z = self.net.sample_hiddens(self.n_samples_per_update)

    def f_and_fprime(self, w):
        self.net.set_param_from_vec(w)
        self.net.clear_gradient()
        if self.i_sample_update_iter % self.n_sample_update_iters == 0:
            self.sample_hiddens()
        self.i_sample_update_iter = (self.i_sample_update_iter + 1) % self.n_sample_update_iters
        self.net.forward_prop(self.z, add_noise=True, compute_loss=True)
        loss = self.net.get_loss() / self.z.shape[0]
        self.net.backward_prop()
        grad = self.net.get_grad_vec() / self.z.shape[0]
        return loss, grad

    def create_minibatch_generator(self, minibatch_size):
        self.minibatch_generator = learner.MiniBatchGenerator(
                self.x_train, minibatch_size=minibatch_size, random_order=True)

    def f_and_fprime_minibatch(self, w):
        self.net.set_param_from_vec(w)
        self.net.clear_gradient()

        if self.i_sample_update_iter % self.n_sample_update_iters == 0:
            if self.minibatch_load_target:
                x = self.minibatch_generator.next()
                self.net.load_target(x)
            self.sample_hiddens()

        self.i_sample_update_iter = (self.i_sample_update_iter + 1) % self.n_sample_update_iters

        self.net.forward_prop(self.z, add_noise=True, compute_loss=True)
        loss = self.net.get_loss() / self.z.shape[0]
        self.net.backward_prop()
        grad = self.net.get_grad_vec() / self.z.shape[0]

        return loss, grad

    def train_stochastic_lbfgs(self, **kwargs):
        self._prepare_for_training()
        if 'minibatch_size' in kwargs:
            minibatch_size = kwargs['minibatch_size']
            del kwargs['minibatch_size']
        else:
            minibatch_size = 100

        self.create_minibatch_generator(minibatch_size)
        self._process_options(kwargs)
        #self.print_options(kwargs)
        self.best_w, self.best_obj, d = spopt.fmin_l_bfgs_b(self.f_and_fprime_minibatch, self.init_w, **kwargs)
        self.best_grad = d['grad']
        return self.f_post_training()

    def f_info(self, w):
        """
        train_loss = None

        w_0 = self.net.get_param_vec()
        self.net.set_noiseless_param_from_vec(w)

        y = self.net.forward_prop(self.x_train, add_noise=False, compute_loss=True)
        train_loss = self.net.get_loss() / self.x_train.shape[0]
        train_acc = (self.t_train == y.argmax(axis=1)).mean()

        if self.use_validation:
            y = self.net.forward_prop(self.x_val, add_noise=False, compute_loss=False)
            val_acc = (self.t_val == y.argmax(axis=1)).mean()

            s = 'train loss %.4f, acc %.4f, val acc ' % (train_loss, train_acc)
            if self.best_obj is None or val_acc > self.best_obj:
                self.best_obj = val_acc 
                self.best_w = w.copy()
                s += co.good_colored_str('%.4f' % val_acc)
            else:
                s += '%.4f' % val_acc
        else:
            s = 'train loss %.4f, acc ' % train_loss
            if self.best_obj is None or train_acc < self.best_obj:
                self.best_obj = train_acc
                self.best_w = w.copy()
                s += co.good_colored_str('%.4f' % train_acc)
            else:
                s += '%.4f' % train_acc

        self.net.set_param_from_vec(w_0)
        return s
        """
        return '<place holder>'

    def _process_options(self, kwargs):
        if 'n_samples_per_update' in kwargs:
            self.n_samples_per_update = kwargs['n_samples_per_update']
            del kwargs['n_samples_per_update']
        if 'n_sample_update_iters' in kwargs:
            self.n_sample_update_iters = kwargs['n_sample_update_iters']
            del kwargs['n_sample_update_iters']

        self.i_sample_update_iter = 0

        if 'minibatch_size' in kwargs:
            minibatch_size = kwargs['minibatch_size']
            del kwargs['minibatch_size']
        else:
            minibatch_size = 100

        self.create_minibatch_generator(minibatch_size)

        self.minibatch_load_target = True
        if 'minibatch_load_target' in kwargs:
            self.minibatch_load_target = kwargs['minibatch_load_target']
            del kwargs['minibatch_load_target']

    def f_post_training(self):
        # self.net.set_param_from_vec(self.best_w)
        if hasattr(self, 'best_grad') and hasattr(self, 'best_obj'):
            return self.best_obj, self.best_grad

    def save_model(self):
        # self.net.save_model_to_file(self.output_dir + '/gen_%s.pdata' % (time.strftime('%Y%m%d_%H%M%S', time.localtime())))
        self.net.save_model_to_file(self.output_dir + '/gen_end.pdata')

    def save_checkpoint(self, label):
        self.net.save_model_to_file(self.output_dir + '/checkpoint_%s.pdata' % str(label))

class StochasticGenerativeNetLearnerAutoScale(learner.Learner):
    """
    Used for learning the StochasticGenerativeNet model with MMD loss.  The
    scale parameter will be automatically tuned.
    """
    def __init__(self, net):
        super(StochasticGenerativeNetLearnerAutoScale, self).__init__(net)
        self.n_samples_per_update = 100
        self.n_sample_update_iters = 1
        self.i_sample_update_iter = 0
        self.i_scale_update_iter = 0
        self.n_scale_update_iters = 0
        self.n_scale_update_samples = 2000
        self._scale_selection_range = np.logspace(0, 8, 30)

        self.set_output_dir('.')

    def load_data(self, x_train):
        self.x_train = util.to_garray(x_train)

    def load_train_target(self):
        self.net.load_target(self.x_train)

    def sample_hiddens(self):
        self.z = self.net.sample_hiddens(self.n_samples_per_update)

    def update_loss_scale(self):
        """
        Automatically set the scale of the loss.
        """
        n_data_samples = min(self.x_train.shape[0], self.n_scale_update_samples)
        data = self.x_train[np.random.permutation(self.x_train.shape[0])[:n_data_samples]]
        samples = self.net.generate_samples(n_samples=self.n_scale_update_samples)

        max_loss = 0
        max_sigma = 1
        for s in self._scale_selection_range:
            mmd = ls.get_loss_from_type_name(self.net.loss.get_name(), sigma=s, scale_weight=self.net.loss.scale_weight[0])
            mmd.load_target(data)
            loss = mmd.compute_not_weighted_loss_and_grad(samples, compute_grad=False)[0]
            if loss > max_loss:
                max_loss = loss
                max_sigma = s

        print '>>> Reset loss...'
        self.net.loss.sigma = [float(max_sigma)]
        self.net.loss.scale_weight = [float(self.net.loss.scale_weight[0])]
        print '>>>',
        print self.net.loss

    def f_and_fprime(self, w):
        self.net.set_param_from_vec(w)
        self.net.clear_gradient()

        # resample if necessary
        if self.i_sample_update_iter % self.n_sample_update_iters == 0:
            self.sample_hiddens()
        self.i_sample_update_iter = (self.i_sample_update_iter + 1) % self.n_sample_update_iters

        # update scale of the loss if necessary
        if self.n_scale_update_iters > 0:
            if self.i_scale_update_iter % self.n_scale_update_iters == 0:
                self.update_loss_scale()
            self.i_scale_update_iter = (self.i_scale_update_iter + 1) % self.n_scale_update_iters

        self.net.forward_prop(self.z, add_noise=True, compute_loss=True)
        loss = self.net.get_loss() / self.z.shape[0]
        self.net.backward_prop()
        grad = self.net.get_grad_vec() / self.z.shape[0]
        return loss, grad

    def create_minibatch_generator(self, minibatch_size):
        self.minibatch_generator = learner.MiniBatchGenerator(
                self.x_train, minibatch_size=minibatch_size, random_order=True)

    def f_and_fprime_minibatch(self, w):
        self.net.set_param_from_vec(w)
        self.net.clear_gradient()

        if self.i_sample_update_iter % self.n_sample_update_iters == 0:
            if self.minibatch_load_target:
                x = self.minibatch_generator.next()
                self.net.load_target(x)
            self.sample_hiddens()

        self.i_sample_update_iter = (self.i_sample_update_iter + 1) % self.n_sample_update_iters

        if self.n_scale_update_iters > 0:
            if self.i_scale_update_iter % self.n_scale_update_iters == 0:
                self.update_loss_scale()
            self.i_scale_update_iter = (self.i_scale_update_iter + 1) % self.n_scale_update_iters

        self.net.forward_prop(self.z, add_noise=True, compute_loss=True)
        loss = self.net.get_loss() / self.z.shape[0]
        self.net.backward_prop()
        grad = self.net.get_grad_vec() / self.z.shape[0]

        return loss, grad

    def train_stochastic_lbfgs(self, **kwargs):
        self._prepare_for_training()
        if 'minibatch_size' in kwargs:
            minibatch_size = kwargs['minibatch_size']
            del kwargs['minibatch_size']
        else:
            minibatch_size = 100

        self.create_minibatch_generator(minibatch_size)
        self._process_options(kwargs)
        #self.print_options(kwargs)
        self.best_w, self.best_obj, d = spopt.fmin_l_bfgs_b(self.f_and_fprime_minibatch, self.init_w, **kwargs)
        self.best_grad = d['grad']
        return self.f_post_training()

    def f_info(self, w):
        """
        train_loss = None

        w_0 = self.net.get_param_vec()
        self.net.set_noiseless_param_from_vec(w)

        y = self.net.forward_prop(self.x_train, add_noise=False, compute_loss=True)
        train_loss = self.net.get_loss() / self.x_train.shape[0]
        train_acc = (self.t_train == y.argmax(axis=1)).mean()

        if self.use_validation:
            y = self.net.forward_prop(self.x_val, add_noise=False, compute_loss=False)
            val_acc = (self.t_val == y.argmax(axis=1)).mean()

            s = 'train loss %.4f, acc %.4f, val acc ' % (train_loss, train_acc)
            if self.best_obj is None or val_acc > self.best_obj:
                self.best_obj = val_acc 
                self.best_w = w.copy()
                s += co.good_colored_str('%.4f' % val_acc)
            else:
                s += '%.4f' % val_acc
        else:
            s = 'train loss %.4f, acc ' % train_loss
            if self.best_obj is None or train_acc < self.best_obj:
                self.best_obj = train_acc
                self.best_w = w.copy()
                s += co.good_colored_str('%.4f' % train_acc)
            else:
                s += '%.4f' % train_acc

        self.net.set_param_from_vec(w_0)
        return s
        """
        return '<place holder>'

    def _process_options(self, kwargs):
        if 'n_samples_per_update' in kwargs:
            self.n_samples_per_update = kwargs['n_samples_per_update']
            del kwargs['n_samples_per_update']
        if 'n_sample_update_iters' in kwargs:
            self.n_sample_update_iters = kwargs['n_sample_update_iters']
            del kwargs['n_sample_update_iters']

        self.i_sample_update_iter = 0

        if 'minibatch_size' in kwargs:
            minibatch_size = kwargs['minibatch_size']
            del kwargs['minibatch_size']
        else:
            minibatch_size = 100

        self.create_minibatch_generator(minibatch_size)

        self.minibatch_load_target = True
        if 'minibatch_load_target' in kwargs:
            self.minibatch_load_target = kwargs['minibatch_load_target']
            del kwargs['minibatch_load_target']

        if 'i_scale_update' in kwargs:
            self.n_scale_update_iters = kwargs['i_scale_update']
            del kwargs['i_scale_update']
        else:
            self.n_scale_update_iters = 0

        if 'n_scale_update_samples' in kwargs:
            self.n_scale_update_samples = kwargs['n_scale_update_samples']
            del kwargs['n_scale_update_samples']
        else:
            self.n_scale_update_samples = 2000

    def f_post_training(self):
        # self.net.set_param_from_vec(self.best_w)
        if hasattr(self, 'best_grad') and hasattr(self, 'best_obj'):
            return self.best_obj, self.best_grad

    def save_model(self):
        # self.net.save_model_to_file(self.output_dir + '/gen_%s.pdata' % (time.strftime('%Y%m%d_%H%M%S', time.localtime())))
        self.net.save_model_to_file(self.output_dir + '/gen_end.pdata')

    def save_checkpoint(self, label):
        self.net.save_model_to_file(self.output_dir + '/checkpoint_%s.pdata' % str(label))




