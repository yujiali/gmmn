"""
Debug tests for the datasetbias project.

Yujia Li, 09/2014
"""
import os
os.environ['GNUMPY_CPU_PRECISION'] = '64'

import pynn.nn as nn
import pynn.layer as ly
import pynn.loss as ls
import gnumpy as gnp
import numpy as np
import time
import math

import core.generative as gen

_GRAD_CHECK_EPS = 1e-6
_FDIFF_EPS = 1e-8

_TEMP_FILE_NAME = '_temp_.pdata'

_GOOD_COLOR_BEGINS = '\033[42m'
_BAD_COLOR_BEGINS = '\033[41m'
_COLOR_RESET = '\033[0m'

def good_colored_str(txt):
    return _GOOD_COLOR_BEGINS + txt + _COLOR_RESET

def bad_colored_str(txt):
    return _BAD_COLOR_BEGINS + txt + _COLOR_RESET

def vec_str(v):
    s = '[ '
    for i in range(len(v)):
        s += '%11.8f ' % v[i]
    s += ']'
    return s

def test_vec_pair(v1, msg1, v2, msg2, error_thres=_GRAD_CHECK_EPS):
    print msg1 + ' : ' + vec_str(v1)
    print msg2 + ' : ' + vec_str(v2)
    n_space = len(msg2) - len('diff')
    print ' ' * n_space + 'diff' + ' : ' + vec_str(v1 - v2)
    err = np.sqrt(((v1 - v2)**2).sum())
    print 'err : %.8f' % err

    success = err < error_thres
    print good_colored_str('** SUCCESS **') if success else \
            bad_colored_str('** FAIL **')

    return success

def finite_difference_gradient(f, x):
    grad = x * 0
    for i in range(len(x)):
        x_0 = x[i]
        x[i] = x_0 + _FDIFF_EPS
        f_plus = f(x)
        x[i] = x_0 - _FDIFF_EPS
        f_minus = f(x)
        grad[i] = (f_plus - f_minus) / (2 * _FDIFF_EPS)
        x[i] = x_0

    return grad

def fdiff_grad_generator(net, x, t, add_noise=False, seed=None):
    if t is not None:
        net.load_target(t)

    def f(w):
        if add_noise and seed is not None:
            gnp.seed_rand(seed)
        w_0 = net.get_param_vec()
        net.set_param_from_vec(w)
        net.forward_prop(x, add_noise=add_noise, compute_loss=True)
        loss = net.get_loss()
        net.set_param_from_vec(w_0)

        return loss

    return f

def test_net_io(f_create, f_create_void):
    net1 = f_create()
    print 'Testing %s I/O' % net1.__class__.__name__

    net1.save_model_to_file(_TEMP_FILE_NAME)

    net2 = f_create_void()
    net2.load_model_from_file(_TEMP_FILE_NAME)

    os.remove(_TEMP_FILE_NAME)

    print 'Net #1: \n' + str(net1)
    print 'Net #2: \n' + str(net2)
    test_passed = (str(net1) == str(net2))

    test_passed = test_passed and test_vec_pair(net1.get_param_vec(), 'Net #1',
            net2.get_param_vec(), 'Net #2')
    return test_passed

def test_databias_loss(loss_type, **kwargs):
    print 'Testing Loss <' + loss_type + '> ' \
            + ', '.join([str(k) + '=' + str(v) for k, v in kwargs.iteritems()])

    n_cases = 5
    n_datasets = 3
    in_dim = 2
    
    x = gnp.randn(n_cases, in_dim)
    s = np.arange(n_cases) % n_datasets

    loss = ls.get_loss_from_type_name(loss_type)
    loss.load_target(s, K=n_datasets, **kwargs)

    def f(w):
        return loss.compute_loss_and_grad(w.reshape(x.shape), compute_grad=True)[0]

    backprop_grad = loss.compute_loss_and_grad(x, compute_grad=True)[1].asarray().ravel()
    fdiff_grad = finite_difference_gradient(f, x.asarray().ravel())

    test_passed = test_vec_pair(fdiff_grad, 'Finite Difference Gradient',
            backprop_grad, '  Backpropagation Gradient')
    print ''
    return test_passed

def create_databias_net(dropout_rate):
    net = nn.NeuralNet(3, 2)
    net.add_layer(2, nonlin_type=ly.NONLIN_NAME_TANH, dropout=dropout_rate)
    net.add_layer(0, nonlin_type=ly.NONLIN_NAME_LINEAR, dropout=0)
    return net

def test_databias_loss_with_net(add_noise, loss_type, **kwargs):
    print 'Testing Loss <' + loss_type + '> with network, '\
            + ('with noise' if add_noise else 'without noise') + ', ' \
            + ', '.join([str(k) + '=' + str(v) for k, v in kwargs.iteritems()])
    n_cases = 5
    n_datasets = 3
    seed = 8
    dropout_rate = 0.5 if add_noise else 0

    net = create_databias_net(dropout_rate)
    net.set_loss(loss_type)
    print net
    x = gnp.randn(n_cases, net.in_dim)
    s = np.arange(n_cases) % n_datasets

    net.load_target(s, K=n_datasets, **kwargs)

    if add_noise:
        gnp.seed_rand(seed)
    net.clear_gradient()
    net.forward_prop(x, add_noise=add_noise, compute_loss=True)
    net.backward_prop()

    backprop_grad = net.get_grad_vec()

    f = fdiff_grad_generator(net, x, None, add_noise=add_noise, seed=seed)
    fdiff_grad = finite_difference_gradient(f, net.get_param_vec())

    test_passed = test_vec_pair(fdiff_grad, 'Finite Difference Gradient',
            backprop_grad, '  Backpropagation Gradient')
    print ''

    gnp.seed_rand(int(time.time()))
    return test_passed

def test_generative_mmd_loss(sigma=1):
    print 'Testing generative MMD loss, sigma=%g' % sigma
    n_dims = 3
    n_target = 5
    n_pred = 4

    target = gnp.randn(n_target, n_dims)
    pred = gnp.randn(n_pred, n_dims)

    mmd = ls.get_loss_from_type_name(ls.LOSS_NAME_MMDGEN, sigma=sigma)
    mmd.load_target(target)

    def f(w):
        return mmd.compute_loss_and_grad(w.reshape(pred.shape), compute_grad=False)[0]

    backprop_grad = mmd.compute_loss_and_grad(pred, compute_grad=True)[1].asarray().ravel()
    fdiff_grad = finite_difference_gradient(f, pred.asarray().ravel())

    test_passed = test_vec_pair(fdiff_grad, 'Finite Difference Gradient',
            backprop_grad, '  Backpropagation Gradient')
    print ''
    return test_passed

def test_generative_multi_scale_mmd_loss(sigma=[1, 10], scale_weight=None):
    print 'Testing generative multi-scale MMD loss, sigma=%s' % str(sigma)
    n_dims = 3
    n_target = 5
    n_pred = 4

    target = gnp.randn(n_target, n_dims)
    pred = gnp.randn(n_pred, n_dims)

    mmd = ls.get_loss_from_type_name(ls.LOSS_NAME_MMDGEN_MULTISCALE, sigma=sigma, scale_weight=scale_weight)
    mmd.load_target(target)

    def f(w):
        return mmd.compute_loss_and_grad(w.reshape(pred.shape), compute_grad=False)[0]

    backprop_grad = mmd.compute_loss_and_grad(pred, compute_grad=True)[1].asarray().ravel()
    fdiff_grad = finite_difference_gradient(f, pred.asarray().ravel())

    test_passed = test_vec_pair(fdiff_grad, 'Finite Difference Gradient',
            backprop_grad, '  Backpropagation Gradient')
    print ''
    return test_passed

def test_linear_time_mmd_loss(sigma=1.0, use_modified_loss=False, use_absolute_value=False):
    print 'Testing linear time MMD loss, sigma=%s' % str(sigma)
    n_dims = 3
    n_target = 4
    n_pred = 4

    target = gnp.randn(n_target, n_dims)
    pred = gnp.randn(n_pred, n_dims)

    mmd = ls.get_loss_from_type_name(ls.LOSS_NAME_LINEAR_TIME_MMDGEN, sigma=sigma,
            use_modified_loss=use_modified_loss, use_absolute_value=use_absolute_value)
    mmd.load_target(target)

    def f(w):
        return mmd.compute_loss_and_grad(w.reshape(pred.shape), compute_grad=False)[0]

    backprop_grad = mmd.compute_loss_and_grad(pred, compute_grad=True)[1].asarray().ravel()
    fdiff_grad = finite_difference_gradient(f, pred.asarray().ravel())

    test_passed = test_vec_pair(fdiff_grad, 'Finite Difference Gradient',
            backprop_grad, '  Backpropagation Gradient')
    print ''
    return test_passed

def test_linear_time_minibatch_mmd_loss(sigma=1.0, minibatch_size=100):
    print 'Testing linear time minibatch MMD loss'
    n_dims = 3
    n_target = 10
    n_pred = 10

    target = gnp.randn(n_target, n_dims)
    pred = gnp.randn(n_pred, n_dims)

    mmd = ls.get_loss_from_type_name(ls.LOSS_NAME_LINEAR_TIME_MINIBATCH_MMDGEN,
            sigma=sigma, minibatch_size=minibatch_size)
    mmd.load_target(target)
    print mmd

    def f(w):
        return mmd.compute_loss_and_grad(w.reshape(pred.shape), compute_grad=False)[0]

    backprop_grad = mmd.compute_loss_and_grad(pred, compute_grad=True)[1].asarray().ravel()
    fdiff_grad = finite_difference_gradient(f, pred.asarray().ravel())

    test_passed = test_vec_pair(fdiff_grad, 'Finite Difference Gradient',
            backprop_grad, '  Backpropagation Gradient')
    print ''
    return test_passed

def test_random_feature_mmd_loss(sigma=[1,10], scale_weight=[0.5, 1], n_features=3):
    print 'Testing random feature MMD loss'
    n_dims = 2
    n_target = 5
    n_pred = 5 

    target = gnp.randn(n_target, n_dims)
    pred = gnp.randn(n_pred, n_dims)

    mmd = ls.get_loss_from_type_name(ls.LOSS_NAME_RANDOM_FEATURE_MMDGEN,
            sigma=sigma, scale_weight=scale_weight, n_features=n_features)
    mmd.load_target(target)
    print mmd

    def f(w):
        return mmd.compute_loss_and_grad(w.reshape(pred.shape), compute_grad=False)[0]

    backprop_grad = mmd.compute_loss_and_grad(pred, compute_grad=True)[1].asarray().ravel()
    fdiff_grad = finite_difference_gradient(f, pred.asarray().ravel())

    test_passed = test_vec_pair(fdiff_grad, 'Finite Difference Gradient',
            backprop_grad, '  Backpropagation Gradient')
    print ''
    return test_passed

def test_random_feature_mmd_loss_approximation(sigma=[1,10], scale_weight=[0.5,1],
        n_features=3):
    print 'Testing random feature MMD loss approximation error'

    n_dims = 2
    n_target = 5
    n_pred = 5 

    target = gnp.rand(n_target, n_dims)
    pred = gnp.rand(n_pred, n_dims)

    rand_mmd = ls.get_loss_from_type_name(ls.LOSS_NAME_RANDOM_FEATURE_MMDGEN,
            sigma=sigma, scale_weight=scale_weight, n_features=n_features)
    rand_mmd.load_target(target)
    print rand_mmd

    mmd = ls.get_loss_from_type_name(ls.LOSS_NAME_MMDGEN_MULTISCALE_PAIR,
            sigma=sigma, scale_weight=scale_weight)
    mmd.load_target(target)

    rand_loss, rand_grad = rand_mmd.compute_loss_and_grad(pred, compute_grad=True)
    true_loss, true_grad = mmd.compute_loss_and_grad(pred, compute_grad=True)

    test_passed = test_vec_pair(rand_grad.asarray().ravel(), 'Approximate Gradient',
            true_grad.asarray().ravel(), '       True Gradient', error_thres=1e-2)
    test_passed = test_vec_pair(np.array([rand_loss]), 'Approximate Loss',
            np.array([true_loss]), '       True Loss', error_thres=1e-2) \
            and test_passed
    print ''
    return test_passed

def test_pair_mmd_loss_multiscale(sigma=[1, 10], scale_weight=None):
    print 'Testing generative pair multi-scale MMD loss'
    n_dims = 3
    n_target = 5
    n_pred = 4

    target = gnp.randn(n_target, n_dims)
    pred = gnp.randn(n_pred, n_dims)

    mmd = ls.get_loss_from_type_name(ls.LOSS_NAME_MMDGEN_MULTISCALE_PAIR, sigma=sigma, scale_weight=scale_weight)
    mmd.load_target(target)
    print mmd

    def f(w):
        return mmd.compute_loss_and_grad(w.reshape(pred.shape), compute_grad=False)[0]

    backprop_grad = mmd.compute_loss_and_grad(pred, compute_grad=True)[1].asarray().ravel()
    fdiff_grad = finite_difference_gradient(f, pred.asarray().ravel())

    test_passed = test_vec_pair(fdiff_grad, 'Finite Difference Gradient',
            backprop_grad, '  Backpropagation Gradient')
    print ''
    return test_passed

def test_diff_kernel_mmd_loss(sigma=[1], scale_weight=[1], loss_name=None):
    assert loss_name is not None

    print 'Testing differentiable kernel MMD loss <%s>' % loss_name

    n_dims = 3
    n_target = 5
    n_pred = 4

    target = gnp.randn(n_target, n_dims)
    pred = gnp.randn(n_pred, n_dims)

    mmd = ls.get_loss_from_type_name(loss_name, sigma=sigma, scale_weight=scale_weight)
    mmd.load_target(target)
    print mmd

    def f(w):
        return mmd.compute_loss_and_grad(w.reshape(pred.shape), compute_grad=False)[0]

    backprop_grad = mmd.compute_loss_and_grad(pred, compute_grad=True)[1].asarray().ravel()
    fdiff_grad = finite_difference_gradient(f, pred.asarray().ravel())

    test_passed = test_vec_pair(fdiff_grad, 'Finite Difference Gradient',
            backprop_grad, '  Backpropagation Gradient')
    print ''
    return test_passed

def test_diff_kernel_per_example_mmd_loss(sigma=[1], scale_weight=[1], pred_per_example=1, target_per_example=[1], loss_name=None):
    assert loss_name is not None

    print 'Testing differentiable kernel per example MMD loss <%s>' % loss_name

    if len(target_per_example) == 1:
        target_per_example = target_per_example * 3

    n_dims = 3
    n_target = sum(target_per_example)
    n_pred = len(target_per_example) * pred_per_example

    pred = gnp.randn(n_pred, n_dims)
    target = []
    for i_target in target_per_example:
        target.append(gnp.randn(i_target, n_dims))

    mmd = ls.get_loss_from_type_name(loss_name, sigma=sigma, scale_weight=scale_weight, pred_per_example=pred_per_example)
    mmd.load_target(target)
    print mmd

    def f(w):
        return mmd.compute_loss_and_grad(w.reshape(pred.shape), compute_grad=False)[0]

    backprop_grad = mmd.compute_loss_and_grad(pred, compute_grad=True)[1].asarray().ravel()
    fdiff_grad = finite_difference_gradient(f, pred.asarray().ravel())

    test_passed = test_vec_pair(fdiff_grad, 'Finite Difference Gradient',
            backprop_grad, '  Backpropagation Gradient')
    print ''
    return test_passed

def test_all_diff_kernel_per_example_mmd_loss():
    print ''
    print '==============================================================='
    print 'Testing differentiable kernel per example MMD loss (new design)'
    print '==============================================================='
    print ''

    sigma_list = [1, 10]
    scale_weight_list = [1.0, 3.0]
    target_per_example_list = [[1], [2], [1,2,3]]
    pred_per_example_list = [1,2,3]
    loss_list = [ls.LOSS_NAME_CPU_PER_EXAMPLE_MMDGEN_SQRT_GAUSSIAN]

    n_success = 0
    n_tests = 0
    for loss_name in loss_list:
        for sigma, scale_weight, target_per_example, pred_per_example in zip(sigma_list, scale_weight_list,
                target_per_example_list[:len(sigma_list)], pred_per_example_list[:len(sigma_list)]):
            if test_diff_kernel_per_example_mmd_loss([sigma], [scale_weight], pred_per_example, target_per_example, loss_name):
                n_success += 1
            n_tests += 1

        if test_diff_kernel_per_example_mmd_loss(sigma_list, scale_weight_list, pred_per_example_list[-1], target_per_example_list[-1], loss_name):
            n_success += 1

        n_tests += 1

    print '=============='
    print 'Test finished: %d/%d success, %d failed' % (n_success, n_tests, n_tests - n_success)
    print ''

    return n_success, n_tests



def test_all_diff_kernel_mmd_loss():
    print ''
    print '==================================================='
    print 'Testing differentiable kernel MMD loss (new design)'
    print '==================================================='
    print ''

    sigma_list = [1, 2.5, 10]
    scale_weight_list = [1.0, 2, 3.0]
    loss_list = [ls.LOSS_NAME_MMDGEN_GAUSSIAN, ls.LOSS_NAME_MMDGEN_LAPLACIAN,
            ls.LOSS_NAME_MMDGEN_LAPLACIAN_L1, ls.LOSS_NAME_MMDGEN_SQRT_GAUSSIAN,
            ls.LOSS_NAME_CPU_MMDGEN_GAUSSIAN, ls.LOSS_NAME_CPU_MMDGEN_SQRT_GAUSSIAN]

    n_success = 0
    n_tests = 0
    for loss_name in loss_list:
        for sigma, scale_weight in zip(sigma_list, scale_weight_list):
            if test_diff_kernel_mmd_loss([sigma], [scale_weight], loss_name):
                n_success += 1
            n_tests += 1

        if test_diff_kernel_mmd_loss(sigma_list, scale_weight_list, loss_name):
            n_success += 1
        n_tests += 1

    print '=============='
    print 'Test finished: %d/%d success, %d failed' % (n_success, n_tests, n_tests - n_success)
    print ''

    return n_success, n_tests

def test_all_generative_mmd_loss():
    print ''
    print '========================'
    print 'Testing data bias losses'
    print '========================'
    print ''

    n_success = 0
    if test_generative_mmd_loss(sigma=1):
        n_success += 1
    if test_generative_mmd_loss(sigma=1e-1):
        n_success += 1
    if test_generative_multi_scale_mmd_loss(sigma=[1], scale_weight=[1.0]):
        n_success += 1
    if test_generative_multi_scale_mmd_loss(sigma=[10], scale_weight=[2.0]):
        n_success += 1
    if test_generative_multi_scale_mmd_loss(sigma=[100], scale_weight=[2.0]):
        n_success += 1
    if test_generative_multi_scale_mmd_loss(sigma=[1, 10, 100], scale_weight=[1.0, 2.0, 3.0]):
        n_success += 1
    if test_linear_time_mmd_loss(sigma=1):
        n_success += 1
    if test_linear_time_mmd_loss(sigma=0.1):
        n_success += 1
    if test_linear_time_mmd_loss(sigma=1, use_modified_loss=True):
        n_success += 1
    if test_linear_time_mmd_loss(sigma=0.1, use_modified_loss=True):
        n_success += 1
    if test_linear_time_mmd_loss(sigma=1, use_modified_loss=True, use_absolute_value=True):
        n_success += 1
    if test_linear_time_mmd_loss(sigma=0.1, use_modified_loss=True, use_absolute_value=True):
        n_success += 1
    if test_linear_time_minibatch_mmd_loss(sigma=1.0, minibatch_size=2):
        n_success += 1
    if test_linear_time_minibatch_mmd_loss(sigma=0.1, minibatch_size=3):
        n_success += 1
    if test_pair_mmd_loss_multiscale(sigma=[1], scale_weight=[1.0]):
        n_success += 1
    if test_pair_mmd_loss_multiscale(sigma=[10], scale_weight=[2.0]):
        n_success += 1
    if test_pair_mmd_loss_multiscale(sigma=[100], scale_weight=[2.0]):
        n_success += 1
    if test_pair_mmd_loss_multiscale(sigma=[1, 10, 100], scale_weight=[1.0, 2.0, 3.0]):
        n_success += 1
    if test_random_feature_mmd_loss(sigma=[1], scale_weight=[1.0], n_features=3):
        n_success += 1
    if test_random_feature_mmd_loss(sigma=[1], scale_weight=[1.0], n_features=10):
        n_success += 1
    if test_random_feature_mmd_loss(sigma=[1, 10, 100], scale_weight=[1.0, 2.0, 3.0], n_features=3):
        n_success += 1
    if test_random_feature_mmd_loss(sigma=[1, 10, 100], scale_weight=[1.0, 2.0, 3.0], n_features=10):
        n_success += 1
    if test_random_feature_mmd_loss_approximation(sigma=[5], scale_weight=[1.0], n_features=1024):
        n_success += 1
    if test_random_feature_mmd_loss_approximation(sigma=[5, 10, 80], scale_weight=[1.0, 2.0, 3.0], n_features=1024):
        n_success += 1

    n_tests = 24 

    print '=============='
    print 'Test finished: %d/%d success, %d failed' % (n_success, n_tests, n_tests - n_success)
    print ''

    return n_success, n_tests

def run_all_tests():
    gnp.seed_rand(int(time.time()))

    n_success = 0
    n_tests = 0

    test_list = [test_all_generative_mmd_loss,
            test_all_diff_kernel_mmd_loss, 
            test_all_diff_kernel_per_example_mmd_loss]
    for batch_test in test_list:
        success_in_batch, tests_in_batch = batch_test()
        n_success += success_in_batch
        n_tests += tests_in_batch

    print ''
    print '==================='
    print 'All tests finished: %d/%d success, %d failed' % (n_success, n_tests, n_tests - n_success)
    print ''

if __name__ == '__main__':
    run_all_tests()

