"""
Training script for MNIST/TFD.

Yujia Li, 01/2015
"""

import argparse

import cPickle as pickle
import pynn.nn as nn
import pynn.layer as ly
import pynn.loss as ls
import pynn.learner as learner
import core.generative as gen
import gnumpy as gnp
import numpy as np
import time
import dataio.mnist as mnistio
import dataio.tfd as tfd

import eval_mmd_generative_model as ev

# You may want to change this
OUTPUT_BASE_DIR = 'output'

def write_config(file_name, config):
    """
    file_name: output config file name
    config: dict containing all the configs.
    """
    with open(file_name, 'w') as f:
        for k, v in sorted(config.items(), key=lambda t: t[0]):
            f.write(str(k) + '=' + str(v) + '\n')

def cat_list(lst):
    return '_'.join([str(v) for v in lst])


def load_tfd_fold(fold=0):
    """
    Return train, val, test data for the particular fold.
    """
    # note that the training set used here is the 'unlabeled' set in TFD
    x_train, _, _ = tfd.load_proper_fold(fold, 'unlabeled', scale=True)
    x_val,   _, _ = tfd.load_proper_fold(fold, 'val', scale=True)
    x_test,  _, _ = tfd.load_proper_fold(fold, 'test', scale=True)

    imsz = np.prod(x_train.shape[1:])

    return x_train.reshape(x_train.shape[0], imsz), \
            x_val.reshape(x_val.shape[0], imsz), \
            x_test.reshape(x_test.shape[0], imsz)

def load_tfd_all_folds(set_name='val', n_folds=5):
    x = []
    for i_fold in range(n_folds):
        #xx, _, _ = tfd.load_fold(i_fold, set_name, scale=True)
        xx, _, _ = tfd.load_proper_fold(i_fold, set_name, scale=True)
        x.append(xx.reshape(xx.shape[0], np.prod(xx.shape[1:])))
    return x

def mnist_mmd_input_space(n_hids=[10,64,256,256,1024], sigma=[2,5,10,20,40,80], learn_rate=2, momentum=0.9):
    """
    n_hids: number of hidden units on all layers (top-down) in the generative network.
    sigma: a list of scales used for the kernel
    learn_rate, momentum: parameters for the learning process

    return: KDE log_likelihood on validation set.
    """
    gnp.seed_rand(8)

    x_train, x_val, x_test = mnistio.load_data()

    print ''
    print 'Training data: %d x %d' % x_train.shape

    in_dim = n_hids[0]
    out_dim = x_train.shape[1]

    net = gen.StochasticGenerativeNet(in_dim, out_dim)
    for i in range(1, len(n_hids)):
        net.add_layer(n_hids[i], nonlin_type=ly.NONLIN_NAME_RELU, dropout=0)
    net.add_layer(0, nonlin_type=ly.NONLIN_NAME_SIGMOID, dropout=0)

    # place holder loss
    net.set_loss(ls.LOSS_NAME_MMDGEN, loss_after_nonlin=True, sigma=80, loss_weight=1000)

    print ''
    print '========'
    print 'Training'
    print '========'
    print ''
    print net
    print ''

    mmd_learner = gen.StochasticGenerativeNetLearner(net)
    mmd_learner.load_data(x_train)

    output_base = OUTPUT_BASE_DIR + '/mnist/input_space'

    #sigma = [2,5,10,20,40,80]
    sigma_weights = [1,1,1,1,1,1]
    #learn_rate = 1
    #momentum = 0.9

    minibatch_size = 1000
    n_sample_update_iters = 1
    max_iters = 40000
    i_checkpoint = 2000

    output_dir = output_base + '/nhids_%s_sigma_%s_lr_%s_m_%s' % (
            '_'.join([str(nh) for nh in n_hids]), '_'.join([str(s) for s in sigma]), str(learn_rate), str(momentum))

    print ''
    print '>>>> output_dir = %s' % output_dir
    print ''

    mmd_learner.set_output_dir(output_dir)
    #net.set_loss(ls.LOSS_NAME_MMDGEN_MULTISCALE, loss_after_nonlin=True, sigma=sigma, scale_weight=sigma_weights, loss_weight=1000)
    net.set_loss(ls.LOSS_NAME_MMDGEN_SQRT_GAUSSIAN, loss_after_nonlin=True, sigma=sigma, scale_weight=sigma_weights, loss_weight=1000)

    print '**********************************'
    print net.loss
    print '**********************************'
    print ''

    def f_checkpoint(i_iter, w):
        mmd_learner.save_checkpoint('%d' % i_iter)

    mmd_learner.train_sgd(minibatch_size=minibatch_size, n_samples_per_update=minibatch_size, 
            n_sample_update_iters=n_sample_update_iters, learn_rate=learn_rate, momentum=momentum, 
            weight_decay=0, learn_rate_schedule={10000:learn_rate/10.0},
            momentum_schedule={10000:1-(1-momentum)/10.0},
            learn_rate_drop_iters=0, decrease_type='linear', adagrad_start_iter=0,
            max_iters=max_iters, iprint=100, i_exe=i_checkpoint, f_exe=f_checkpoint)

    mmd_learner.save_model()

    print ''
    print '===================='
    print 'Evaluating the model'
    print '===================='
    print ''

    log_prob, std, sigma = ev.kde_eval_mnist(net, x_val, verbose=False)
    test_log_prob, test_std, _ = ev.kde_eval_mnist(net, x_test, sigma_range=[sigma], verbose=False)

    print 'Validation: %.2f (%.2f)' % (log_prob, std)
    print 'Test      : %.2f (%.2f)' % (test_log_prob, test_std)
    print ''

    write_config(output_dir + '/params_and_results.cfg', { 'n_hids': n_hids,
        'sigma': sigma, 'sigma_weights': sigma_weights, 'learn_rate': learn_rate,
        'momentum': momentum, 'minibatch_size': minibatch_size, 
        'n_sample_update_iters': n_sample_update_iters, 'max_iters': max_iters,
        'i_checkpoint': i_checkpoint, 'val_log_prob': log_prob, 'val_std': std, 
        'test_log_prob': test_log_prob, 'test_std': test_std })

    print '>>>> output_dir = %s' % output_dir
    print ''

    return log_prob


def mnist_mmd_code_space(
        ae_n_hids=[1024, 32], 
        ae_dropout=[0.2, 0.5],
        ae_learn_rate=1e-1, 
        ae_momentum=0.9,
        mmd_n_hids=[10, 64, 256, 256, 1024], 
        mmd_sigma=1,
        mmd_learn_rate=2,
        mmd_momentum=0.9):
    """
    ae_n_hids: #hid for the encoder, bottom-up
    ae_dropout: the amount of dropout for each layer in the encoder, same order
    ae_learn_rate, ae_momentum: .
    mmd_n_hids: #hid for the generative net, top-down
    mmd_sigma: scale of the kernel
    mmd_learn_rate, mmd_momentum: .

    Return KDE log_likelihood on the validation set.
    """
    gnp.seed_rand(8)
    x_train, x_val, x_test = mnistio.load_data()

    common_output_base = OUTPUT_BASE_DIR + '/mnist/code_space'
    output_base = common_output_base + '/aeh_%s_dr_%s_aelr_%s_aem_%s_nh_%s_s_%s_lr_%s_m_%s' % (
            cat_list(ae_n_hids), cat_list(ae_dropout), str(ae_learn_rate), str(ae_momentum),
            cat_list(mmd_n_hids), str(mmd_sigma), str(mmd_learn_rate), str(mmd_momentum))

    #######################
    # Auto-encoder training
    #######################

    n_dims = x_train.shape[1]
    h_dim = ae_n_hids[-1]

    encoder = nn.NeuralNet(n_dims, h_dim)
    for i in range(len(ae_n_hids) - 1):
        encoder.add_layer(ae_n_hids[i], nonlin_type=ly.NONLIN_NAME_SIGMOID, dropout=ae_dropout[i])
    encoder.add_layer(0, nonlin_type=ly.NONLIN_NAME_SIGMOID, dropout=ae_dropout[-1])

    decoder = nn.NeuralNet(h_dim, n_dims)
    for i in range(len(ae_n_hids) - 1)[::-1]:
        decoder.add_layer(ae_n_hids[i], nonlin_type=ly.NONLIN_NAME_SIGMOID, dropout=0)
    decoder.add_layer(0, nonlin_type=ly.NONLIN_NAME_SIGMOID, dropout=0)
    decoder.set_loss(ls.LOSS_NAME_BINARY_CROSSENTROPY, loss_weight=1)

    autoenc = nn.AutoEncoder(encoder=encoder, decoder=decoder)

    print ''
    print autoenc
    print ''

    learn_rate = ae_learn_rate
    final_momentum = ae_momentum
    max_iters = 3000

    nn_pretrainer = learner.AutoEncoderPretrainer(autoenc)
    nn_pretrainer.load_data(x_train)
    nn_pretrainer.pretrain_network(learn_rate=1e-1, momentum=0.5, weight_decay=0, minibatch_size=100,
            max_grad_norm=10, max_iters=max_iters, iprint=100)

    nn_learner = learner.Learner(autoenc)
    nn_learner.set_output_dir(output_base + '/ae')
    nn_learner.load_data(x_train, x_train)

    def f_checkpoint(i_iter, w):
        nn_learner.save_checkpoint('%d' % i_iter)

    nn_learner.train_sgd(learn_rate=learn_rate, momentum=0, weight_decay=0, minibatch_size=100,
            learn_rate_schedule=None, momentum_schedule={50:0.5, 200:final_momentum}, 
            max_grad_norm=10, learn_rate_drop_iters=0, decrease_type='linear', adagrad_start_iter=0,
            max_iters=max_iters, iprint=100, i_exe=2000, f_exe=f_checkpoint)
    nn_learner.save_checkpoint('best')

    ##################
    # Training MMD net
    ##################

    n_hids = mmd_n_hids

    in_dim = n_hids[0]
    out_dim = autoenc.encoder.out_dim

    net = gen.StochasticGenerativeNetWithAutoencoder(in_dim, out_dim, autoenc)
    for i in range(1, len(n_hids)):
        net.add_layer(n_hids[i], nonlin_type=ly.NONLIN_NAME_RELU, dropout=0)
    net.add_layer(0, nonlin_type=ly.NONLIN_NAME_SIGMOID, dropout=0)

    print ''
    print '========'
    print 'Training'
    print '========'
    print ''
    print net
    print ''

    mmd_learner = gen.StochasticGenerativeNetLearner(net)
    mmd_learner.load_data(x_train)

    sigma = [mmd_sigma]
    sigma_weights = [1]
    learn_rate = mmd_learn_rate
    momentum = mmd_momentum

    minibatch_size = 1000
    n_sample_update_iters = 1
    max_iters = 40000
    i_checkpoint = 2000

    mmd_learner.set_output_dir(output_base + '/mmd')
    #net.set_loss(ls.LOSS_NAME_MMDGEN_MULTISCALE, loss_after_nonlin=True, sigma=sigma, scale_weight=sigma_weights, loss_weight=1000)
    net.set_loss(ls.LOSS_NAME_MMDGEN_SQRT_GAUSSIAN, loss_after_nonlin=True, sigma=sigma, scale_weight=sigma_weights, loss_weight=1000)

    print '**********************************'
    print net.loss
    print '**********************************'
    print ''

    def f_checkpoint(i_iter, w):
        mmd_learner.save_checkpoint('%d' % i_iter)

    mmd_learner.train_sgd(minibatch_size=minibatch_size, n_samples_per_update=minibatch_size, 
            n_sample_update_iters=n_sample_update_iters, learn_rate=learn_rate, momentum=momentum, 
            weight_decay=0, learn_rate_schedule={10000:learn_rate/10.0},
            momentum_schedule={10000:1-(1-momentum)/10.0},
            learn_rate_drop_iters=0, decrease_type='linear', adagrad_start_iter=0,
            max_iters=max_iters, iprint=100, i_exe=i_checkpoint, f_exe=f_checkpoint)
    mmd_learner.save_model()

    # Evaluation

    print ''
    print '===================='
    print 'Evaluating the model'
    print '===================='
    print ''

    log_prob, std, sigma = ev.kde_eval_mnist(net, x_val, verbose=False)
    test_log_prob, test_std, _ = ev.kde_eval_mnist(net, x_test, sigma_range=[sigma], verbose=False)

    print 'Validation: %.2f (%.2f)' % (log_prob, std)
    print 'Test      : %.2f (%.2f)' % (test_log_prob, test_std)
    print ''

    write_config(output_base + '/params_and_results.cfg', { 
        'ae_n_hids' : ae_n_hids, 'ae_dropout' : ae_dropout, 'ae_learn_rate' : ae_learn_rate,
        'ae_momentum' : ae_momentum, 'mmd_n_hids': mmd_n_hids,
        'mmd_sigma': mmd_sigma, 'mmd_sigma_weights': sigma_weights, 'mmd_learn_rate': mmd_learn_rate,
        'mmd_momentum': mmd_momentum, 'mmd_minibatch_size': minibatch_size, 
        'mmd_n_sample_update_iters': n_sample_update_iters, 'mmd_max_iters': max_iters,
        'mmd_i_checkpoint': i_checkpoint, 'val_log_prob': log_prob, 'val_std': std, 
        'test_log_prob': test_log_prob, 'test_std': test_std })

    print '>>>> output_dir = %s' % output_base
    print ''

    return log_prob

def tfd_mmd_input_space(n_hids=[10,64,256,256,1024], sigma=[5,10,20,40,80,160], learn_rate=2, momentum=0.9):
    """
    return validation log prob.
    """
    gnp.seed_rand(8)

    # train on only one fold - that's enough as the training set is the same across folds
    x_train, x_val, x_test = load_tfd_fold(0)

    print ''
    print 'Training data: %d x %d' % x_train.shape

    in_dim = n_hids[0]
    out_dim = x_train.shape[1]

    net = gen.StochasticGenerativeNet(in_dim, out_dim)
    for i in range(1, len(n_hids)):
        net.add_layer(n_hids[i], nonlin_type=ly.NONLIN_NAME_RELU, dropout=0)
    net.add_layer(0, nonlin_type=ly.NONLIN_NAME_SIGMOID, dropout=0)

    # place holder loss
    net.set_loss(ls.LOSS_NAME_MMDGEN, loss_after_nonlin=True, sigma=80, loss_weight=1000)

    print ''
    print '========'
    print 'Training'
    print '========'
    print ''
    print net
    print ''

    mmd_learner = gen.StochasticGenerativeNetLearner(net)
    mmd_learner.load_data(x_train)

    output_base = OUTPUT_BASE_DIR + '/tfd/input_space'

    #sigma = [2,5,10,20,40,80]
    sigma_weights = [1,1,1,1,1,1]
    #learn_rate = 1
    #momentum = 0.9

    minibatch_size = 1000
    n_sample_update_iters = 1
    max_iters = 48000
    i_checkpoint = 2000

    output_dir = output_base + '/nhids_%s_sigma_%s_lr_%s_m_%s' % (
            '_'.join([str(nh) for nh in n_hids]), '_'.join([str(s) for s in sigma]), str(learn_rate), str(momentum))

    print ''
    print '>>>> output_dir = %s' % output_dir
    print ''

    mmd_learner.set_output_dir(output_dir)
    #net.set_loss(ls.LOSS_NAME_MMDGEN_MULTISCALE, loss_after_nonlin=True, sigma=sigma, scale_weight=sigma_weights, loss_weight=1000)
    net.set_loss(ls.LOSS_NAME_MMDGEN_SQRT_GAUSSIAN, loss_after_nonlin=True, sigma=sigma, scale_weight=sigma_weights, loss_weight=1000)

    print '**********************************'
    print net.loss
    print '**********************************'
    print ''

    def f_checkpoint(i_iter, w):
        mmd_learner.save_checkpoint('%d' % i_iter)

    mmd_learner.train_sgd(minibatch_size=minibatch_size, n_samples_per_update=minibatch_size, 
            n_sample_update_iters=n_sample_update_iters, learn_rate=learn_rate, momentum=momentum, 
            weight_decay=0, learn_rate_schedule={10000:learn_rate/10.0},
            momentum_schedule={10000:1-(1-momentum)/10.0},
            learn_rate_drop_iters=0, decrease_type='linear', adagrad_start_iter=0,
            max_iters=max_iters, iprint=100, i_exe=i_checkpoint, f_exe=f_checkpoint)

    mmd_learner.save_model()

    print ''
    print '===================='
    print 'Evaluating the model'
    print '===================='
    print ''

    x_val = load_tfd_all_folds('val')
    x_test = load_tfd_all_folds('test')

    log_prob, std, sigma = ev.kde_eval_tfd(net, x_val, verbose=False)
    test_log_prob, test_std, _ = ev.kde_eval_tfd(net, x_test, sigma_range=[sigma], verbose=False)

    print 'Validation: %.2f (%.2f)' % (log_prob, std)
    print 'Test      : %.2f (%.2f)' % (test_log_prob, test_std)
    print ''

    write_config(output_dir + '/params_and_results.cfg', { 'n_hids': n_hids,
        'sigma': sigma, 'sigma_weights': sigma_weights, 'learn_rate': learn_rate,
        'momentum': momentum, 'minibatch_size': minibatch_size, 
        'n_sample_update_iters': n_sample_update_iters, 'max_iters': max_iters,
        'i_checkpoint': i_checkpoint, 'val_log_prob': log_prob, 'val_std': std, 
        'test_log_prob': test_log_prob, 'test_std': test_std })

    print '>>>> output_dir = %s' % output_dir
    print ''

    return log_prob


def tfd_mmd_code_space(
        ae_n_hids=[512, 512, 128], 
        ae_dropout=[0.1, 0.1, 0.1],
        ae_learn_rate=1e-1, 
        ae_momentum=0,
        mmd_n_hids=[10, 64, 256, 256, 1024], 
        mmd_sigma=[1,2,5,10,20,40],
        mmd_learn_rate=1e-1,
        mmd_momentum=0.9):
    """
    ae_n_hids: #hid for the encoder, bottom-up
    ae_dropout: the amount of dropout for each layer in the encoder, same order
    ae_learn_rate, ae_momentum: .
    mmd_n_hids: #hid for the generative net, top-down
    mmd_sigma: scale of the kernel
    mmd_learn_rate, mmd_momentum: .

    Return KDE log_likelihood on the validation set.
    """
    gnp.seed_rand(8)
    x_train, x_val, x_test = load_tfd_fold(0)

    common_output_base = OUTPUT_BASE_DIR + '/tfd/code_space'
    output_base = common_output_base + '/aeh_%s_dr_%s_aelr_%s_aem_%s_nh_%s_s_%s_lr_%s_m_%s' % (
            cat_list(ae_n_hids), cat_list(ae_dropout), str(ae_learn_rate), str(ae_momentum),
            cat_list(mmd_n_hids), cat_list(mmd_sigma), str(mmd_learn_rate), str(mmd_momentum))

    #######################
    # Auto-encoder training
    #######################

    n_dims = x_train.shape[1]
    h_dim = ae_n_hids[-1]

    encoder = nn.NeuralNet(n_dims, h_dim)
    for i in range(len(ae_n_hids) - 1):
        encoder.add_layer(ae_n_hids[i], nonlin_type=ly.NONLIN_NAME_SIGMOID, dropout=ae_dropout[i])
    encoder.add_layer(0, nonlin_type=ly.NONLIN_NAME_SIGMOID, dropout=ae_dropout[-1])

    decoder = nn.NeuralNet(h_dim, n_dims)
    for i in range(len(ae_n_hids) - 1)[::-1]:
        decoder.add_layer(ae_n_hids[i], nonlin_type=ly.NONLIN_NAME_SIGMOID, dropout=0)
    decoder.add_layer(0, nonlin_type=ly.NONLIN_NAME_SIGMOID, dropout=0)
    decoder.set_loss(ls.LOSS_NAME_BINARY_CROSSENTROPY, loss_weight=1)

    autoenc = nn.AutoEncoder(encoder=encoder, decoder=decoder)

    print ''
    print autoenc
    print ''

    learn_rate = ae_learn_rate
    final_momentum = ae_momentum
    max_iters = 15000
    #max_iters = 200

    nn_pretrainer = learner.AutoEncoderPretrainer(autoenc)
    nn_pretrainer.load_data(x_train)
    nn_pretrainer.pretrain_network(learn_rate=1e-1, momentum=0.5, weight_decay=0, minibatch_size=100,
            max_grad_norm=10, max_iters=max_iters, iprint=100)

    nn_learner = learner.Learner(autoenc)
    nn_learner.set_output_dir(output_base + '/ae')
    nn_learner.load_data(x_train, x_train)

    def f_checkpoint(i_iter, w):
        nn_learner.save_checkpoint('%d' % i_iter)

    nn_learner.train_sgd(learn_rate=learn_rate, momentum=0, weight_decay=0, minibatch_size=100,
            learn_rate_schedule=None, momentum_schedule={50:0.5, 200:final_momentum}, 
            max_grad_norm=10, learn_rate_drop_iters=0, decrease_type='linear', adagrad_start_iter=0,
            max_iters=max_iters, iprint=100, i_exe=2000, f_exe=f_checkpoint)
    nn_learner.save_checkpoint('best')

    ##################
    # Training MMD net
    ##################

    n_hids = mmd_n_hids

    in_dim = n_hids[0]
    out_dim = autoenc.encoder.out_dim

    net = gen.StochasticGenerativeNetWithAutoencoder(in_dim, out_dim, autoenc)
    for i in range(1, len(n_hids)):
        net.add_layer(n_hids[i], nonlin_type=ly.NONLIN_NAME_RELU, dropout=0)
    net.add_layer(0, nonlin_type=ly.NONLIN_NAME_SIGMOID, dropout=0)

    print ''
    print '========'
    print 'Training'
    print '========'
    print ''
    print net
    print ''

    mmd_learner = gen.StochasticGenerativeNetLearner(net)
    mmd_learner.load_data(x_train)

    sigma = mmd_sigma
    sigma_weights = [1] * len(sigma)
    learn_rate = mmd_learn_rate
    momentum = mmd_momentum

    minibatch_size = 1000
    n_sample_update_iters = 1
    max_iters = 48000
    #max_iters = 200
    i_checkpoint = 2000

    mmd_learner.set_output_dir(output_base + '/mmd')
    #net.set_loss(ls.LOSS_NAME_MMDGEN_MULTISCALE, loss_after_nonlin=True, sigma=sigma, scale_weight=sigma_weights, loss_weight=1000)
    net.set_loss(ls.LOSS_NAME_MMDGEN_SQRT_GAUSSIAN, loss_after_nonlin=True, sigma=sigma, scale_weight=sigma_weights, loss_weight=1000)

    print '**********************************'
    print net.loss
    print '**********************************'
    print ''

    def f_checkpoint(i_iter, w):
        mmd_learner.save_checkpoint('%d' % i_iter)

    mmd_learner.train_sgd(minibatch_size=minibatch_size, n_samples_per_update=minibatch_size, 
            n_sample_update_iters=n_sample_update_iters, learn_rate=learn_rate, momentum=momentum, 
            weight_decay=0, learn_rate_schedule={10000:learn_rate/10.0},
            momentum_schedule={10000:1-(1-momentum)/10.0},
            learn_rate_drop_iters=0, decrease_type='linear', adagrad_start_iter=0,
            max_iters=max_iters, iprint=100, i_exe=i_checkpoint, f_exe=f_checkpoint)
    mmd_learner.save_model()

    # Evaluation

    print ''
    print '===================='
    print 'Evaluating the model'
    print '===================='
    print ''

    x_val = load_tfd_all_folds('val')
    x_test = load_tfd_all_folds('test')

    log_prob, std, sigma = ev.kde_eval_tfd(net, x_val, verbose=False)
    test_log_prob, test_std, _ = ev.kde_eval_tfd(net, x_test, sigma_range=[sigma], verbose=False)

    print 'Validation: %.2f (%.2f)' % (log_prob, std)
    print 'Test      : %.2f (%.2f)' % (test_log_prob, test_std)
    print ''

    write_config(output_base + '/params_and_results.cfg', { 
        'ae_n_hids' : ae_n_hids, 'ae_dropout' : ae_dropout, 'ae_learn_rate' : ae_learn_rate,
        'ae_momentum' : ae_momentum, 'mmd_n_hids': mmd_n_hids,
        'mmd_sigma': mmd_sigma, 'mmd_sigma_weights': sigma_weights, 'mmd_learn_rate': mmd_learn_rate,
        'mmd_momentum': mmd_momentum, 'mmd_minibatch_size': minibatch_size, 
        'mmd_n_sample_update_iters': n_sample_update_iters, 'mmd_max_iters': max_iters,
        'mmd_i_checkpoint': i_checkpoint, 'val_log_prob': log_prob, 'val_std': std, 
        'test_log_prob': test_log_prob, 'test_std': test_std })

    print '>>>> output_dir = %s' % output_base
    print ''

    return log_prob

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model parameter tuning')
    parser.add_argument('-m', '--mode', choices=['mnistinput', 'mnistcode', 'tfdinput', 'tfdcode'])
    args = parser.parse_args()

    print ''
    print '************************'
    print 'Testing %s' % args.mode
    print '************************'
    print ''

    if args.mode == 'mnistinput':
        mnist_mmd_input_space()
    elif args.mode == 'mnistcode':
        mnist_mmd_code_space()
    elif args.mode == 'tfdinput':
        tfd_mmd_input_space()
    elif args.mode == 'tfdcode':
        tfd_mmd_code_space()


