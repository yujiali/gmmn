"""
Script used for generating sample figures used in the paper.

Yujia Li, 02/2015
"""

import core.generative as gen
import pynn.nn as nn
import matplotlib.pyplot as plt
import vistools as vt
import visualize as vis
import dataio.tfd as tfd
import dataio.mnist as mnistio
import gnumpy as gnp
import numpy as np
import os

plt.ion()

# fill in the paths to the model files here
BEST_MNIST_INPUT_SPACE_MODEL = ''
BEST_MNIST_AUTOENCODER = ''
BEST_MNIST_CODE_SPACE_MODEL = ''
BEST_TFD_INPUT_SPACE_MODEL = ''
BEST_TFD_AUTOENCODER = ''
BEST_TFD_CODE_SPACE_MODEL = ''

def get_mnist_input_space_model():
    net = gen.StochasticGenerativeNet()
    net.load_model_from_file(BEST_MNIST_INPUT_SPACE_MODEL)
    return net

def get_mnist_code_space_model():
    ae = nn.AutoEncoder()
    ae.load_model_from_file(BEST_MNIST_AUTOENCODER)
    net = gen.StochasticGenerativeNetWithAutoencoder()
    net.load_model_from_file(BEST_MNIST_CODE_SPACE_MODEL)
    net.autoencoder = ae
    return net

def get_tfd_input_space_model():
    net = gen.StochasticGenerativeNet()
    net.load_model_from_file(BEST_TFD_INPUT_SPACE_MODEL)
    return net

def get_tfd_code_space_model():
    ae = nn.AutoEncoder()
    ae.load_model_from_file(BEST_TFD_AUTOENCODER)
    net = gen.StochasticGenerativeNetWithAutoencoder()
    net.load_model_from_file(BEST_TFD_CODE_SPACE_MODEL)
    net.autoencoder = ae
    return net

def get_model(dataset='mnist', mode='input_space'):
    if dataset == 'mnist':
        if mode == 'input_space':
            return get_mnist_input_space_model()
        elif mode == 'code_space':
            return get_mnist_code_space_model()
    elif dataset == 'tfd':
        if mode == 'input_space':
            return get_tfd_input_space_model()
        elif mode == 'code_space':
            return get_tfd_code_space_model()

def generate_samples(dataset='mnist', mode='input_space'):
    imsz = [28,28] if dataset=='mnist' else [48,48]
    net = get_model(dataset=dataset, mode=mode)
    plt.figure()
    vt.bwpatchview(net.generate_samples(n_samples=30).asarray(), imsz, 5, gridintensity=1)
    if not os.path.exists('figs'):
        os.makedirs('figs')
    plt.savefig('figs/samples_%s_%s.pdf' % (dataset, mode), bbox_inches='tight')

def generate_all_samples():
    generate_samples(dataset='mnist', mode='input_space')
    generate_samples(dataset='mnist', mode='code_space')
    #generate_samples(dataset='tfd', mode='input_space')
    #generate_samples(dataset='tfd', mode='code_space')

def load_train_data(dataset='mnist'):
    if dataset == 'mnist':
        train_data, _, _ = mnistio.load_data()
    elif dataset == 'tfd':
        train_data, _, _ = tfd.load_proper_fold(0, 'unlabeled', scale=True)
        train_data = train_data.reshape(train_data.shape[0], np.prod(train_data.shape[1:]))

    return train_data

def get_nearest_neighbor(dataset='mnist', mode='input_space'):
    imsz = [28,28] if dataset=='mnist' else [48,48]
    net = get_model(dataset=dataset, mode=mode)
    train_data = load_train_data(dataset=dataset)

    if not os.path.exists('figs'):
        os.makedirs('figs')
    vis.nn_search(net.generate_samples(n_samples=12), train_data, top_k=1, imsz=imsz,
            orientation='horizontal', output_file='figs/nn_%s_%s.pdf' % (dataset, mode), pad=0.1)

def get_all_nearest_neighbors():
    get_nearest_neighbor(dataset='mnist', mode='input_space')
    get_nearest_neighbor(dataset='mnist', mode='code_space')
    #get_nearest_neighbor(dataset='tfd', mode='input_space')
    #get_nearest_neighbor(dataset='tfd', mode='code_space')

def get_morphing_figure(dataset='mnist', mode='input_space'):
    imsz = [28,28] if dataset=='mnist' else [48,48]
    net = get_model(dataset=dataset, mode=mode)
    plt.figure()
    gnp.seed_rand(8)
    vis.generation_on_a_line(net, n_points=24, imsz=imsz, nrows=10, h_seeds=net.sample_hiddens(5))

    if not os.path.exists('figs'):
        os.makedirs('figs')
    plt.savefig('figs/morphing_%s_%s.pdf' % (dataset, mode), bbox_inches='tight')

def get_all_morphing_figures():
    get_morphing_figure(dataset='mnist', mode='code_space')
    #get_morphing_figure(dataset='tfd', mode='code_space')

if __name__ == '__main__':
    generate_all_samples()
    get_all_nearest_neighbors()
    get_all_morphing_figures()
