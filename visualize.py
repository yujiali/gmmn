import matplotlib.pyplot as plt
import numpy as np
import gnumpy as gnp
import vistools as vt
import core.generative as gen
import os
import time
import core.util as util
import scipy.misc as misc

from mpl_toolkits.axes_grid1 import AxesGrid

plt.ion()

def nn_search(samples, database, top_k=1, imsz=[28,28], orientation='horizontal', output_file=None, pad=0.1):
    if orientation not in ['horizontal', 'vertical']:
        print '[Error] orientation must be either horizontal or vertical'
        return

    g_samples = util.to_garray(samples)
    g_database = util.to_garray(database)

    if isinstance(database, gnp.garray):
        database = database.asarray()
    if isinstance(samples, gnp.garray):
        samples = samples.asarray()

    n_samples, n_dims = samples.shape
    nn = np.empty((n_samples * top_k, n_dims), dtype=np.float)

    for i in range(n_samples):
        v = g_samples[i]
        d = ((g_database - v)**2).sum(axis=1)
        idx = d.asarray().argsort()
        top_candidates = database[idx[:top_k]]
        if orientation == 'horizontal':
            nn[np.arange(i, i+n_samples*top_k, n_samples)] = top_candidates
        elif orientation == 'vertical':
            nn[i*top_k:(i+1)*top_k] = top_candidates

    f = plt.figure()
    grid = AxesGrid(f, 111, nrows_ncols=(2,1), axes_pad=pad)

    vt.bwpatchview(samples, imsz, 1, gridintensity=1, ax=grid[0])
    if orientation == 'horizontal':
        vt.bwpatchview(nn, imsz, top_k, gridintensity=1, ax=grid[1])
    elif orientation == 'vertical':
        vt.bwpatchview(nn, imsz, n_samples, gridintensity=1, ax=grid[1])

    if output_file is not None:
        plt.savefig(output_file, bbox_inches='tight')

def view_checkpoints(model_dir, sigma, imsz=[28,28], figid=101):
    """
    checkpoint files should have a name matching the following:
    <model_dir>/checkpoint_<sigma>_<iter>.pdata
    """
    prefix = '%s/checkpoint_%s' % (model_dir, str(sigma))
    checkpoint_numbers = sorted([int(fpath.split('.')[0].split('_')[-1]) for fpath in os.listdir(model_dir) if fpath.startswith('checkpoint_%s' % str(sigma))])

    net = gen.StochasticGenerativeNet()

    plt.figure(figid, figsize=(10,8))
    ax = plt.subplot(111)

    for i in checkpoint_numbers:
        net.load_model_from_file(prefix + '_%d.pdata' % i)
        w = net.layers[-1].params.W.asarray()
        ax.cla()
        vt.bwpatchview(w[:400], imsz, int(np.sqrt(w[:400].shape[0])), rowmajor=True, gridintensity=1, ax=ax)
        plt.draw()
        plt.show()
        print 'Checkpoint %d' % i
        time.sleep(0.04)

def generation_on_a_line(net, n_points=100, imsz=[28,28], nrows=10, h_seeds=None):
    if h_seeds is None:
        h = net.sample_hiddens(2)
        z = gnp.zeros((n_points, h.shape[1]))
        diff = h[1] - h[0]
        step = diff / (n_points - 1)
        for i in range(n_points):
            z[i] = h[0] + step * i
    else:
        n_seeds = h_seeds.shape[0]
        z = gnp.zeros((n_points * n_seeds, h_seeds.shape[1]))
        for i in range(n_seeds):
            h0 = h_seeds[i]
            h1 = h_seeds[(i+1) % n_seeds]
            diff = h1 - h0
            step = diff / (n_points - 1)
            for j in range(n_points):
                z[i*n_points+j] = h0 + step * j

    x = net.generate_samples(z=z)
    vt.bwpatchview(x.asarray(), imsz, nrows, rowmajor=True, gridintensity=1)

def generate_morphing_video(net, h_seeds, n_points=100, imsz=[28,28], output_dir='video'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    n_seeds = h_seeds.shape[0]
    z = gnp.zeros((n_points * n_seeds, h_seeds.shape[1]))

    for i in range(n_seeds):
        h0 = h_seeds[i]
        h1 = h_seeds[(i+1) % n_seeds]
        diff = h1 - h0
        step = diff / (n_points - 1)
        for j in range(n_points):
            z[i*n_points+j] = h0 + step * j

    x = net.generate_samples(z=z).asarray()
    for i in range(x.shape[0]):
        misc.imsave(output_dir + '/%d.png' % i, x[i].reshape(imsz))

###################################
# For old experiments
###################################

def plot_dataset(x, t, ax=None):
    if ax is None:
        plt.figure()
        ax = plt.subplot(111)
    ax.plot(x[t==0,0], x[t==0,1], 'o')
    ax.plot(x[t==1,0], x[t==1,1], 'o')

    x_min = x[:,0].min()
    x_max = x[:,0].max()
    y_min = x[:,1].min()
    y_max = x[:,1].max()
    ax.set_xlim([x_min - (x_max - x_min) / 10.0, x_max + (x_max - x_min) / 10.0])
    ax.set_ylim([y_min - (y_max - y_min) / 10.0, y_max + (y_max - y_min) / 10.0])

    plt.show()

    return ax

def plot_decision_boundary(f, x_range, y_range, density, ax=None, **kwargs):
    if ax is None:
        plt.figure()
        ax = plt.subplot(111)

    x, y = np.meshgrid(np.arange(x_range[0], x_range[1], density),
            np.arange(y_range[0], y_range[1], density))

    data = np.c_[x.reshape(x.size,1), y.reshape(y.size,1)]
    z = f(data).reshape(x.shape)

    ax.contour(x, y, z, levels=[0], **kwargs)

    plt.show()

    return ax
