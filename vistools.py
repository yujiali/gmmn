"""This module contains useful tools that makes data visualization easier.

Yujia Li, 03/2013
"""

import numpy as np
import matplotlib.pyplot as plt

def bwpatchview(data, imsz, nrows, gridwidth=1, gridintensity=0, rowmajor=True, ax=None):
    """Display a list of images in grid view.

    data: N*D matrix, each row is an image
    imsz: 2-D tuple, size of the images
    nrows: number of rows to arrange the images in a plot
    gridwidth: number of pixels to use for the grid
    gridintensity: the intensity value for the grid
    rowmajor: are the images stored in a row-major order or coloumn-major order
    ax: if provided, the image will be shown on the given axis.

    The images are orgainzed in rows from left to right.
    """

    N, D = data.shape
    sx, sy = imsz

    ncols = N // nrows
    if N % nrows:
        ncols += 1

    img = np.ones((sx * nrows + gridwidth * (nrows + 1), 
        sy * ncols + gridwidth * (ncols + 1))) * gridintensity

    for ix in range(0, nrows):
        for iy in range(0, ncols):
            idx = ix * ncols + iy
            if idx >= N:
                break
            xstart = gridwidth + ix * (sx + gridwidth)
            xend = xstart + sx
            ystart = gridwidth + iy * (sy + gridwidth)
            yend = ystart + sy

            if rowmajor:
                img[xstart:xend, ystart:yend] = data[idx].reshape(imsz)
            else:
                img[xstart:xend, ystart:yend] = data[idx].reshape((imsz[1], imsz[0])).T

    if ax != None:
        ax.imshow(img, cmap='gray', interpolation='nearest')
        ax.axis('off')
    else:
        plt.imshow(img, cmap='gray', interpolation='nearest')
        plt.axis('off')
    plt.show()

def cpatchview(data, imsz, nrows, gridwidth=1, gridintensity=0, rowmajor=True, ax=None, normalize=False):
    """Display a list of color images in grid view.

    data: N*(3*D) matrix, each row is a color image
    imsz: 2-D tuple, size of the images, should have prod(imsz)=D
    nrows: number of rows to arrange the images in a plot
    gridwidth: number of pixels to use for the grid
    gridintensity: the intensity value for the grid
    rowmajor: specify whether the images are stored in row-major order or 
        column-major order
    ax: if provided, the image will be shown on the given axis.
    normalize: if set and data is real valued, data is normalized to within [0,1]
    
    The images are organized in rows from left to right.
    """
    N, D = data.shape
    D = D / 3
    sx, sy = imsz

    ncols = N / nrows
    if N % nrows:
        ncols += 1

    img = np.ones((sx * nrows + gridwidth * (nrows + 1), 
        sy * ncols + gridwidth * (ncols + 1), 3), dtype=data.dtype) * gridintensity

    for ix in range(0, nrows):
        for iy in range(0, ncols):
            idx = ix * ncols + iy
            if idx >= N:
                break
            xstart = gridwidth + ix * (sx + gridwidth)
            xend = xstart + sx
            ystart = gridwidth + iy * (sy + gridwidth)
            yend = ystart + sy

            if rowmajor:
                img[xstart:xend, ystart:yend, :] = data[idx].reshape((3,sx,sy)).transpose((1,2,0))
            else:
                img[xstart:xend, ystart:yend] = data[idx].reshape((3,sy,sx)).transpose((2,1,0))

    if ax != None:
        if normalize:
            ax.imshow((img - img.min()) / (img.max() - img.min() + 1e-20), interpolation='nearest')
        else:
            ax.imshow(img, interpolation='nearest')
        ax.axis('off')
    else:
        if normalize:
            plt.imshow((img - img.min()) / (img.max() - img.min() + 1e-20), interpolation='nearest')
        else:
            plt.imshow(img, interpolation='nearest')
        plt.axis('off')
    plt.show()

def listpatchview(data, nrows, gridwidth=1, gridintensity=0, ax=None):
    """Display a list of images in grid view.

    data: a list of images of the same size, can be either color or gray
        images, but should be consistent.
    nrows: number of rows to arrange the images in a plot
    gridwidth: number of pixels to use for the grid
    gridintensity: the intensity value for the grid
    ax: if provided, the image will be shown on the given axis
    
    The images are organized in rows from left to right.
    """
    N = len(data)
    sx, sy = data[0].shape[:2]
    D = sx * sy

    ncols = N / nrows
    if N % nrows:
        ncols += 1

    if len(data[0].shape) < 3 or data[0].shape[2] == 1:
        n_color = 1
        img = np.ones((sx * nrows + gridwidth * (nrows + 1), 
            sy * ncols + gridwidth * (ncols + 1)),dtype=data[0].dtype) * gridintensity
    else:
        n_color = 3
        assert(data[0].shape[2] == n_color)
        img = np.ones((sx * nrows + gridwidth * (nrows + 1), 
            sy * ncols + gridwidth * (ncols + 1), n_color),dtype=data[0].dtype) * gridintensity

    for ix in range(0, nrows):
        for iy in range(0, ncols):
            idx = ix * ncols + iy
            if idx >= N:
                break
            xstart = gridwidth + ix * (sx + gridwidth)
            xend = xstart + sx
            ystart = gridwidth + iy * (sy + gridwidth)
            yend = ystart + sy

            if n_color == 3:
                img[xstart:xend, ystart:yend, :] = data[idx]
            else:
                img[xstart:xend, ystart:yend] = data[idx]

    if ax == None:
        ax = plt
    if n_color == 3:
        ax.imshow(img, interpolation='nearest')
    else:
        ax.imshow(img, cmap='gray', interpolation='nearest')
    ax.axis('off')
    plt.show()

def plot2dgaussian(mu, sigma, npoints=100, linespec=None, linewidth=1, ax=None, *args, **kwargs):
    """Plot a 2D Gaussian distribution. Showing on the plot are the mean of 
    the Gaussian and an ellipse corresponding to 1 standard deviation (not
    strictly speaking standard deviation, but similar).
    """
    eig, Q = np.linalg.eig(sigma)
    scale = np.sqrt(eig).reshape(1,2)

    x = np.zeros((npoints + 1, 2))

    for n in range(npoints):
        angle = 2 * np.pi * n / npoints
        x[n,:] = mu + (scale * np.array([[np.cos(angle), np.sin(angle)]])).dot(Q.T)

    x[npoints,:] = x[0,:]

    if ax == None:
        ax = plt

    if linespec:
        ax.plot(x[:,0], x[:,1], linespec, linewidth=linewidth, *args, **kwargs)
    else:
        ax.plot(x[:,0], x[:,1], linewidth=linewidth, *args, **kwargs)
    plt.show()

def intarray_to_rgb(x, cmap):
    """
    x: MxN is an array of int indices into the cmap
    cmap: int->(r,g,b) mapping

    Return converted y of shape MxNx3
    """
    y = np.zeros((x.shape[0], x.shape[1], 3), dtype=np.uint8)

    if isinstance(cmap, dict):
        for c in cmap:
            y[x == c] = cmap[c]
    elif isinstance(cmap, np.ndarray):
        for i in range(cmap.shape[0]):
            y[x == i] = cmap[i]

    return y

def pil_png_cmap_to_dict(pil_palette):
    """
    // cmap is a color map from PIL after loading a color png file. Format: list
    // of (rgb, idx) tuples. rgb is an integer representation of the RGB value.

    pil_palette is a list of palette values. Should be 3xC where C is the 
    number of colors.

    Return a dict of (idx -> (r,g,b)).
    """
    cm = {}
    p = np.array(pil_palette, dtype=np.uint8).reshape(len(pil_palette)/3, 3)
    for i in range(p.shape[0]):
        cm[i] = p[i]
    return cm

