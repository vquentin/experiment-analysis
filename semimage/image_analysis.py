# Copyright (c) 2021, Quentin Van Overmeere
# Licensed under MIT License

import numpy as np
from tifffile import TiffFile as tf
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import AxesGrid
import math

from pathlib import Path
import pwlf
import logging

from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
from skimage import segmentation, feature, future, morphology
from skimage.measure import label
from skimage import transform
from sklearn.ensemble import RandomForestClassifier
from functools import partial

from scipy.interpolate import interp1d

from semimage.line import Line
from semimage.sem_image import SEMZeissImage
from semimage.sem_metadata import SEMZeissMetadata
import semimage.config as config

import random

from skimage.segmentation import (morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  checkerboard_level_set)


def get_porous_thickness(sem_image):
    edges = edges_filter(sem_image, show=False)
    edges_sides = edges_on_side(edges, show=False, image=sem_image)
    lines = find_lines(mask_center_h(edges_sides, 0.5),
                       show=True, image=sem_image)
    cavities = find_cavity(lines, show=True)
    return random.randint(0, 10)


def __overlay(*args):
    """Returns an array overlaying binary values (e.g. as returned by canny)
    with alpha channel

    Inputs:
    *args: a sequence of 2-D arrays of bool, or a 3-D array of bool (images
    stacked along last dimension)
    """
    if np.squeeze(args[0]).ndim is 3:
        args = args[0].transpose(2, 0, 1)

    R = np.zeros_like(np.squeeze(args[0]), dtype=int)
    G = R.copy()
    B = R.copy()

    for i, features in enumerate(args):
        R += features*config.colors[i % (len(config.colors))][0]
        G += features*config.colors[i % (len(config.colors))][1]
        B += features*config.colors[i % (len(config.colors))][2]
    return np.stack([R, G, B, 255*np.logical_or.reduce(args)], axis=2)


def __plt_overlay(image, *args, axes=None, title=None):
    """Use to overlay an image and features

    Inputs:
    *args: ndarray representing the edges to overlay
    image: ndarray representing the image to show in gray scale
        (default: the instance's self.image)
    axes: a mathplotlib axes object to pass when using subplots
        (default: None, plots in a new window)
    title: a title for the plot
    """
    if axes is None:
        plt.imshow(image, cmap=cm.gray, vmin=0, vmax=255)
        plt.imshow(__overlay(*args))
        if title is not None:
            plt.title(title)
    else:
        axes.imshow(image, cmap=cm.gray, vmin=0, vmax=255)
        axes.imshow(__overlay(*args))
        if title is not None:
            axes.set_title(title)


def edges_filter(sem_image, show=False):
    """Return edges detected in image sem_image."""
    edges = canny(
        sem_image.image, sigma=1.0, low_threshold=None, high_threshold=None,
        mask=sem_image.mask, use_quantiles=False)
    selem = morphology.rectangle(2, 5, dtype=bool)
    closed = morphology.binary_closing(edges, selem=selem)
    noise_reduced = np.array(
        morphology.remove_small_objects(
                label(closed), min_size=100, connectivity=2),
        dtype=bool)

    if show:
        _, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True,
                               figsize=(15, 8))
        ax = axes.ravel()
        __plt_overlay(sem_image.image, edges, axes=ax[0],
                      title=f"Canny filter on {sem_image.image_name}")
        __plt_overlay(sem_image.image, closed, axes=ax[1],
                      title='Binary closing')
        __plt_overlay(sem_image.image, noise_reduced, axes=ax[2],
                      title='Small objects removed')
    return noise_reduced


def edges_on_side(edges, show=False, image=None):
    """
    Return edges detected on top and bottom side of binary image.

    Positional arguments:
    edges: a binary image

    Keyword arguments:
    show: show diagnostics
    image: a sem_image if we want to plot it with diagnostics
    """
    # TODO implement a version of this that is orientation independent?
    edges_on_sides = np.zeros(edges.shape+(2,), dtype=bool)
    idx = np.arange(edges.shape[1])[np.newaxis, :]
    weight_matrix = np.tile(
        np.arange(1, edges.shape[0]+1, 1)[:, np.newaxis],
        (1, edges.shape[1]))  # defined such that side 0 = bottom, side 1 = top
    weights = np.stack((weight_matrix, np.flipud(weight_matrix)), axis=-1)
    mask = edges == 0

    # for each side
    for i in range(weights.shape[-1]):
        edges_on_sides[np.argmax(edges*weights[..., i], axis=0), idx, i] = True
    edges_on_sides[mask, :] = False

    if show:
        if isinstance(image, SEMZeissImage):
            image = image.image
        else:
            image = np.ones(edges.shape)
        _, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True,
                               figsize=(15, 8))
        ax = axes.ravel()
        __plt_overlay(image, edges, axes=ax[0],
                      title='Edges')
        __plt_overlay(image, edges_on_sides, axes=ax[1],
                      title='Edges on sides')
    return edges_on_sides


def mask_center_h(image, portion):
    """Return the image with columns in center portion removed."""
    ncols = int(round(image.shape[1]*portion/2))
    image_c = image.copy()
    image_c[:, ncols:-ncols] = 0
    return image_c


def find_lines(edges_sides, show=False, image=None):
    """Returns two lines for each image in a stack.

    Positional arguments:
    edges_sides: a stack of binary images along the last dimension
    Keyword arguments:
    show: overlay original image and lines found (default: False)
    image: a sem_image object
    """
    n_lines_max = 1
    lines = []
    # TODO replace with orientation insensitive code
    theta = 90  # 90 degrees = horizontal line
    dTheta = 10  # delta around which to search
    resTheta = 0.05  # smallest resolvable angle
    thetas = np.linspace(
        (theta-dTheta)*(np.pi / 180), (theta+dTheta)*(np.pi / 180),
        round(2*dTheta/resTheta))
    for i in range(edges_sides.shape[-1]):
        accum, angles, dists = hough_line_peaks(
            *hough_line(edges_sides[..., i], theta=thetas),
            num_peaks=n_lines_max)
        # for each line
        for _, angle, dist in zip(accum, angles, dists):
            lines.append(Line(side=i, angle=angle, dist=dist, image=image))

    if show:
        _, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True,
                               figsize=(15, 8))
        ax = axes.ravel()
        __plt_overlay(image.image, edges_sides, axes=ax[0],
                      title="Edges on sides")
        ax[1].imshow(image.image, cmap=cm.gray, vmin=0, vmax=255)
        for line in lines:
            line.show(ax[1])
        plt.tight_layout()
    return lines


def find_cavity(lines, show=True):
    """
    Return a list of cavities from a starting set of lines
    """
    for line in lines:
        print(f"Line has bgd on same side? {line.background_on_side()}")
    return None


def classify(self, lines=None, debug=False):
    """Return a list of dictionnary items consisting in the line and its classification

    Keyword arguments:
    lines: a list of lines, (default: self._lines)
    debug: a flag to plot the lines detected
    """
    if lines is None:
        lines = self._lines
    classLines = []
    for line in lines:
        classLines.append({
            'Type': line.classify(debug=False),
            'Line': line
        })

    if debug:
        _, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 5), sharex=True, sharey=True)
        ax = axes.ravel()
        self.__plt_overlay(self.edges, axes=ax[0], title="Edges")
        ax[1].imshow(self.image, cmap=cm.gray)
        ax[2].imshow(self.image, cmap=cm.gray)
        for i, line in enumerate(classLines):
            ax[1].plot(*line['Line'].plotPoints, '-', c=np.array(config.colors[i])/255)
            if line['Type'] is 'isCavity':
                ax[2].plot(*line['Line'].plotPoints, '-', c=np.array(config.colors[1])/255)
        ax[2].set_title("Detected lines")
        ax[2].set_title("Line classification (blue=cavity)")

    return classLines


def analyze(self, analyses=None):
    """Analyze the image.

    Keyword arguments:
    analyses: a list of analysis to perform on an individual image. Each analysis can
    be one of 'cavity_xy', 'cavity_y', 'thickness_xy', "thickness_y' (default: None).
    """
    pass


def analyzeCavity(self, edges=None, lines=None):
    if lines is None:
        lines = self._classLines
    if edges is None:
        edges = self._edgesOnSides
    for line in lines:
        if line['Type'] is 'isCavity':
            x, y = line['Line'].distToEdge(edges[..., line['Line'].side], debug=False)
            self.cavity = self.fitCavity(x, y, debug=False)


def fitCavity(self, x, y, debug=False):
    pwlfCavity = pwlf.PiecewiseLinFit(x, y)
    breaks = pwlfCavity.fit(5)
    # TODO: handle singular matrix case
    _ = pwlfCavity.p_values(method='non-linear', step_size=1e-4)
    se = pwlfCavity.se  # standard errors

    width_nm = (breaks[4]-breaks[1])*self.pixelSize
    width_nm_unc = (se[8]**2+se[5]**2)**0.5*self.pixelSize

    rangeThickness = np.logical_and(x > breaks[1], x < breaks[4])
    thickness_nm = np.median(y[rangeThickness])*self.pixelSize
    thickness_nm_unc = np.std(y[rangeThickness])*self.pixelSize
    log.debug(f"Cavity width: {width_nm} +- {width_nm_unc} nm")
    log.debug(f"Cavity thickness: {thickness_nm} +- {thickness_nm_unc} nm")

    if debug:
        plt.figure()
        xHat = np.arange(min(x), max(x))
        yHat = pwlfCavity.predict(xHat)
        plt.plot(x, y, 'ko')
        plt.plot(xHat, yHat, 'k-')

    return {'width': width_nm, 'width_unc': width_nm_unc, 'thick': thickness_nm, 'thick_unc': thickness_nm_unc}


def analyzePorous(self, edges=None, lines=None):
    if lines is None:
        lines = self._classLines
    if edges is None:
        edges = self._edgesOnSides
    for line in lines:
        if line['Type'] is 'isCavity':
            side = line['Line'].side
            otherSide = math.floor(side/2)*2+(side+1) % 2
            x, y = line['Line'].distToEdge(edges[..., otherSide], debug=False)
            self.porous = self.fitPorous(x, y, debug=False)


def fitPorous(self, x, y, debug=False):
    baseline = np.isclose(y, [0], rtol=0.1, atol=2)
    # try linear pwlf first, should work well when porous follows cavity lines
    pwlfTest = pwlf.PiecewiseLinFit(x, y, degree=0)
    _ = pwlfTest.fit(3)
    R2 = pwlfTest.r_squared()

    # spline
    f = interp1d(x, y, kind='cubic')
    xHat = np.arange(min(x), max(x), 0.1)
    yHat = f(xHat)

    log.debug(f"R2-value for porous fit is {R2}")

    if R2 > 0.94:
        # porous shape is same as cavity-shape (3 horizontal lines)
        pwlfPorous = pwlf.PiecewiseLinFit(x, y, degree=1)
        breaks = pwlfPorous.fit(5)
        _ = pwlfPorous.p_values(method='non-linear', step_size=1e-4)
        se = pwlfPorous.se  # standard errors
        width_nm = (breaks[4]-breaks[1])*self.pixelSize
        width_nm_unc = (se[8]**2+se[5]**2)**0.5*self.pixelSize
        rangeThickness = np.logical_and(x > breaks[1], x < breaks[4])
        thickness_nm = np.median(y[rangeThickness])*self.pixelSize
        thickness_nm_unc = np.std(y[rangeThickness])*self.pixelSize
        log.debug(f"Porous width: {width_nm} +- {width_nm_unc} nm")
        log.debug(f"Porous thickness: {thickness_nm} +- {thickness_nm_unc} nm")
    else:
        x1 = x[np.argmin(baseline)]
        xRev = x[::-1]
        x2 = xRev[np.argmin(baseline[::-1])]
        width_nm = ((x2-x1)*self.pixelSize - self.cavity['width'])*0.5
        width_nm_unc = (1*self.pixelSize**2 + self.cavity['width_unc']**2)*0.5
        thickness_nm = (np.max(yHat)*self.pixelSize - self.cavity['thick'])
        thickness_nm_unc = (self.pixelSize**2 + self.cavity['thick_unc']**2)**0.5
        log.debug(f"Porous width: {width_nm} +- {width_nm_unc} nm")
        log.debug(f"Porous thickness: {thickness_nm} +- {thickness_nm_unc} nm")

    if debug:
        yHatTest = pwlfTest.predict(xHat)
        plt.figure()
        plt.plot(x, y, 'o', xHat, yHat, '-', xHat, yHatTest, '--')
    return {'width': width_nm, 'width_unc': width_nm_unc, 'thick': thickness_nm, 'thick_unc': thickness_nm_unc}
