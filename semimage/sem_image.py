# Copyright (c) 2021, Quentin Van Overmeere
# Licensed under MIT License

import json
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

from semimage.lines import Line
import semimage.config as config

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class SEMImage(object):
    """This class represents an image, and its associated properties.
    Upon instanciation, the image is loaded as a numpy array, and its metadata
    is loaded.
    Intended for images generated with Zeiss Scanning Electron Microscopes.

    Keyword arguments:
    filePath: a path to a file name (default: None, returns an error)
    debug: a flag to write the metadata to a file and plot the image (default: False)
    """
    def __init__(self, filePath=None, debug=False):
        self.imagePath = Path(filePath)
        with tf(self.imagePath) as image:
            self.imageName = self.imagePath.stem
            if not hasattr(image, 'sem_metadata'):
                raise Exception("Image is likely not from a Zeiss scanning electron microscope")
            self.image = image.pages[0].asarray()
            log.debug(self.imageName)
            self.__parse_metadata(metaData=image.sem_metadata, debug=False)
            self.mask(debug=False)
        if debug:
            self.plot_image_raw()
        # self.canny(debug=True, sigma = 1.0)
        self.edges = self.canny_closing_skeleton(debug=False)
        self._lines = self.lines(edges=self.edges, debug=False)
        self._classLines = self.classify(self._lines, debug=False)
        self.analyzeCavity()
        self.analyzePorous()
        plt.show()

    def __parse_metadata(self, metaData=None, debug=False):
        """Makes the useful metadata accessible directly in the instance of the object

        Keyword arguments:
        metaData: the metadata from the Zeiss image (default: None)
        debug: flag to write the metadata in json format in the same directory than the file
        """
        if debug:
            # print all the metadata in a file
            f = self.imagePath.parent.joinpath(self.imagePath.stem + '_meta.txt')
            with open(f, mode='w') as fid:
                print(json.dumps(metaData, indent=4), file=fid)
        self.lineCount = metaData["ap_line_counter"][1]
        self.pixelSize = metaData["ap_image_pixel_size"][1]
        self.beamXOffset = metaData["ap_beam_offset_x"][1]
        self.beamYOffset = metaData["ap_beam_offset_y"][1]
        self.stageX = metaData["ap_stage_at_x"][1]
        self.stageY = metaData["ap_stage_at_y"][1]

    def plot_image_raw(self):
        """Plots the image in a new window, without any treatment and show basic diagnostics.
        """
        _, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 5), gridspec_kw={'width_ratios': [1024, 256, 256]})
        ax = axes.ravel()
        ax[0].imshow(self.image, cmap=cm.gray)
        ax[0].set_title('Original image')
        rows = np.arange(0, self.image.shape[0])
        ax[1].plot(self.image.min(axis=1), rows, '-b', label='Min.')
        ax[1].plot(self.image.max(axis=1), rows, '-r', label='Max.')
        ax[1].plot(self.image.mean(axis=1), rows, '-g', label='Mean')
        ax[1].plot(np.median(self.image, axis=1), rows, '-k', label='Median')
        ax[1].legend()
        ax[1].set_title('Statistics')
        ax[1].sharey(ax[0])
        if hasattr(self, 'mask'):
            ax[2].hist(self.image[self._mask].ravel(), bins=256)
            ax[2].set_title('Histogram (mask applied)')
        else:
            ax[2].hist(self.image.ravel(), bins=256)
            ax[2].set_title('Histogram (no mask)')
        plt.tight_layout()

    def mask(self, debug=False):
        """Creates a mask for the bottom portion where no scanning was done.
        If a banner is present, the banner and lines below will be masked.

        Keyword arguments:
        debug: shows the picture of the masked image (default: False)
        """
        lineBanner = 676

        lineMask_min = self.image.min(axis=1)
        lineMask_max = self.image.max(axis=1)
        maskLines = np.ones(lineMask_min.shape, dtype=bool)
        if not hasattr(self, 'lineCount'):
            raise Exception("This function requires meta data")
        maskFirstLine = self.lineCount
        hasBanner = False
        if lineMask_min[lineBanner] == 0 and lineMask_max[lineBanner+1] == 255:
            hasBanner = True
            maskFirstLine = min(lineBanner, maskFirstLine)
        maskLines[maskFirstLine:] = False
        self._mask = np.tile(maskLines, [self.image.shape[1], 1]).T

        if debug:
            maskedImage = self.image.copy()
            maskedImage[~self._mask] = 0
            _, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
            ax = axes.ravel()
            ax[0].imshow(self.image, cmap=cm.gray)
            ax[0].set_title(f"Original image (banner: {hasBanner})")
            ax[1].imshow(maskedImage, cmap=cm.gray)
            ax[1].set_title('Masked image')
            plt.tight_layout()

    def canny(self, image=None, debug=False, sigma=1.0):
        """Return a canny edge filter for the masked image.

        Keyword arguments:
        image: the image as an ndarray on which to apply the canny filter (default: the
            instance's self.image). The instance mask is used even if image is specified.
        debug: show a graph overlaying the edges with the original image (default: False)
        sigma: override the default sigma value in the canny filter
        """
        if image is None:
            image = self.image
        edges = canny(image, sigma=sigma, low_threshold=None, high_threshold=None, mask=self._mask, use_quantiles=False)
        if debug:
            plt.figure()
            self.__plt_imshow_overlay(edges, title=f"Detected edges with sigma = {sigma}")
            plt.tight_layout()
        return edges

    def __overlay(self, *args):
        """Returns an array overlaying binary values (e.g. as returned by canny) with alpha channel

        Inputs:
        *args: a sequence of 2-D arrays of bool, or a 3-D array of bool (images stacked along last dimension)
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

    def __plt_imshow_overlay(self, *args, image=None, axes=None, title=None):
        """Use to overlay an image and features

        Inputs:
        *args: ndarray representing the edges to overlay
        image: ndarray representing the image to show in gray scale (default: the instance's self.image)
        axes: a mathplotlib axes object to pass when using subplots (default: None, plots in a new window)
        title: a title for the plot
        """
        if image is None:
            image = self.image
        if axes is None:
            plt.imshow(image, cmap=cm.gray)
            plt.imshow(self.__overlay(*args))
            if title is not None:
                plt.title(title)
        else:
            axes.imshow(image, cmap=cm.gray)
            axes.imshow(self.__overlay(*args))
            if title is not None:
                axes.set_title(title)

    def canny_closing_skeleton(self, debug=False):
        edges = self.canny(sigma=1.1)
        closed1 = morphology.closing(edges)
        skeleton1 = morphology.skeletonize(closed1)
        closed2 = morphology.area_closing(skeleton1, area_threshold=1000, connectivity=1, parent=None, tree_traverser=None)
        skeleton2 = morphology.skeletonize(closed2)
        noise_reduced = np.array(morphology.remove_small_objects(label(skeleton2), min_size=100,), dtype=bool)

        if debug:
            _, axes = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True, figsize=(15, 8))
            ax = axes.ravel()
            ax[0].imshow(self.image, cmap=cm.gray)
            ax[0].set_title('Original image')
            self.__plt_imshow_overlay(skeleton1, axes=ax[1], title='Skeleton after closing')
            self.__plt_imshow_overlay(skeleton2, axes=ax[2], title='Skeleton after area closing')
            self.__plt_imshow_overlay(noise_reduced, axes=ax[3], title='Small object removal')
            self.__plt_imshow_overlay(edges, axes=ax[4], title='Canny')
            self.__plt_imshow_overlay(np.logical_xor(edges, closed1), axes=ax[5], title='Dismissed closing')
            self.__plt_imshow_overlay(np.logical_xor(skeleton1, closed2), axes=ax[6], title='Dismissed area closing')
            self.__plt_imshow_overlay(np.logical_xor(skeleton2, noise_reduced), axes=ax[7], title='Dismissed small object')
        return noise_reduced

    def lines(self, edges=None, debug=False):
        """Returns at most two and two lines, respectively from each side of the image.
        Starting point to classify these lines as cavities or interface.

        Keyword arguments:
        edges: a binary image from edge detection algorithm (default: will apply self.canny_closing_skeleton())
        debug: overlay original image and lines found (default: False)
        orientation: the orientation of the image (default: Horizontal, accepted values are Horizontal, Vertical, Oblique)
        """
        if edges is None:
            edges = self.canny_closing_skeleton(debug=False)

        # TODO implement a version of this that is orientation independent by rotating the edge image

        edgesOnSides = np.zeros(edges.shape+(2,), dtype=bool)  # array that will contain the edges from both sides of interest
        idx = np.arange(edges.shape[1])[np.newaxis, :]
        weightMatrix = np.tile(np.arange(1, edges.shape[0]+1, 1)[:, np.newaxis], (1, edges.shape[1]))  # defined such that side 0 = bottom, side 1 = top
        weights = np.stack((weightMatrix, np.flipud(weightMatrix)), axis=-1)

        lines = []
        mask = edges == 0

        # for each side
        for i in range(weights.shape[-1]):
            edgesOnSides[np.argmax(edges*weights[..., i], axis=0), idx, i] = True
        edgesOnSides[mask, :] = False
        self._edgesOnSides = edgesOnSides
        for i in range(weights.shape[-1]):
            # TODO replace with orientation insensitive code
            theta = 90  # 90 degrees = horizontal line
            dTheta = 10  # delta around which to search
            resTheta = 0.05  # smallest resolvable angle
            thetas = np.linspace((theta-dTheta)*(np.pi / 180), (theta+dTheta)*(np.pi / 180), round(2*dTheta/resTheta))
            accum, angles, dists = hough_line_peaks(*hough_line(edgesOnSides[..., i], theta=thetas), num_peaks=2)
            # for each line
            for _, angle, dist in zip(accum, angles, dists):
                lines.append(Line(side=i, angle=angle, dist=dist, image=self.image))
        if debug:
            _, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(15, 8))
            ax = axes.ravel()
            self.__plt_imshow_overlay(edgesOnSides, axes=ax[0], title="Edges on sides")
            ax[1].imshow(self.image, cmap=cm.gray)
            ax[1].set_title("Lines detected")
            plt.tight_layout()
            for k, line in enumerate(lines):
                ax[1].plot(*line.plotPoints, '-', c=np.array(config.colors[k])/255)
        return lines

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
            self.__plt_imshow_overlay(self.edges, axes=ax[0], title="Edges")
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
