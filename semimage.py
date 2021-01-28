# Copyright (c) 2021, Quentin Van Overmeere
# Licensed under MIT License

import json
import numpy as np
from tifffile import TiffFile as tf
from matplotlib import pyplot as plt
from pathlib import Path


from matplotlib import cm

from skimage import data, io, filters
from skimage.feature import blob_dog, blob_log, blob_doh, canny
from skimage.color import rgb2gray
from skimage.transform import probabilistic_hough_line, hough_line, hough_line_peaks
from skimage import segmentation, feature, future, morphology
from skimage.measure import label
from sklearn.ensemble import RandomForestClassifier
from functools import partial

from mpl_toolkits.axes_grid1 import AxesGrid

class SEMImage(object):
    """This class instanciates an image and loads the metadata. 
    Intended for images generated with Zeiss Scanning Electron Microscopes.

    Keyword arguments:
    filePath: a path to a file name (default: None, returns an error)
    debug: a flag to write the metadata to a file and plot the image (default: False)
    """
    def __init__(self, filePath = None, debug = False):
        self.imagePath = Path(filePath)
        with tf(self.imagePath) as image:
            self.imageName = self.imagePath.stem
            if not hasattr(image, 'sem_metadata'):
                raise Exception("Image is likely not from a Zeiss scanning electron microscope")
            self.image = image.pages[0].asarray()
            self.__parse_metadata(metaData = image.sem_metadata, debug = False)
            self.mask(debug = False)
        if debug:
            self.plot_image_raw()
        #self.canny(debug=True, sigma = 1.0)
        self.canny_closing_skeleton(debug=True)
        #self.classifier(debug=True)
        #self.lines_h_all(debug=False)
        #self.lines_h(debug=True)
        #self.silicon_baseline(debug=True)
        plt.show()

    def __parse_metadata(self, metaData = None, debug = False):
        """Makes the useful metadata accessible directly in the instance of the object
        
        Keyword arguments: 
        metaData: the metadata from the Zeiss image (default: None)
        debug: flag to write the metadata in json format in the same directory than the file
        """
        if debug:
            #print all the metadata in a file
            f = self.imagePath.parent.joinpath(self.imagePath.stem + '_meta.txt')
            with open(f, mode='w') as fid:
                print(json.dumps(metaData, indent = 4), file=fid)
        try:
            self.lineCount = metaData["ap_line_counter"][1]
            self.pixelSize = metaData["ap_image_pixel_size"][1]
            self.beamXOffset = metaData["ap_beam_offset_x"][1]
            self.beamYOffset = metaData["ap_beam_offset_y"][1]
            self.stageX = metaData["ap_stage_at_x"][1]
            self.stageY = metaData["ap_stage_at_y"][1]
        except Exception as e:
            print(f"Failed to read image metadata")
            print(e)

    def plot_image_raw(self):
        """Plots the image in a new window, without any treatment and show basic diagnostics.
        """
        fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize=(12,5), gridspec_kw={'width_ratios': [1024, 256, 256]})
        ax = axes.ravel()
        ax[0].imshow(self.image, cmap=cm.gray)
        ax[0].set_title('Original image')
        rows = np.arange(0, self.image.shape[0])
        ax[1].plot(self.image.min(axis=1), rows, '-b', label='Min.')
        ax[1].plot(self.image.max(axis=1), rows, '-r', label='Max.')
        ax[1].plot(self.image.mean(axis=1), rows, '-g', label='Mean')
        ax[1].plot(np.median(self.image,axis=1), rows, '-k', label='Median')
        ax[1].legend()
        ax[1].set_title('Statistics')
        ax[1].sharey(ax[0])
        if hasattr(self, 'mask'):
            ax[2].hist(self.image[self.mask].ravel(), bins = 256)
            ax[2].set_title('Histogram (mask applied)')
        else:
            ax[2].hist(self.image.ravel(), bins = 256)
            ax[2].set_title('Histogram (no mask)')
        plt.tight_layout()

    def mask(self, debug = False):
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
        self.mask = np.tile(maskLines,[self.image.shape[1],1]).T

        if debug:
            maskedImage = self.image.copy()
            maskedImage[~self.mask] = 0
            fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize=(12,6), sharey=True)
            ax = axes.ravel()
            ax[0].imshow(self.image, cmap=cm.gray)
            ax[0].set_title(f"Original image (banner: {hasBanner})")
            ax[1].imshow(maskedImage, cmap=cm.gray)
            ax[1].set_title('Masked image')
            plt.tight_layout()

    def canny(self, image = None, debug = False, sigma = 1.0):
        """Return a canny edge filter for the masked image.

        Keyword arguments:
        image: the image as an ndarray on which to apply the canny filter (default: the 
            instance's self.image). The instance mask is used even if image is specified.
        debug: show a graph overlaying the edges with the original image (default: False)
        sigma: override the default sigma value in the canny filter
        """
        if image is None:
            image = self.image
        edges = canny(image,sigma=sigma, low_threshold=None, high_threshold=None, mask=self.mask, use_quantiles=False)
        if debug:
            plt.figure()
            self.__plt_imshow_overlay(edges, title = f"Detected edges with sigma = {sigma}")
            plt.tight_layout()
        return edges

    def __overlay(self, edges):
        """Use to overlay binary values (e.g. as returned by canny) in red on an image
        
        Inputs:
        edges: an array of 1,0 values
        """
        return np.stack([edges,np.zeros_like(edges),np.zeros_like(edges),1.0*edges], axis=2)

    def __plt_imshow_overlay(self, features, image=None, axes=None, title=None):
        """Use to overlay an image and features

        Inputs:
        features: ndarray representing the edges to overlay in red
        image: ndarray representing the image to show in gray scale (default: the instance's self.image)
        axes: a mathplotlib axes object to pass when using subplots (default: None, plots in a new window)
        title: a title for the plot
        """
        if image is None:
            image = self.image
        if axes is None:
            plt.imshow(image, cmap=cm.gray)
            plt.imshow(self.__overlay(features))
            if title is not None:
                plt.title(title)
        else:
            axes.imshow(image, cmap=cm.gray)
            axes.imshow(self.__overlay(features))
            if title is not None:
                axes.set_title(title)

    def canny_closing_skeleton(self, debug = False):
        edges = self.canny(sigma = 1.1)
        #closed1 = diameter_closing(img,100,connectivity=10)
        closed1 = morphology.closing(edges)
        skeleton1 = morphology.skeletonize(closed1)
        closed2 = morphology.area_closing(skeleton1, area_threshold=1000, connectivity=1, parent=None, tree_traverser=None)
        skeleton2 = morphology.skeletonize(closed2)
        noise_reduced = morphology.remove_small_objects(label(skeleton2), 100,)

        if debug:
            fig, axes = plt.subplots(nrows = 2, ncols = 4, sharex=True, sharey=True, figsize=(15,8))
            ax = axes.ravel()
            ax[0].imshow(self.image, cmap=cm.gray)
            ax[0].set_title('Original image')
            self.__plt_imshow_overlay(skeleton1, axes=ax[1], title='Skeleton after closing')
            self.__plt_imshow_overlay(skeleton2, axes=ax[2], title='Skeleton after area closing')
            self.__plt_imshow_overlay(noise_reduced, axes=ax[3], title='Small object removal')
            self.__plt_imshow_overlay(edges, axes=ax[4], title='Canny')
            self.__plt_imshow_overlay(np.logical_xor(edges,closed1), axes=ax[5], title='Dismissed closing')
            self.__plt_imshow_overlay(np.logical_xor(skeleton1,closed2), axes=ax[6], title='Dismissed area closing')
            self.__plt_imshow_overlay(np.logical_xor(skeleton2,noise_reduced), axes=ax[7], title='Dismissed small object')
        return noise_reduced
            

    def classifier(self, debug = False):
        img = np.reshape(self.image[self.mask],(-1,self.image.shape[1]))


        # Build an array of labels for training the segmentation.
        # Here we use rectangles but visualization libraries such as plotly
        # (and napari?) can be used to draw a mask on the image.
        training_labels = np.zeros_like(img, dtype=np.uint8)
        training_labels[:32] = 1
        #training_labels[:170, :400] = 1
        training_labels[90:250, 125:860] = 2
        training_labels[300:550, 380:630] = 2
        training_labels[660:] = 3
        #training_labels[150:200, 720:860] = 4

        sigma_min = 0.1
        sigma_max = 100
        features_func = partial(feature.multiscale_basic_features,
                                intensity=True, edges=True, texture=True,
                                sigma_min=sigma_min, sigma_max=sigma_max, num_sigma=20,
                                multichannel=False, num_workers=None)
        #features = features_func(img)
        #selem = disk(10)
        #eroded = erosion(img, selem)
        #features = diameter_closing(img,100,connectivity=10)
        edges = canny(self.image, sigma=0.9, mask=self.mask)

        #features = morphology.area_closing(edges, area_threshold=1000, connectivity=1, parent=None, tree_traverser=None)
        features = morphology.closing(edges)
        skeleton = morphology.skeletonize(features)
        features2 = morphology.area_closing(skeleton, area_threshold=1000)
        skeleton2 = morphology.skeletonize(features2)

        print(features.shape)
        plt.figure()
        plt.imshow(edges, cmap=cm.gray)
        plt.title('Canny')
        plt.figure()
        plt.imshow(np.logical_not(np.logical_xor(edges,features)), cmap=cm.gray)
        plt.title('Dismissed with closing')
        plt.figure()
        plt.imshow(features, cmap=cm.gray)
        plt.title('Canny-closing')
        plt.figure()
        plt.imshow(skeleton, cmap=cm.gray)
        plt.title("Skeleton")
        plt.figure()
        plt.imshow(features2, cmap=cm.gray)
        plt.title("Closing2")
        plt.figure()
        self.__plt_imshow_overlay(skeleton2, title='Skeleton2')


        """
        clf = RandomForestClassifier(n_estimators=50, n_jobs=-1,
                                    max_depth=10, max_samples=0.05)
        clf = future.fit_segmenter(training_labels, features, clf)
        result = future.predict_segmenter(features, clf)

        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(9, 4))
        ax[0].imshow(segmentation.mark_boundaries(img, result, mode='thick'))
        ax[0].contour(training_labels)
        ax[0].set_title('Image, mask and segmentation boundaries')
        ax[1].imshow(result)
        ax[1].set_title('Segmentation')
        fig.tight_layout()"""

    def lines_h_all(self, debug = False):
        """Detect all horizontal lines in the image

        Keyword arguments:
        debug: overlay original image and lines found (default: False)
        """
        #hough transform
        angle = 90 # 90 degrees = horizontal line
        dAngle = 10 # delta around which to search
        angles = np.linspace((angle-dAngle)*(np.pi / 180), (angle+dAngle)*(np.pi / 180), 500)
        h, thetas, d = hough_line(self.edgesHard, theta=angles)
        self.lines_h_all_hough_peaks = hough_line_peaks(h, thetas, d, num_peaks=3)

        if debug:
            plt.figure()
            plt.imshow(self.image, cmap=cm.gray)
             
            origin = np.array((0, self.image.shape[1]))
            for _, theta, dist in zip(*self.lines_h_all_hough_peaks):
                y0, y1 = (dist - origin * np.cos(theta)) / np.sin(theta)
                plt.plot(origin, (y0, y1), '-r')
            plt.xlim(origin)
            plt.ylim((self.image.shape[0], 0))
            plt.title('Detected lines')
            plt.tight_layout()

    def silicon_baseline(self, debug = False):
        """Detect the silicon interface from which we will perform calculations

        Keyword arguments:
        debug: overlay original image and line found (default: False)
        """
        offset = 10 #the distance in pixels from the line to calculate the score
        origin = np.array((0, self.image.shape[1]))
        x=np.arange(0,self.image.shape[1])
        scorePlus = []
        scoreMinus = []
        for _, theta, dist in zip(*self.lines_h_all_hough_peaks):
            y0, y1 = (dist - origin * np.cos(theta)) / np.sin(theta)
            #checking for bounds
            yPlus = np.minimum(np.around(y1+(y1-y0)/(origin[1]-origin[0])*(x-origin[1])).astype(int)+offset,self.image.shape[0])
            yMinus = np.maximum(np.around(y1+(y1-y0)/(origin[1]-origin[0])*(x-origin[1])).astype(int)-offset,0)
            scorePlus.append(np.mean(self.image[yPlus, x]))
            scoreMinus.append(np.mean(self.image[yMinus, x]))
                
        print(f"Score plus is {scorePlus} and Score minus is {scoreMinus}")
        if debug:
            plt.figure()
            plt.imshow(self.image, cmap=cm.gray)
             
            origin = np.array((0, self.image.shape[1]))
            for _, theta, dist in zip(*self.lines_h_all_hough_peaks):
                y0, y1 = (dist - origin * np.cos(theta)) / np.sin(theta)
                plt.plot(origin, (y0, y1), '-r')
            plt.xlim(origin)
            plt.ylim((self.image.shape[0], 0))
            plt.title('Detected lines')
            plt.tight_layout()

    def lines_h(self, debug = False, number = 2):
        """Detect horizontal lines

        Keyword arguments:
        debug: overlay original image and lines found (default: False)
        number: the number of horizontal lines to detect
        """
        angle = 90 # 90 degrees = horizontal line
        dAngle = 10 # delta around which to search
        angles = np.linspace((angle-dAngle)*(np.pi / 180), (angle+dAngle)*(np.pi / 180), 500)

        sigmas = np.geomspace(20,0.1,50)
        print(sigmas)
        optimalSigma = 6.0
        for sigma in sigmas:
            edges = canny(self.image,sigma=sigma, low_threshold=None, high_threshold=None, mask=self.mask, use_quantiles=False)
            #hough transform
            h, thetas, d = hough_line(edges, theta=angles)
            lines_h_all_hough_peaks = hough_line_peaks(h, thetas, d)
            print(len(lines_h_all_hough_peaks[0]))
            if(len(lines_h_all_hough_peaks[0])>=number):
                optimalSigma=sigma
                break
        edges = canny(self.image,sigma=optimalSigma, low_threshold=None, high_threshold=None, mask=self.mask, use_quantiles=False)
        h, thetas, d = hough_line(edges, theta=angles)
        lines_h_all_hough_peaks = hough_line_peaks(h, thetas, d,num_peaks=number)

        if debug:
            plt.figure()
            self.__plt_imshow_overlay(edges, title = f"With sigma = {optimalSigma}")
            origin = np.array((0, self.image.shape[1]))
            for _, theta, dist in zip(*lines_h_all_hough_peaks):
                y0, y1 = (dist - origin * np.cos(theta)) / np.sin(theta)
                plt.plot(origin, (y0, y1), '-r')
            plt.xlim(origin)
            plt.ylim((self.image.shape[0], 0))
            plt.tight_layout()

    def analyze(self, analyses = None):
        """Analyze the image.

        Keyword arguments: 
        analyses: a list of analysis to perform on an individual image. Each analysis can 
        be one of 'cavity_xy', 'cavity_y', 'thickness_xy', "thickness_y' (default: None).
        """
        pass