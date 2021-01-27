# Copyright (c) 2021, Quentin Van Overmeere
# Licensed under MIT License

import json
import numpy as np
from tifffile import TiffFile as tf
from matplotlib import pyplot as plt
from pathlib import Path


from matplotlib import cm
from statistics import mean, median

from skimage import data, io, filters
from skimage.feature import blob_dog, blob_log, blob_doh, canny
from skimage.color import rgb2gray
from skimage.transform import probabilistic_hough_line, hough_line, hough_line_peaks
from skimage.util import crop

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
            self.parse_metadata(metaData = image.sem_metadata, debug = False)
            self.mask(debug = False)
        if debug:
            #plot stuff
            self.plot_image_raw()
        self.canny(debug=True, sigma = 1.0)
        #self.lines_h_all(debug=False)
        #self.lines_h(debug=True)
        #self.silicon_baseline(debug=True)
        plt.show()

    def parse_metadata(self, metaData = None, debug = False):
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

    def canny(self, debug = False, sigma = 1.0):
        """Return a canny edge filter for the masked image.

        Keyword arguments:
        debug: compares the edges and the original image (default: False)
        sigma: override the default sigma value in the canny filter
        """
        edges = canny(self.image,sigma=sigma, low_threshold=None, high_threshold=None, mask=self.mask, use_quantiles=False)
        if debug:
            plt.figure()
            plt.imshow(self.image, cmap=cm.gray)
            plt.imshow(np.stack([edges,np.zeros_like(edges),np.zeros_like(edges),1.0*edges], axis=2))
            plt.title(f"Detected edges with sigma = {sigma}")
            plt.tight_layout()
        return edges

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
            plt.imshow(self.image, cmap=cm.gray)
            plt.imshow(np.stack([edges,np.zeros_like(edges),np.zeros_like(edges),np.ones_like(edges)*0.5], axis=2))
            origin = np.array((0, self.image.shape[1]))
            for _, theta, dist in zip(*lines_h_all_hough_peaks):
                y0, y1 = (dist - origin * np.cos(theta)) / np.sin(theta)
                plt.plot(origin, (y0, y1), '-r')
            plt.xlim(origin)
            plt.ylim((self.image.shape[0], 0))
            plt.title(f"With sigma = {optimalSigma}")
            plt.tight_layout()

    def analyze(self, analyses = None):
        """Analyze the image.

        Keyword arguments: 
        analyses: a list of analysis to perform on an individual image. Each analysis can 
        be one of 'cavity_xy', 'cavity_y', 'thickness_xy', "thickness_y' (default: None).
        """
        pass