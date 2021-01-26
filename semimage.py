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
    Intended for Zeiss Scanning Electron Microscopes.

    Keyword arguments:
    filePath: a path to a file name (default: None, returns an error)
    debug: a flag to write the metadata to a file and plot the image (default: False)
    """
    def __init__(self, filePath = None, debug = False):
        try:
            with tf(filePath) as image:
                self.imageName = Path(filePath).name
                if not hasattr(image, 'sem_metadata'):
                    raise Exception("Image is not a Zeiss image")
                for page in image.pages:
                    self.image = page.asarray()
                self.metaDataFull = image.sem_metadata
                self.parse_metadata()
                self.mask(debug=debug)
                self.canny(debug=debug)
                if debug:
                    #print all the metadata in a file
                    metadataFile = open(f"{filePath}_meta.txt", "w")
                    print(json.dumps(self.metaDataFull, indent = 4), file=metadataFile)
                    metadataFile.close()
                    #plot stuff
                    self.plot_image_raw()
                plt.show()
        except Exception as e:
            print(f"An error occurred while trying to load {filePath}")
            print(e)

    def parse_metadata(self):
        """Makes the metadata accessible directly in the instance of the object        
        """
        self.lineCount = self.metaDataFull["ap_line_counter"][1]
        self.pixelSize = self.metaDataFull["ap_image_pixel_size"][1]
        
        self.beamXOffset = self.metaDataFull["ap_beam_offset_x"][1]
        self.beamYOffset = self.metaDataFull["ap_beam_offset_y"][1]
        self.stageX = self.metaDataFull["ap_stage_at_x"][1]
        self.stageY = self.metaDataFull["ap_stage_at_y"][1]

    def plot_image_raw(self, statistics=True):
        """Plots the image in a new window, without any treatment.

        Keyword arguments:
        statistics: a flag to show line-by-line statistics (default: True)
        """
        fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize=(12,6), sharey=True, gridspec_kw={'width_ratios': [1024, 255]})
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
        plt.tight_layout()

    def mask(self, debug = False):
        """Creates a mask and a masked image for the bottom portion where no scanning was done.
        If a banner is present, the banner and lines below will be masked.

        Keyword arguments:
        debug: shows the picture of the masked image (default: False)
        """
        lineBanner = 676

        lineMask_min = self.image.min(axis=1)
        lineMask_max = self.image.max(axis=1)
        maskLines = np.ones(lineMask_min.shape, dtype=bool)
        maskFirstLine = self.lineCount
        if lineMask_min[lineBanner] == 0 and lineMask_max[lineBanner+1] == 255:
            print(f"Image {self.imageName} has banner")
            maskFirstLine = min(lineBanner, maskFirstLine)
        
        maskLines[maskFirstLine:] = False
        self.mask = np.tile(maskLines,[self.image.shape[1],1]).T
        self.maskedImage = self.image.copy()
        self.maskedImage[~self.mask] = 0
        if debug:
            plt.figure()
            plt.imshow(self.maskedImage, cmap=cm.gray)
            plt.title('Masked image')
            plt.tight_layout()

    def canny(self, debug = False):
        """Creates a canny edge filter for the masked image.

        Keyword arguments:
        debug: overlays the edges in red on the image (default: False)
        """
        self.edges = canny(self.image,sigma=1.0, low_threshold=None, high_threshold=None, mask=self.mask, use_quantiles=False)
        if debug:
            fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize=(12,6), sharey=True)
            ax = axes.ravel()
            ax[0].imshow(self.image, cmap=cm.gray)
            ax[0].set_title('Original image')
            ax[1].imshow(self.edges, cmap=cm.gray)
            ax[1].set_title('Detected edges')
            plt.tight_layout()



"""
#hough transform
tested_angles = np.linspace(80*(np.pi / 180), 100*(np.pi / 180), 360)
h, theta, d = hough_line(edges, theta=tested_angles)

# Generating figure
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
ax = axes.ravel()

ax[0].imshow(image_original, cmap=cm.gray)
ax[0].set_title('Input image')
#ax[0].set_axis_off()

ax[1].imshow(edges, cmap=cm.gray)
ax[1].set_title('Canny edges')
""""""
ax[2].imshow(np.log(1 + h),
             extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
             cmap=cm.gray, aspect=1/1.5)
ax[2].set_title('Hough transform')
ax[2].set_xlabel('Angles (degrees)')
ax[2].set_ylabel('Distance (pixels)')
ax[2].axis('image')
""""""
ax[2].imshow(image, cmap=cm.gray)
origin = np.array((0, image.shape[1]))
for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
    y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
    ax[2].plot(origin, (y0, y1), '-r')
ax[2].set_xlim(origin)
ax[2].set_ylim((image.shape[0], 0))
ax[2].set_axis_off()
ax[2].set_title('Detected lines')

plt.tight_layout()
plt.show()
"""