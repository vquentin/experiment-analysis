# Copyright (c) 2021, Quentin Van Overmeere
# Licensed under MIT License

import json
import numpy as np
from tifffile import TiffFile as tf
from matplotlib import pyplot as plt


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
    debug: a flag to write the metadata to a file and plot the image
    """
    def __init__(self, filePath = None, debug = False):
        try:
            with tf(filePath) as image:
                if not hasattr(image, 'sem_metadata'):
                    raise Exception("Image is not a Zeiss image")
                for page in image.pages:
                    self.image = page.asarray()
                self.metaDataFull = image.sem_metadata
                self.parse_metadata()
                self.mask()
                if debug:
                    #print all the metadata in a file
                    metadataFile = open(f"{filePath}_meta.txt", "w")
                    print(json.dumps(self.metaDataFull, indent = 4), file=metadataFile)
                    metadataFile.close()
                    #plot stuff
                    self.plot_image_raw()
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
        statistics: a flag to show line-by-line statistics
        """
        fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(12,6), sharey=True, gridspec_kw={'width_ratios': [1024, 255]})
        ax[0].imshow(self.image, cmap=cm.gray)
        ax[0].set_title('Original image')
        rows = np.arange(0, self.image.shape[0])
        ax[1].plot(self.image.min(axis=1), rows, '-b', label='Min.')
        ax[1].plot(self.image.max(axis=1), rows, '-r', label='Max.')
        ax[1].plot(self.image.mean(axis=1), rows, '-g', label='Mean')
        ax[1].plot(np.median(self.image,axis=1), rows, '-k', label='Median')
        ax[1].legend()
        ax[1].set_title('Statistics')
        plt.show()

    def mask_image(self):
        """Creates a masked image for the bottom portion where no scanning was done.
        If a banner is present, it will be masked.

        """
        minThreshold = 0
        maxThreshold = 253
        meanThresholdLow = 147
        meanThresholdHigh = 178
        medianThresholdLow = 150
        medianThresholdHigh = 210

        line_zeiss_logo = 592

        lineMask_min = image.min(axis=1)
        lineMask_max = image.max(axis=1)
        lineMask_mean = image.mean(axis=1)
        lineMask_median = np.median(image,axis=1)

        mask_lines = ((lineMask_min == minThreshold) * (lineMask_max > maxThreshold) * 
                    (lineMask_mean > meanThresholdLow) * (lineMask_mean < meanThresholdHigh) * 
                    (lineMask_median > medianThresholdLow) * (lineMask_median < medianThresholdHigh))
                    

        first_mask_line = np.argmax(mask_lines)
        if first_mask_line == 0:
            first_mask_line = line_zeiss_logo
        mask_lines[first_mask_line:] = True
        mask = np.tile(mask_lines,[image.shape[1],1]).T
        return mask
"""
mask_noise = mask_noise(image)
image_original = image.copy()

image[mask_noise] = 255
#remove Zeiss banner
#image = crop(image, ((0, 95), (0, 0)), copy=False)
#TODO remove noisy parts of image
#scan rows until it's random?



#edge detection
edges = canny(image,sigma=1.0, low_threshold=None, high_threshold=None, mask=None, use_quantiles=False)

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