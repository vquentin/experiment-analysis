import glob
from math import sqrt
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import cm
from statistics import mean, median
from tifffile import TiffFile as tf

from skimage import data, io, filters
from skimage.feature import blob_dog, blob_log, blob_doh, canny
from skimage.color import rgb2gray
from skimage.transform import probabilistic_hough_line, hough_line, hough_line_peaks
from skimage.util import crop

from mpl_toolkits.axes_grid1 import AxesGrid

import json

from semimage import SEMImage 

#Load example file
#example_file = glob.glob(r"./examples/WF67_1_06.tif")[0]
#example_file = glob.glob(r"./examples/UB19_14.tif")[0]
#example_file = glob.glob(r"./examples/ADB1_28.tif")[0]
#example_file = glob.glob(r"./examples/WF56_1_20.tif")[0]
example_file = glob.glob(r"./examples/ADB1_44.tif")[0]

a = SEMImage(example_file, debug=True)
print(a.lineCount)
"""
#reading the file
#image = io.imread(example_file, as_gray=True)
try:
    with tf(example_file) as t :
        if hasattr(t,'sem_metadata') == True :
            #metadata_file = open("metadata2.txt", "w")
            #print(json.dumps(t.sem_metadata, indent=4),file=metadata_file)
            #metadata_file.close()
            print(len(t.pages))
            print(len(t.series))
            for page in t.pages:
                image = page.asarray()
        else :
            print('there is no Zeiss SEM metadata')
except Exception as e:
    print('error reading file:',example_file)
    print(e)

mask_filter_min = [min(row) for row in image]
mask_filter_max = [max(row) for row in image]
mask_filter_mean = [mean(row) for row in image]
mask_filter_median = [median(row) for row in image]
"""
def mask_noise(image):
    """Return a numpy array mask for portions detected as noisy in Zeiss SEM images.
    If a banner is present, it will be masked.

    Keyword arguments:
    image -- A numpy array representing the image
    """
    debug = True
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
    if debug:
        plt.plot(lineMask_min,'-b', label='Min.')
        plt.plot(lineMask_max,'-r', label='Max.')
        plt.plot(lineMask_mean,'-g', label='Mean')
        plt.plot(lineMask_median, '-k', label='Median')
        plt.plot(mask_lines*255,'--b')
        plt.xlabel("Line")
        plt.legend()

    first_mask_line = np.argmax(mask_lines)
    if first_mask_line == 0:
        first_mask_line = line_zeiss_logo
    mask_lines[first_mask_line:] = True
    if debug:
        print(first_mask_line)
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