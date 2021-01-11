import glob
from math import sqrt
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import cm

from skimage import data, io, filters
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.feature import canny
from skimage.color import rgb2gray
from skimage.transform import probabilistic_hough_line, hough_line, hough_line_peaks
from skimage.util import crop

from mpl_toolkits.axes_grid1 import AxesGrid

#import exifread

#Not working, need something specific to Zeiss
""" # Return Exif tags
f = open('./examples/WF67_1_06.tif', 'rb')
tags = exifread.process_file(f)

# Print the tag/ value pairs
for tag in tags.keys():
    if tag not in ('JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote'):
        print(f"Key: {tag}, value {tags[tag]}") """

#example_file = glob.glob(r"./examples/WF67_1_06.tif")[0]
#example_file = glob.glob(r"./examples/UB19_14.tif")[0]
#example_file = glob.glob(r"./examples/ADB1_28.tif")[0]
example_file = glob.glob(r"./examples/WF56_1_20.tif")[0]


image = io.imread(example_file, as_gray=True)
#remove Zeiss banner
image = crop(image, ((0, 95), (0, 0)), copy=False)


#edges=filters.sobel(image)
edges = canny(image)
#io.imshow(edges)
#io.show()

#hough transform
tested_angles = np.linspace(80*(np.pi / 180), 100*(np.pi / 180), 360)
h, theta, d = hough_line(edges, theta=tested_angles)
# Generating figure 1
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
ax = axes.ravel()

ax[0].imshow(image, cmap=cm.gray)
ax[0].set_title('Input image')
ax[0].set_axis_off()

ax[1].imshow(np.log(1 + h),
             extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
             cmap=cm.gray, aspect=1/1.5)
ax[1].set_title('Hough transform')
ax[1].set_xlabel('Angles (degrees)')
ax[1].set_ylabel('Distance (pixels)')
ax[1].axis('image')

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



lines = probabilistic_hough_line(edges, threshold=100, line_length=700,
                                 line_gap=30)

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image, cmap=cm.gray)
ax[0].set_title('Input image')

ax[1].imshow(edges, cmap=cm.gray)
ax[1].set_title('Canny edges')

ax[2].imshow(edges * 0)
for line in lines:
    p0, p1 = line
    ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
ax[2].set_xlim((0, image.shape[1]))
ax[2].set_ylim((image.shape[0], 0))
ax[2].set_title('Probabilistic Hough')

for a in ax:
    a.set_axis_off()

plt.tight_layout()
plt.show()

##cm_gray = plt.get_cmap('gray')
#plt.imshow(im, cmap=cm_gray)
#plt.show()