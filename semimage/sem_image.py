# Copyright (c) 2021, Quentin Van Overmeere
# Licensed under MIT License

import numpy as np
from tifffile import TiffFile as tf
from matplotlib import pyplot as plt
from matplotlib import cm
from pathlib import Path
import logging

from semimage.sem_metadata import SEMZeissMetadata

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class SEMZeissImage(object):
    """This class represents an image, and its associated properties.
    Upon instanciation, the image is loaded as a numpy array, and its metadata
    is created.
    Intended for images generated with Zeiss Scanning Electron Microscopes.

    Positional arguments:
    file_path: a path to a file name (default: None, throws an error)
    """
    def __init__(self, file_path):
        self.path = Path(file_path)
        with tf(self.path) as image:
            self.image_name = self.path.stem
            if not hasattr(image, 'sem_metadata'):
                raise TypeError("Image is likely not from a Zeiss scanning \
                                 electron microscope")
            self.image = image.pages[0].asarray()
            self.metadata = SEMZeissMetadata(image.sem_metadata, self.path)
            self.mask = self.__mask(self.metadata.line_count)
        log.debug(f"Image {self.image_name} was loaded.")

    def show(self):
        """Plots the image in a new window, without any treatment and show
        line-by-line and statisticals diagnostics.
        """
        _, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 5),
                               gridspec_kw={'width_ratios': [1024, 256, 256]})
        ax = axes.ravel()
        ax[0].imshow(self.image, cmap=cm.gray, vmin=0, vmax=255)
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
            ax[2].hist(self.image[self.mask].ravel(), bins=256)
            ax[2].set_title('Histogram (mask applied)')
        else:
            ax[2].hist(self.image.ravel(), bins=256)
            ax[2].set_title('Histogram (no mask)')
        plt.tight_layout()

    def __mask(self, line_count):
        """Returns a mask for the bottom portion where no scanning was done.
        If a banner is present, the banner and lines below will be masked.
        """
        line_banner = 676  # valid for Zeiss microscope in Winfab
        mask = np.ones(self.image.shape, dtype=bool)
        if (np.amin(self.image[line_banner, :]) == 0
                and np.amax(self.image[line_banner+1, :]) == 255):
            has_banner = True
            mask_first_line = min(line_banner, line_count)
        else:
            has_banner = False
            mask_first_line = line_count
        mask[mask_first_line:, :] = False
        log.debug(f"Mask applied to image {self.image_name}. \
                    Banner {has_banner}.")
        return mask
