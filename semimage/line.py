# Copyright (c) 2021, Quentin Van Overmeere
# Licensed under MIT License

import numpy as np
import math
import logging
from skimage import draw
from skimage import transform
from skimage import filters
import semimage.config as config
from matplotlib import pyplot as plt


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class Line(object):
    """
    Creates a line with useful methods.

    Keyword arguments:
    side: the side of the image the line was found at
    angle: the Hough angle (from Hough line detection)
    dist: the Hough dist (from Hough line detection)
    image: a sem_image object
    """
    def __init__(self, side=None, angle=None, dist=None, image=None):
        self.side = side
        self.__image = image
        self.__angle = angle
        self.__dist = dist
        self.rows, self.columns = self.__to_coordinates()
        self.end_points = np.array([[self.rows[-1], self.columns[-1]],
                                   [self.rows[0], self.columns[0]]])
        log.debug(f"Image {self.__image.image_name}: "
                  f"Line created on side {self.side}.")

    def __repr__(self):
        return (f"{self.__class__.__name__}(side={self.side}, "
                f"angle={self.__angle}, dist={self.__dist}, "
                f"image={self.__image})")

    def __to_coordinates(self):
        """ Returns the row, column pairs for the specified line in Hough
        coordinates.
        """
        try:
            r0 = int(round(self.__dist/math.sin(self.__angle)))
            c0 = 0
            r1 = int(round((
                self.__dist - self.__image.image.shape[1]*math.cos(self.__angle))
                / math.sin(self.__angle)))
            c1 = self.__image.image.shape[1]
        except ZeroDivisionError:
            # handle the edge case of vertical line
            r0 = 0
            c0 = self.__dist
            r1 = self.__image.image.shape[0]
            c1 = self.__dist
        finally:
            rows, cols = draw.line(r0, c0, r1, c1)
        # clip row, col to image bounds
        in_image = np.logical_and.reduce(
            [rows >= 0, rows < self.__image.image.shape[0], cols >= 0,
             cols < self.__image.image.shape[1]])
        return rows[in_image], cols[in_image]

    @property
    def plot_points(self):
        return ([self.columns[0], self.columns[-1]],
                [self.rows[0], self.rows[-1]])

    @property
    def _slope(self):
        return ((self.rows[0]-self.rows[-1])
                / (self.columns[0]-self.columns[-1]))

    @property
    def intensity(self):
        return self.__image.image[self.rows, self.columns]

    def lines_at_offset(self, distance=0):
        """Return two lines at distance from the current line.

        Keyword arguments:
        distance: the distance from the line (in pixels)
        """
        side_minus = math.floor(self.side/2)*2+1
        side_plus = math.floor(self.side/2)*2
        line_minus = Line(side=side_minus, angle=self.__angle,
                          dist=self.__dist-distance, image=self.__image)
        line_plus = Line(side=side_plus, angle=self.__angle,
                         dist=self.__dist+distance, image=self.__image)
        return line_minus, line_plus

    def show(self, axes):
        """Plot the line on the specified figure"""
        axes.plot(*self.plot_points, '-',
                  c=np.array(config.colors[self.side])/255)

    def point_projected(self, p):
        """Return the projection of p on current line row, col

        p is a numpy array of point coordinates (row, col)
        Based on https://stackoverflow.com/a/61342198/13969506
        """
        ap = p - self.end_points[0]
        ab = self.end_points[1] - self.end_points[0]
        t = np.dot(ap, ab) / np.dot(ab, ab)
        # if you need the closest point belonging to the segment
        t = max(0, min(1, t))
        return np.round(self.end_points[0] + t * ab).astype(int)

    def line_projected(self, line):
        """Return the intensity along the largest line that can be projected
        on current line and given line.
        Warning ! taking the difference can result in casting errors

        Return a ndarray of intensity
        """
        p0 = self.point_projected(line.end_points[0])
        p1 = self.point_projected(line.end_points[1])
        row, col = draw.line(p0[0], p0[1], p1[0], p1[1])
        return self.__image.image[row, col]

    def background_on_side(self):
        """
        Return True if image intensity on same side than line is likely to
        correspond to a uniform background.
        """
        frac_bgd_threshold = 0.93
        dark_bgd_threshold = 181
        clear_bgd_threshold = 2.1e5

        if not self.in_masked_image():
            return False
        # dark background
        img_on_side = self.image_on_side()
        img_threshold = filters.threshold_otsu(
            image=self.__image.image[self.__image.mask])
        count_bgd = img_on_side[~np.isnan(img_on_side)] < img_threshold
        frac_bgd = np.count_nonzero(count_bgd)/np.size(count_bgd)
        if (frac_bgd > frac_bgd_threshold
                and img_threshold < dark_bgd_threshold):
            return True
        # clear background
        bgd_cols = np.sum([[np.nanmedian(img_on_side, axis=0)],
                           [np.nanmean(img_on_side, axis=0)],
                           [np.nanmax(img_on_side, axis=0)]], axis=-1)
        if np.all(bgd_cols > clear_bgd_threshold):
            return True
        return False

    def in_masked_image(self):
        """Return True if line contained in masked image."""
        line_count = np.sum(self.__image.mask.any(1))
        return np.all(self.rows <= line_count)

    def image_on_side(self):
        """
        Return a masked image with only what's on the same side than the
        line.
        """
        # TODO adapt for non-horizontal lines
        n_rows, n_cols = self.__image.image.shape
        rows, cols = np.ogrid[0:n_rows, 0:n_cols]
        if self.side == 0:
            mask = rows < self.rows[0] + (self._slope)*cols
        elif self.side == 1:
            mask = rows > self.rows[0] + (self._slope)*cols
        image = self.__image.image.astype(float)
        image[~self.__image.mask] = np.nan
        image[mask] = np.nan
        return image
