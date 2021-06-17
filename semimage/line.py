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
    edges: the edge matrix from which the line was detected
    """
    def __init__(self, side=None, angle=None, dist=None, image=None,
                 edges=None):
        self.side = side
        self.__image = image
        self.__angle = angle
        self.__dist = dist
        self.__edges = edges
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
                self.__dist
                - self.__image.image.shape[1]*math.cos(self.__angle))
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
        return self.__image.image[self.rows, self.columns].astype(np.int16)

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

    def show2(self, **kwargs):
        """Plot the line on the specified figure"""
        plt.plot(*self.plot_points, '-', **kwargs)

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
        frac_bgd_threshold = 0.5
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

    def intensity_difference_offset(self, distance):
        """
        Return the intensity difference between line at other side of current
        line, and line at same side, separated by 2*distance.
        """
        line_minus, line_plus = self.lines_at_offset(distance=distance)
        if self.side == 0:
            return line_plus.intensity-line_minus.intensity
        elif self.side == 1:
            return line_minus.intensity - line_plus.intensity

    def total_intensity_difference_offset(self, distance):
        """
        Return the sum of intensity differences for all lines up to distance.
        """
        return np.sum((self.intensity_difference_offset(dist) for dist
                       in range(distance)))

    def classify(self, distance):
        """Return a string corresponding to a candidate line type."""
        sum_intensity_diff = self.total_intensity_difference_offset(distance)
        if self.cavity_score(sum_intensity_diff, distance):
            log.debug('Line is a cavity')
            return 'Cavity'
        elif self.si_void_interface_score(sum_intensity_diff, distance):
            log.debug('Line is Si/void interface')
            return 'Si/void interface'
        elif self.porous_void_interface_score(sum_intensity_diff, distance):
            log.debug("Line is Porous Si/void interface")
            return 'Porous Si/void interface'

    def si_void_interface_score(self, sum_intensity_diff, distance):
        """Return True if Si/void interface is likely."""
        median_threshold = 100*distance
        log.debug(f"Median sum of diffs: {abs(np.median(sum_intensity_diff))}")
        return abs(np.median(sum_intensity_diff)) > median_threshold

    def porous_void_interface_score(self, sum_intensity_diff, distance):
        """Return True if porous Si/void interface is likely."""
        median_threshold = 22*distance
        return abs(np.median(sum_intensity_diff)) > median_threshold

    def cavity_score(self, sum_intensity_diff, distance):
        """
        Return True if cavity is likely, based on equality of
        intensities in center of line.
        """
        min_frac_cavity = 0.1
        min_dist_border = 10
        zero_treshold = 6*distance

        close_to_zero = np.isclose(sum_intensity_diff, 0, rtol=0,
                                   atol=zero_treshold)
        a = np.diff(np.logical_and(np.roll(close_to_zero, -1, axis=0), 
                                   close_to_zero).astype(np.int16))
        consecutive_zeros_start_index = np.nonzero(a == 1)[0]
        try:
            consecutive_zeros_length = (np.nonzero(a == -1)[0] 
                                    - consecutive_zeros_start_index)
        except ValueError:
            #raised if zeros at edges
            return False
        if consecutive_zeros_length.size == 0:
            return False
        longest_seq_idx = np.argmax(consecutive_zeros_length)
        cons_zeros_is_long = (np.max(consecutive_zeros_length)
                              > min_frac_cavity*self.__image.image.shape[1])
        cons_zeros_not_at_edge = (self.__image.image.shape[1]-min_dist_border
                                  > np.nonzero(a == -1)[0][longest_seq_idx] 
                                  and consecutive_zeros_start_index[longest_seq_idx] 
                                  > min_dist_border)
        if cons_zeros_is_long and cons_zeros_not_at_edge:
            return True

    def distance_real_units(self, line):
        pass

    def distance_to_edge_um(self, edge, show=False):
        """Return the median and deviation of distances to line point by point."""
        idx = np.column_stack(np.where(edge))
        dist = self.distance_to_points(idx)*self.__image.metadata.pixel_size/1000
        if show:
            plt.figure()
            plt.plot(dist)
        return abs(np.median(dist)), np.std(dist)

    def distance_to_points(self, p):
        """
        Return the Euclidian distance between p and the instance.
        Based on https://stackoverflow.com/a/54442561/13969506 
        """
        # TODO for you: consider implementing @Eskapp's suggestions
        a = self.end_points[0]
        b = self.end_points[1]
        if np.all(a == b):
            return np.linalg.norm(p - a, axis=1)
        d = np.divide(np.cross(b-a, p-a), np.linalg.norm(b-a))
        return d

    def distance_to_edge_exclude_zero_um(self, edge, show=False):
        """Return the median and deviation of distances to line point by point, excluding zeros."""
        idx = np.column_stack(np.where(edge))
        dist = self.distance_to_points(idx)*self.__image.metadata.pixel_size/1000
        dist_excl_zeros = dist[~np.isclose(dist, 0, rtol=0, atol=0.1)]
        if show:
            plt.figure()
            plt.plot(dist)
        return abs(np.median(dist_excl_zeros)), np.std(dist_excl_zeros)
