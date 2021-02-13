# Copyright (c) 2021, Quentin Van Overmeere
# Licensed under MIT License

import numpy as np
import math
import logging
from skimage import draw
from skimage import transform
import semimage.config as config


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class Line(object):
    def __init__(self, side=None, angle=None, dist=None, image=None):
        self.side = side
        self.__image = image
        self.__angle = angle
        self.__dist = dist
        self.rows, self.columns = self.__to_coordinates()
        self.end_points = np.array([self.rows[-1], self.columns[-1]],
                                   [self.rows[0], self.columns[0]])
        log.debug(f"Image {self.__image.image_name}: \
            line created on side {self.side}.")

    def __repr__(self):
        return f"{self.__class__.__name__}(side={self.side}, \
                 angle={self.__angle}, dist={self.__dist}, \
                 image={self.__image})"

    def __to_coordinates(self):
        """ Returns the row, column pairs for the specified line in Hough
        coordinates.
        """
        try:
            r0 = int(round(self.__dist/math.sin(self.__angle)))
            c0 = 0
            r1 = int(round((
                self.__dist - self.__image.shape[1]*math.cos(self.__angle))
                / math.sin(self.__angle)))
            c1 = self.__image.shape[1]
        except ZeroDivisionError:
            # handle the edge case of vertical line
            r0 = 0
            c0 = self.__dist
            r1 = self.__image.shape[0]
            c1 = self.__dist
        finally:
            rows, cols = draw.line(r0, c0, r1, c1)
        # clip row, col to image bounds
        in_image = np.logical_and.reduce(
            [rows >= 0, rows < self.__image.shape[0], cols >= 0,
             cols < self.__image.shape[1]])
        return rows[in_image], cols[in_image]

    @property
    def plot_points(self):
        return ([self.columns[0], self.columns[-1]],
                [self.rows[0], self.rows[-1]])

    @property
    def intensity(self):
        return self.__image[self.rows, self.columns]

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
                  c=np.array(config.colors[self.side]/255))

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
        return self.__image[row, col]
