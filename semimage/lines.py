# Copyright (c) 2021, Quentin Van Overmeere
# Licensed under MIT License

import numpy as np
import math
import logging
from skimage import draw
from skimage import transform
from semimage.feature_test import FeatureTest
from matplotlib import pyplot as plt


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class Line(object):
    def __init__(self, side=None, angle=None, dist=None, image=None):
        self.side = side
        self.image = image
        self._angle = angle
        self._dist = dist
        self.rowsColsFromHough(angle=angle, dist=dist)

    def rowsColsFromHough(self, angle=None, dist=None):
        """ Returns the row, column pairs for the specified Hough line

        Keyword arguments:
        angle: angle in radians
        dist: distance in pixels
        """
        if angle is None:
            angle = self._angle
        if dist is None:
            dist = self._dist
        try:
            r0 = int(round(dist/math.sin(angle)))
            c0 = 0
            r1 = int(round((dist-self.image.shape[1]*math.cos(angle))/math.sin(angle)))
            c1 = self.image.shape[1]
        except Exception as e:
            # handle the edge case of vertical line
            log.dbug(e)
            r0 = 0
            c0 = dist
            r1 = self.image.shape[0]
            c1 = dist
        finally:
            row, col = draw.line(r0, c0, r1, c1)
            # clip row, col to image bounds
            inImage = np.logical_and.reduce([row >= 0, row < self.image.shape[0], col >= 0, col < self.image.shape[1]])
            self.row = row[inImage]
            self.col = col[inImage]
            if self.col.size != 0:
                self.endPoints = np.array([[self.row[0], self.col[0]], [self.row[-1], self.col[-1]]])
            else:
                self.endPoints = np.empty(2, dtype=int)

    @property
    def plotPoints(self):
        return [self.col[0], self.col[-1]], [self.row[0], self.row[-1]]

    @property
    def intensity(self):
        return self.image[self.row, self.col]

    def linesOffset(self, distance=0):
        """Return two lines at distance from the current line. side attribute corresponds to side conventions relative to the current line.

        Keyword arguments:
        distance: the distance from the line in pixels
        """
        return (Line(side=math.floor(self.side/2)*2+1, angle=self._angle, dist=self._dist-distance, image=self.image),
                Line(side=math.floor(self.side/2)*2, angle=self._angle, dist=self._dist+distance, image=self.image))

    def classify(self, debug=False):
        """Return a string guessing the line type based on its features.
        Currently supports isCavity, isNothing, TODO: support for a straight interface.

        Keyword arguments:
        debug: a flag to show diagnostics (passed down).

        Returns a dictionary describing the cavity
        """
        # initialize classification
        feature = FeatureTest(self, debug=debug)
        if feature.assessCavity():
            return 'isCavity'
        elif feature.assessLine():
            return 'isLine'
        else:
            return 'isNothing'

    def distToEdge(self, edge, debug=False):
        """Return euclidian distance to edge vs position along the line.

        edge is a binary image array
        Keyword arguments:
        debug: a flag to show diagnostics

        Returns a tuple of numpy arrays with distance to edge along the line vs line pixel, without NaN values.
        """
        # rotate line and edge images
        angle = -(90-math.degrees(self._angle))
        lineImage = np.zeros_like(self.image, dtype=bool)
        lineImage[self.row, self.col] = True
        lineImageRotated = transform.rotate(lineImage, angle, resize=True, center=(0, 0))
        edgeRotated = transform.rotate(edge, angle, resize=True, center=(0, 0))
        # assume no edge is found by default
        dist = np.full_like(lineImageRotated[0, :], np.nan, dtype=np.float64)
        rowLineR = np.mean(np.nonzero(lineImageRotated)[0])
        rowsEdgeR = np.argmax(edgeRotated, axis=0)
        mask = rowsEdgeR == 0
        dist = rowsEdgeR-rowLineR
        colsR = np.arange(0, edgeRotated.shape[1])
        if debug:
            plt.figure()
            plt.plot(colsR[~mask], dist[~mask], '-k')
        return (colsR[~mask], dist[~mask])

    def subtractIntensity(self, line):
        """Return a tuple of intensity values subtracted from the line vs. position along the line

        line is another Line object.
        """
        # rotate both lines and their associated images

        # mask where the lines do not pass

    def pointProjection(self, p):
        """Return the projection of p on current line row, col

        p is a numpy array of point coordinates (row, col)
        Based on https://stackoverflow.com/a/61342198/13969506
        """
        ap = p - self.endPoints[0]
        ab = self.endPoints[1] - self.endPoints[0]
        t = np.dot(ap, ab) / np.dot(ab, ab)
        # if you need the closest point belonging to the segment
        t = max(0, min(1, t))
        return np.round(self.endPoints[0] + t * ab).astype(int)

    def lineProjection(self, line):
        """Return the intensity along the largest line that can be projected on current line and given line.
        Return a ndarray of intensity
        """
        p0 = self.pointProjection(line.endPoints[0])
        p1 = self.pointProjection(line.endPoints[1])
        row, col = draw.line(p0[0], p0[1], p1[0], p1[1])
        # type conversion to avoid issues when taking difference
        return self.image[row, col].astype(np.int16)
