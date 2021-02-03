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
            #handle the special case of vertical line
            log.dbug(e)
            r0 = 0
            c0 = dist
            r1 = self.image.shape[0]
            c1 = dist
        finally:
            row, col = draw.line(r0, c0, r1, c1)
            #clip row, col to image bounds
            inImage = np.logical_and.reduce([row >= 0, row < self.image.shape[0], col >= 0, col < self.image.shape[1]])
            self.row = row[inImage]
            self.col = col[inImage]

    @property
    def plot_points(self):
        return [self.col[0], self.col[-1]], [self.row[0], self.row[-1]]

    @property
    def intensity(self):
        return self.image[self.row, self.col]

    def linesOffset(self, distance = 0):
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
        #initilize classification
        feature = FeatureTest(self)
        if (feature.assessCavity(debug=False)):
            return 'isCavity'
        else:
            return 'isNothing'

    def distToEdge(self, edge, debug=False):
        """Return distance to edge vs position along the line.
        
        Keyword arguments:
        debug: a flag to show diagnostics
        
        Returns a tuple of numpy arrays with distance to edge along the line, without NaN values.
        """
        #TODO: current implementation is slow. Alternative would be to rotate the edge image and search along each column. Also use np
        dist=np.full_like(self.col, np.nan, dtype=np.float64)
        lineImage = np.zeros_like(self.image, dtype=bool)
        lineImage[self.row, self.col] = True
        lineImageRotated = transform.rotate(lineImage, -(90-math.degrees(self._angle)), resize=True, center=(0,0))
        rowsLine = np.argmax(lineImageRotated, axis=0)
        maskLine = rowsLine==0

        edgeRotated = transform.rotate(edge, -(90-math.degrees(self._angle)), resize=True, center=(0,0))
        rowsEdge = np.argmax(edgeRotated, axis=0)
        maskEdge = rowsEdge==0
        dist = rowsEdge-rowsLine
        mask = np.logical_or(maskLine, maskEdge)
        print(mask.shape)
        print(dist.shape)
        dist = dist[~mask]
        if debug:
            plt.figure()
            plt.plot(self.col, dist, '-k')
        return (np.arange(0, dist.shape[0]), dist)