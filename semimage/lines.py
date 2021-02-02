import numpy as np
import math
from skimage import draw
from semimage.feature_test import FeatureTest
from matplotlib import pyplot as plt

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
        return (Line(side = math.floor(self.side/2)*2+1, angle = self._angle, dist = self._dist-distance, image = self.image),
                Line(side = math.floor(self.side/2)*2, angle = self._angle, dist = self._dist+distance, image = self.image))

    def classify(self, debug=False):
        """Guess if the current line is representative of a characteristic feature, such as a cavity, or a straight interface.
        
        Keyword arguments:
        debug: a flag to show diagnostics.

        Returns a dictionary describing the cavity
        """
        #initilize classification
        feature = FeatureTest(self)
        cavity = feature.assessCavity(debug=False)
                    
        return {'isCavity': cavity}

    def distToEdge(self, edge, debug=False):
        """Return x,y coordinates of distance to edge.

        Keyword arguments:
        debug: a flag to show diagnostics

        Returns a tuple of numpy arrays with distance to edge along the line.
        """
        dist=[]
        for i, val in np.ndenumerate(self.col):
            orthoLine = Line(side = self.side, angle = self._angle-math.pi/2, dist = val-self.row[i]/math.tan(self._angle), image = edge)
            """if debug and (i[0] == 750):
                plt.figure()
                plt.imshow(edge)
                plt.plot(*orthoLine.plot_points, '-r')
                plt.title(f"line for i={i}")"""
            idx=np.argmax(orthoLine.intensity)
            dist.append(math.sqrt(abs(val-orthoLine.col[idx])**2 + abs(self.row[i]-orthoLine.row[idx])**2))
        if debug:
            plt.figure()
            plt.plot(self.col, dist, '-k')
        return (self.col, dist)

