import numpy as np
import numpy.ma as ma
from matplotlib import pyplot as plt
import pwlf
import math
import statistics
from sklearn.linear_model import LinearRegression
from skimage import draw

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
        """Guess if the current line is a cavity.
        
        Returns a score (high value = more odds it's a cavity), and the side it's on
        """
        nSegCavity = 3
        relativeError = 0.33
        relativeErrorSym = 0.52
        absoluteError = 5
        R2min = 0.99 #threshold to apply a 3-piece linear fit
        straigthSlopeThreshold = 26.68
        lineMinus, linePlus = self.linesOffset(distance=3)

        #initilize classification
        isCavity = False
        side = -1
        
        if lineMinus.col.size is not 0 and linePlus.col.size is not 0:
            #perform piece-wise fit of intensity
            x1 = lineMinus.col
            x2 = linePlus.col
            y1 = np.cumsum(lineMinus.intensity)
            y2 = np.cumsum(linePlus.intensity)

            # fit the data
            pwlfLinePlus = pwlf.PiecewiseLinFit(x1, y1)
            pwlfLineMinus = pwlf.PiecewiseLinFit(x2, y2)
            res1 = pwlfLinePlus.fit(nSegCavity)
            res2 = pwlfLineMinus.fit(nSegCavity)
            straigth1 = LinearRegression(fit_intercept=False).fit(x1.reshape((-1, 1)),y1)
            straigth2 = LinearRegression(fit_intercept=False).fit(x2.reshape((-1, 1)),y2)
            slope1 = pwlfLinePlus.calc_slopes()
            slope2 = pwlfLineMinus.calc_slopes()
            slope1s = straigth1.coef_
            slope2s = straigth2.coef_

            #check if cavity likely
            line1IsStraight = True
            line2IsStraight = True
            
            print(f"R2 value line 1: {straigth1.score(x1.reshape((-1, 1)), y1)}, slope: {slope1s}")
            print(f"R2 value line 2: {straigth2.score(x2.reshape((-1, 1)), y2)}, slope: {slope2s}")
            
            if straigth1.score(x1.reshape((-1, 1)), y1) < R2min and slope1s > straigthSlopeThreshold:
                line1IsStraight = False
                print("Line 1 is not straight")
            if straigth2.score(x2.reshape((-1, 1)), y2) < R2min and slope2s > straigthSlopeThreshold:
                line2IsStraight = False
                print("Line 2 is not straight")
            
            if line1IsStraight and not line2IsStraight:
                line2Symetrical = math.isclose(slope2[0], slope2[2], rel_tol = relativeErrorSym, abs_tol = absoluteError)
                if line2Symetrical:
                    line2cavity = math.isclose(slope2[1], slope1s, rel_tol = relativeError, abs_tol = straigthSlopeThreshold) and self.side is math.floor(self.side/2)*2+1
                    if line2cavity:
                        isCavity = True
                        side = math.floor(self.side/2)*2+1
            elif line2IsStraight and not line1IsStraight:
                line1Symetrical = math.isclose(slope1[0], slope1[2], rel_tol = relativeErrorSym, abs_tol = absoluteError)
                if line1Symetrical:
                    line1cavity = math.isclose(slope1[1], slope2s, rel_tol = relativeError, abs_tol = straigthSlopeThreshold) and self.side is math.floor(self.side/2)*2
                    if line1cavity:
                        isCavity = True
                        side = math.floor(self.side/2)*2

            if debug:
                print(slope1)
                print(slope2)
                plt.figure(figsize=(4,4))
                x1Hat = np.arange(min(x1), max(x1))
                x2Hat = np.arange(min(x2), max(x2))
                y1Hat = pwlfLinePlus.predict(x1Hat)
                y2Hat = pwlfLineMinus.predict(x2Hat)
                plt.plot(x1, y1, ',k', x2, y2, ',b', x1Hat, y1Hat, '-k', x2Hat, y2Hat, '-b')
                plt.title(f"On side {self.side}, Cavity: {isCavity}, side: {side}")
            
            return {'isCavity': isCavity}