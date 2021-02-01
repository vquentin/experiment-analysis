import numpy as np
from matplotlib import pyplot as plt
import pwlf
import math
import statistics
from sklearn.linear_model import LinearRegression

class Line(object):
    def __init__(self, side=None, angle=None, dist=None, image=None):
        self.side = side
        self.image = image
        self._angle = angle
        self._dist = dist

    @property
    def col(self):
        #TODO: need to bind col and rows to the image
        return np.arange(0, self.image.shape[1])

    @property
    def row_float(self):
        #TODO avoid throwing an error if angle = 0 (vertical line)
        #TODO: need to bind col and rows to the image
        return (self._dist - self.col * np.cos(self._angle)) / np.sin(self._angle)

    @property
    def row(self):
        #TODO: need to bind col and rows to the image
        return np.array(self.row_float).astype(int)

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

    def isCavity(self, debug=False):
        """Guess if the current line is a cavity.
        
        Returns a score (high value = more odds it's a cavity), and the side it's on
        """
        nSeg = 3
        relativeError = 0.2
        absoluteError = 5
        R2min = 0.97 #threshold to apply a 3-piece linear fit
        straigthSlopeThreshold = 23.62
        
        lineMinus, linePlus = self.linesOffset(distance=3)
        #perform piece-wise fit of intensity
        x1 = lineMinus.col
        x2 = linePlus.col
        y1 = np.cumsum(lineMinus.intensity)
        y2 = np.cumsum(linePlus.intensity)

        # fit the data
        pwlfLinePlus = pwlf.PiecewiseLinFit(x1, y1)
        pwlfLineMinus = pwlf.PiecewiseLinFit(x2, y2)
        res1 = pwlfLinePlus.fit(nSeg)
        res2 = pwlfLineMinus.fit(nSeg)
        straigth1 = LinearRegression(fit_intercept=False).fit(x1.reshape((-1, 1)),y1)
        straigth2 = LinearRegression(fit_intercept=False).fit(x2.reshape((-1, 1)),y2)
        slope1 = pwlfLinePlus.calc_slopes()
        slope2 = pwlfLineMinus.calc_slopes()
        slope1s = straigth1.coef_
        slope2s = straigth2.coef_

        #check if cavity likely
        line1IsStraight = True
        line2IsStraight = True
        isCavity = False
        side = -1
        print(f"R2 value line 1: {straigth1.score(x1.reshape((-1, 1)), y1)}, slope: {slope1s}")
        print(f"R2 value line 2: {straigth2.score(x2.reshape((-1, 1)), y2)}, slope: {slope2s}")
        
        if straigth1.score(x1.reshape((-1, 1)), y1) < R2min and slope1s > straigthSlopeThreshold:
            line1IsStraight = False
            print("Line 1 is not straight")
        if straigth2.score(x2.reshape((-1, 1)), y2) < R2min and slope2s > straigthSlopeThreshold:
            line2IsStraight = False
            print("Line 2 is not straight")
        
        if line1IsStraight:
            line2Symetrical = math.isclose(slope2[0], slope2[2], rel_tol = relativeError, abs_tol = absoluteError)
            if line2Symetrical:
                line2cavity = math.isclose(slope2[1], slope1s, rel_tol = relativeError, abs_tol = straigthSlopeThreshold)
                if line2cavity:
                    isCavity = True
                    side = math.floor(self.side/2)*2+1
        elif line2IsStraight:
            line1Symetrical = math.isclose(slope1[0], slope1[2], rel_tol = relativeError, abs_tol = absoluteError)
            if line1Symetrical:
                line1cavity = math.isclose(slope1[1], slope2s, rel_tol = relativeError, abs_tol = straigthSlopeThreshold)
                if line1cavity:
                    isCavity = True
                    side = math.floor(self.side/2)*2

        if debug:
            print(slope1)
            print(slope2)
            plt.figure(figsize=(3,3))
            x1Hat = np.arange(min(x1), max(x1))
            x2Hat = np.arange(min(x2), max(x2))
            y1Hat = pwlfLinePlus.predict(x1Hat)
            y2Hat = pwlfLineMinus.predict(x2Hat)
            plt.plot(x1, y1, ',k', x2, y2, ',b', x1Hat, y1Hat, '-k', x2Hat, y2Hat, '-b')
            plt.title(f"On side {self.side}, Cavity: {isCavity}, side: {side}")
            