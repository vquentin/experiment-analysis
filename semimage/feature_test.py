import numpy as np
from matplotlib import pyplot as plt
import pwlf
import math
import logging
from sklearn.linear_model import LinearRegression

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

class FeatureTest(object):
    def __init__(self, line):
        self.line = line
        self.isNotCavity = False

        #hard coded variables
        cavityTestDistance = 3 # all cavity features should be within 3 pixels from current line

        #perform the fits
        self.cavityFits = FeatureFit(line, cavityTestDistance, multiSeg=True)

    def assessCavity(self, debug=False):
        """Guess if the current line is representative of a cavity.
        
        Keyword arguments:
        debug: a flag to plot diagnostics.

        Returns a dictionary describing the cavity
        """
        if not self.cavityFits.edgeTooClose:
            self.isNotCavity = True
            if self.cavityFits.isLineStraight(mainSide = self.line.side, side='Same'):
                log.debug("Line on same side is straight")
                if not self.cavityFits.isLineStraight(mainSide = self.line.side, side='Other'):
                    log.debug("Line on other side is not straight")
                    if self.cavityFits.isLineSym(mainSide = self.line.side, side='Other'):
                        log.debug("Line on other side is symetrical")
                        if self.cavityFits.isCenterSame(mainSide = self.line.side, side='Other'):
                            log.debug("Line is cavity.")
                            self.isNotCavity = False
        else:
            self.isNotCavity = True

        return (not self.isNotCavity)
            
class FeatureFit(object):
    def __init__(self, line, distance, multiSeg=True):
        cavityTestNSeg = 3

        self.edgeTooClose = False
        lineMinus, linePlus = line.linesOffset(distance=distance)

        if lineMinus.col.size is not 0 and linePlus.col.size is not 0:
            # construct intensity vs. line length
            xMinus = lineMinus.col
            xPlus = linePlus.col
            yMinus = np.cumsum(lineMinus.intensity)
            yPlus = np.cumsum(linePlus.intensity)

            # fit with a straight line
            lineMinus_oneSegment = LinearRegression(fit_intercept=False).fit(xMinus.reshape((-1, 1)),yMinus)
            linePlus_oneSegment = LinearRegression(fit_intercept=False).fit(xPlus.reshape((-1, 1)),yPlus)
            self.slopes_oneSegment = [lineMinus_oneSegment.coef_, linePlus_oneSegment.coef_]
            self.R2_oneSegment = [lineMinus_oneSegment.score(xMinus.reshape((-1, 1)), yMinus), linePlus_oneSegment.score(xPlus.reshape((-1, 1)), yPlus)]

            # fit with a piece-wise line
            pwlfLinePlus = pwlf.PiecewiseLinFit(xMinus, yMinus)
            pwlfLineMinus = pwlf.PiecewiseLinFit(xPlus, yPlus)
            pwlfLinePlus.fit(cavityTestNSeg)
            pwlfLineMinus.fit(cavityTestNSeg)
            self.slopes_threeSegment = [pwlfLinePlus.calc_slopes(), pwlfLineMinus.calc_slopes()]
            self.sides = [lineMinus.side, linePlus.side]
            log.debug((f"Sides: {self.sides} \n"
                        f"Slopes of one segment lines: {self.slopes_oneSegment} \n"
                        f"R2: {self.R2_oneSegment}\n"
                        f"Slopes of multi-segment lines: {self.slopes_threeSegment}"))
        else:
            self.edgeTooClose = True

    def isLineStraight(self, mainSide=None, side=None):
        R2min = 0.99 #threshold to say line is straight
        straigthSlopeThreshold = 36.82

        i = self._side(mainSide, side)
        isStraight = self.R2_oneSegment[i] > R2min or self.slopes_oneSegment[i] < straigthSlopeThreshold
        #TODO add code to remove a bump in the data
        isStraightNoisy = False
        return isStraight or isStraightNoisy

    def isLineSym(self, mainSide=None, side=None):
        relativeError = 0.52
        absoluteError = 5

        i = self._side(mainSide, side)
        isSym = math.isclose(self.slopes_threeSegment[i][0], self.slopes_threeSegment[i][2], rel_tol = relativeError, abs_tol = absoluteError)
        return isSym

    def isCenterSame(self, mainSide=None, side=None):
        relativeError = 0.33
        absoluteError = 23

        iOther = self._side(mainSide, side)
        iSame = self._side(mainSide, 'Same')

        sameSlope = math.isclose(self.slopes_threeSegment[iOther][1], self.slopes_oneSegment[iSame], rel_tol = relativeError, abs_tol = absoluteError)
        return sameSlope

    def _side(self, mainSide, side):
        if mainSide is None or side is None:
            raise Exception("argument cannot be None")
        
        iSame = self.sides.index(mainSide)
        iOther = iSame-1
        if side is 'Same':
            i = iSame
        else:
            i = iOther
        return i

