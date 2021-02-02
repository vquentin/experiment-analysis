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
            lineMinus_oneSegment = LinearRegression(fit_intercept=False).fit(xMinus.reshape((-1, 1)).astype(np.float32), yMinus)
            linePlus_oneSegment = LinearRegression(fit_intercept=False).fit(xPlus.reshape((-1, 1)).astype(np.float32), yPlus)
            self.slopes_oneSegment = [lineMinus_oneSegment.coef_, linePlus_oneSegment.coef_]
            self.R2_oneSegment = [lineMinus_oneSegment.score(xMinus.reshape((-1, 1)), yMinus), linePlus_oneSegment.score(xPlus.reshape((-1, 1)), yPlus)]

            # fit with a piece-wise line
            pwlfLineMinus = pwlf.PiecewiseLinFit(xMinus, yMinus)
            pwlfLinePlus = pwlf.PiecewiseLinFit(xPlus, yPlus)
            self.breaks_threeSegment = [pwlfLineMinus.fit(cavityTestNSeg), pwlfLinePlus.fit(cavityTestNSeg)]
            self.slopes_threeSegment = [pwlfLineMinus.calc_slopes(), pwlfLinePlus.calc_slopes()]
            self.sides = [lineMinus.side, linePlus.side]
            log.debug((f"Fit results \n"
                        f"\tSides: {self.sides} \n"
                        f"\tOne segment lines: \n"
                        f"\t\tSlope: {self.slopes_oneSegment} \n"
                        f"\t\tR2: {self.R2_oneSegment}\n"
                        f"\t{cavityTestNSeg} segment lines: \n"
                        f"\t\tSlopes: {self.slopes_threeSegment}\n"
                        f"\t\tBreakpoints: {self.breaks_threeSegment}"))
            if debug:
                plt.figure(figsize = (4,3))
                xMinusHat = np.arange(xMinus[0], xMinus[-1])
                yminusHat = pwlfLineMinus.predict(xMinusHat)
                xPlusHat = np.arange(xPlus[0], xPlus[-1])
                yPlusHat = pwlfLinePlus.predict(xPlusHat)
                plt.plot(xMinus, yMinus, 'b,', label=f'Side {lineMinus.side}')
                plt.plot(xPlus, yPlus, 'r,', label=f'Side {linePlus.side}')
                plt.plot(xMinusHat, yminusHat, 'b--')
                plt.plot(xPlusHat, yPlusHat, 'r--')
                plt.legend()
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

    def cleanNoisy(self, i):
        #hard coded parameters
        maxSep = 150 #number of pixels max between breakpoints due to artifact
        relativeError = 0.2
        absoluteError = 5

        breakSep = self.breaks_threeSegment[i][2]-self.breaks_threeSegment[i][1]
        slopesAreClose = math.isclose(self.slopes_threeSegment[i][0], self.slopes_threeSegment[i][2], rel_tol = relativeError, abs_tol = absoluteError)
        if breakSep < maxSep and slopesAreClose:
            self.slopes_oneSegment[i] = np.mean([self.slopes_threeSegment[i][0], self.slopes_threeSegment[i][2]])
            return True
        else:
            return False

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

