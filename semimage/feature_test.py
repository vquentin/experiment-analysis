import numpy as np
from matplotlib import pyplot as plt
import pwlf
import math
import logging
from sklearn.linear_model import LinearRegression
import random

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
np.set_printoptions(precision=2)

class FeatureTest(object):
    def __init__(self, line, debug=False):
        self.line = line
        self.isNotCavity = True
        self.isNotLine = True
        #perform the fits
        self.cavityFits = FeatureFit(line, featureTests='cavity', debug=debug)
        self.straightLineFits = FeatureFit(line, featureTests='line', debug=debug)
        random.seed()

    def assessCavity(self):
        """Guess if the current line is representative of a cavity.
        
        Keyword arguments:
        debug: a flag to plot diagnostics.

        Return True if confident it's a cavity
        """
        if not self.cavityFits.edgeTooClose:
            log.debug("PASS Cavity test 1/5: edge not too close")
            if self.cavityFits.isVoid(mainSide = self.line.side, side='Same'):
                log.debug("PASS Cavity test 2/5: Intensity on outer side is uniform")
                if not self.cavityFits.isLineStraight(mainSide = self.line.side, side='Other'):
                    log.debug("PASS Cavity test 3/5: Intensity on inner side is not uniform")
                    if self.cavityFits.isLineSym(mainSide = self.line.side, side='Other'):
                        log.debug("PASS Cavity test 4/5: Intensity in inner side is symetrical")
                        if self.cavityFits.isCenterSame(mainSide = self.line.side, side='Other'):
                            log.debug("PASS Cavity test 5/5: Intensity in center of inner side same as outer side.")
                            self.isNotCavity = False
        return (not self.isNotCavity)

    def assessLine(self):
        """Guess if the current line is representative of a straight line porous Si/Si or Si/void or porous Si/void.
        
        Keyword arguments:
        debug: a flag to plot diagnostics.

        Return true if confident it's a line
        """
        #TODO implement feature
        return False
            
class FeatureFit(object):
    def __init__(self, line, featureTests=None, debug=False):
        if featureTests == 'cavity':
            dist = 3
        elif featureTests == 'line':
            dist = 10

        self.id = random.randint(0,1000)

        lineMinus, linePlus = line.linesOffset(distance=dist)
        self.lineMinus = lineMinus
        self.linePlus = linePlus
        self.edgeTooClose = False
        if lineMinus.col.size == 0 or linePlus.col.size == 0:
            self.edgeTooClose = True

        if not self.edgeTooClose:
            # construct intensity vs. line length
            xMinus = lineMinus.col
            xPlus = linePlus.col
            yMinus = np.cumsum(lineMinus.intensity)
            yPlus = np.cumsum(linePlus.intensity)

            #construct line intensity difference
            lineDiff = self._diffLineIntensity(lineMinus, linePlus)
            self.lineDiff = np.cumsum(self._diffLineIntensity(lineMinus, linePlus))

            # fit with a straight line
            lineMinus_oneSegment = LinearRegression(fit_intercept=False).fit(xMinus.reshape((-1, 1)).astype(np.float32), yMinus)
            linePlus_oneSegment = LinearRegression(fit_intercept=False).fit(xPlus.reshape((-1, 1)).astype(np.float32), yPlus)
            self.lines = [lineMinus, linePlus]
            self.sides = [lineMinus.side, linePlus.side]
            self.sideNames = [self._sideName(lineMinus.side, line.side), self._sideName(linePlus.side, line.side)]
            self.slopes_oneSegment = [lineMinus_oneSegment.coef_[0], linePlus_oneSegment.coef_[0]]
            self.R2_oneSegment = [lineMinus_oneSegment.score(xMinus.reshape((-1, 1)), yMinus), linePlus_oneSegment.score(xPlus.reshape((-1, 1)), yPlus)]
            self.ssr_oneSegment = [np.sum((yMinus-lineMinus_oneSegment.predict(xMinus.reshape((-1, 1))))**2), np.sum((yPlus-linePlus_oneSegment.predict(xPlus.reshape((-1, 1))))**2)]
            if featureTests == 'line':
                log.debug((f"Fit results ID {self.id} ({featureTests})\n"
                        f"\tSides: {self.sideNames} \n"
                            f"\tOne segment lines: \n"
                            f"\t\tSlope: {self.slopes_oneSegment} \n"
                            f"\t\tR2: {self.R2_oneSegment}\n"
                            f"\t\tResiduals: {self.ssr_oneSegment}"))
                if debug:
                    plt.figure(figsize = (4,3))
                    xMinusHat = np.arange(xMinus[0], xMinus[-1])
                    yminusHat = 0 #TODO change if fit needed
                    xPlusHat = np.arange(xPlus[0], xPlus[-1])
                    yPlusHat = 0 #TODO change if fit display needed
                    plt.plot(xMinus, yMinus, 'b,', label=f'Side {self.sideNames[0]}')
                    plt.plot(xPlus, yPlus, 'r,', label=f'Side {self.sideNames[1]}')
                    #plt.plot(xMinus,np.cumsum(linePlus.intensity-lineMinus.intensity), 'k-', label="Difference")
                    plt.legend()
                    plt.title(f"Feature test ID {self.id} ({featureTests})")

            if featureTests == 'cavity':
                cavityTestNSeg = 3
                # fit with a piece-wise line
                pwlfLineMinus = pwlf.PiecewiseLinFit(xMinus, yMinus)
                pwlfLinePlus = pwlf.PiecewiseLinFit(xPlus, yPlus)
                self.breaks_threeSegment = [pwlfLineMinus.fitfast(cavityTestNSeg, pop=10), pwlfLinePlus.fitfast(cavityTestNSeg, pop=10)]
                self.slopes_threeSegment = [pwlfLineMinus.calc_slopes(), pwlfLinePlus.calc_slopes()]
                self.ssr_threeSegment = [pwlfLineMinus.ssr, pwlfLinePlus.ssr]
                log.debug((f"Fit results ID {self.id} ({featureTests})\n"
                            f"\tSides: {self.sideNames} \n"
                            f"\tOne segment lines: \n"
                            f"\t\tSlope: {self.slopes_oneSegment} \n"
                            f"\t\tR2: {self.R2_oneSegment}\n"
                            f"\t\tResiduals: {self.ssr_oneSegment}\n"
                            f"\t{cavityTestNSeg} segment lines: \n"
                            f"\t\tSlopes: {self.slopes_threeSegment}\n"
                            f"\t\tBreakpoints: {self.breaks_threeSegment}\n"
                            f"\t\tResiduals: {self.ssr_threeSegment}"))
                if debug:
                    plt.figure(figsize = (4,3))
                    xMinusHat = np.arange(xMinus[0], xMinus[-1])
                    yminusHat = pwlfLineMinus.predict(xMinusHat)
                    xPlusHat = np.arange(xPlus[0], xPlus[-1])
                    yPlusHat = pwlfLinePlus.predict(xPlusHat)
                    plt.plot(xMinus, yMinus, 'b,', label=f'Side {self.sideNames[0]}')
                    plt.plot(xPlus, yPlus, 'r,', label=f'Side {self.sideNames[1]}')
                    plt.plot(xMinusHat, yminusHat, 'b-')
                    plt.plot(xPlusHat, yPlusHat, 'r-')
                    #plt.plot(xMinus,np.cumsum(linePlus.intensity-lineMinus.intensity), 'k-', label="Difference")
                    plt.legend()
                    plt.title(f"Feature test ID {self.id} ({featureTests})")

                    plt.figure()
                    plt.plot(lineDiff, '-k')
                    plt.title(f"Diff line ID {self.id} ({featureTests})")
        
    def isVoid(self, mainSide=None, side=None):
        maxStdDev = 22 #threshold to say line is straight (UB19_14)
        thresholdMin = 35
        thresholdMax = 255-thresholdMin
        thresholdSuperMin = 4
        thresholdSuperMax = 255-thresholdSuperMin
        
        i = self._side(mainSide, side)
        stdDev = np.std(self.lines[i].intensity)
        avg = np.median(self.lines[i].intensity)
        print(f"Hey stddev is {stdDev} and avg is {avg}")

        isUniform = (avg < thresholdSuperMin or avg > thresholdSuperMax) or (stdDev < maxStdDev and (avg < thresholdMin or avg > thresholdMax))
        #TODO add code to remove a bump in the data
        
        return isUniform
    
    def isLineStraight(self, mainSide=None, side=None):
        R2min = 0.99 #threshold to say line is straight
        straigthSlopeThreshold = 30

        i = self._side(mainSide, side)
        isStraight = self.R2_oneSegment[i] > R2min or self.slopes_oneSegment[i] < straigthSlopeThreshold
        #TODO add code to remove a bump in the data
        if side is 'Same':
            isStraightNoisy = self.cleanNoisy(i)
        else:
            isStraightNoisy = False
        return isStraight or isStraightNoisy

    def cleanNoisy(self, i):
        #hard coded parameters
        maxSep = 150 #number of pixels max between breakpoints due to artifact
        relativeError = 0.2
        absoluteError = 13

        breakSep = self.breaks_threeSegment[i][2]-self.breaks_threeSegment[i][1]
        slopesAreClose = math.isclose(self.slopes_threeSegment[i][0], self.slopes_threeSegment[i][2], rel_tol = relativeError, abs_tol = absoluteError)
        if breakSep < maxSep and slopesAreClose:
            self.slopes_oneSegment[i] = np.mean([self.slopes_threeSegment[i][0], self.slopes_threeSegment[i][2]])
            log.debug(f"Line on side {i} cleaned")
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
        pwlfC = pwlf.PiecewiseLinFit(np.arange(0,self.lineDiff.shape[0]), self.lineDiff)
        pwlfCb = pwlfC.fitfast(3, pop=10)
        pwlfCs = pwlfC.calc_slopes()
        sameSlope = math.isclose(pwlfCs[1], 0, rel_tol=0.1, abs_tol=10)

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

    def _sideName(self, sideTest, mainSide):
        if sideTest == mainSide:
            return 'Same'
        else:
            return 'Other'

    def _diffLineIntensity(self, lineMinus, linePlus):
        return (linePlus.lineProjection(lineMinus)-lineMinus.lineProjection(linePlus))