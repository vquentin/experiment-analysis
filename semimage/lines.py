import numpy as np

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
        return self.image[row, col]

    def linesOffset(self, distance = 0):
        """Return two lines at distance from the current line.
                
        Keyword arguments:
        distance: the distance from the line in pixels
        """
        return (Line(side = self.side, angle = self._angle, dist = self._dist-distance, image = self.image),
                Line(side = self.side, angle = self._angle, dist = self._dist+distance, image = self.image))

    def isCavity(self):
        """Guess if the current line is a cavity.
        
        Returns a score (high value = more odds it's a cavity)
        """
        pass
