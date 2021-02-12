from semimage.feature_test import FeatureTest

class Feature(object):

    def distance_to_edge(self, edge):
            """Return euclidian distance to edge vs position along the line.

            edge is a binary image array
            Returns a tuple of numpy arrays with distance to edge along the line
            vs. line pixel, without NaN values.
            """
            # rotate line and edge images
            angle = -(90-math.degrees(self.__angle))
            line_as_image = np.zeros_like(self.__image, dtype=bool)
            line_as_image[self.rows, self.columns] = True
            line_as_image_rotated = transform.rotate(
                line_as_image, angle, resize=True, center=(0, 0))
            edge_rotated = transform.rotate(edge, angle, resize=True, center=(0, 0))
            # assume no edge is found by default
            dist = np.full_like(line_as_image_rotated[0, :], np.nan, 
                                dtype=np.float64)
            row_of_line_rotated = np.mean(np.nonzero(line_as_image_rotated)[0])
            rows_of_edge_rotated = np.argmax(edge_rotated, axis=0)
            mask = rows_of_edge_rotated == 0
            dist = rows_of_edge_rotated-row_of_line_rotated
            columns = np.arange(0, edge_rotated.shape[1])
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