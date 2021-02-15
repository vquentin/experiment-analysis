# Copyright (c) 2021, Quentin Van Overmeere
# Licensed under MIT License

from pathlib import Path
import yaml
import logging
import glob
from matplotlib import pyplot as plt
from semimage.sem_image import SEMZeissImage
import semimage.image_analysis as ia

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# use LibYAML bindings if available (faster)
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
    log.warning("Could not import yaml.CLoader")

# load sample descriptions
path = Path(__file__)
samples_description = yaml.safe_load(open(path.parent / 'samples.yml'))

# constant variables for experiment types
UNIFORMITYSEMCS = 'uniformity-SEM-CS'
UVST = 'u-vs-t'


class Experiment(object):

    def __init__(self, experiment_id):
        self._type = experiment_id

    def run(self):
        raise NotImplementedError

    def plot(self):
        raise NotImplementedError

    def get_path(self, *args):
        return ([samples_description[sample]['experiments']
                [self._type]['path'] for sample in args])


class UniformitySEMCS(Experiment):

    def __init__(self, *samples, **ignored_kwargs):
        super().__init__(UNIFORMITYSEMCS)
        self._sample_paths = [glob.glob(sample) 
                              for sample in self.get_path(*samples)]
        self._result = []

    def run(self):
        for sample_path in self._sample_paths:
            for image_file in sample_path:
                plt.show()
                self._result.append(
                    ia.get_porous_thickness(SEMZeissImage(image_file)))

    def plot(self):
        plt.figure()
        plt.plot(self._result, 'o')
        plt.show()


class UvsT(Experiment):

    def __init__(self, *samples, **ignored_kwargs):
        super().__init__(UVST)
        for sample in samples:
            print("OK")

    def run(self):
        pass

    def plot(self):
        print("it's plotted")


factory = {
        UNIFORMITYSEMCS: UniformitySEMCS,
        UVST: UvsT
    }


def get_experiment(experiment_id, *args, **kwargs):
    return factory[experiment_id](*args, **kwargs)
