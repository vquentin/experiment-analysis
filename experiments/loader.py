# Copyright (c) 2021, Quentin Van Overmeere
# Licensed under MIT License

from pathlib import Path
import yaml
import logging
import glob
from matplotlib import pyplot as plt
from semimage.sem_image import SEMZeissImage
import semimage.image_analysis as ia
import voltage.electrochemistry as ec

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

    @staticmethod
    def get_nested(dict_, *keys, default=None):
        """From https://stackoverflow.com/a/34958071/13969506"""
        if not isinstance(dict_, dict):
            return default
        elem = dict_.get(keys[0], default)
        if len(keys) == 1:
            return elem
        return Experiment.get_nested(elem, *keys[1:], default=default)

    def __init__(self, experiment_id, *samples):
        self._type = experiment_id
        self._samples = samples
        print(samples)
        print(*samples)

    def run(self):
        raise NotImplementedError

    def plot(self):
        raise NotImplementedError

    def get_path(self, *args):
        return ([Experiment.get_nested(samples_description, sample, 'experiments', self._type, 'path') for sample in args])
        return ([samples_description[sample]['experiments']
                [self._type]['path'] for sample in args])

    def get_legend(self, legend_struct=None):
        if legend_struct is None:
            return ([samples_description[sample]['name']
                     for sample in self._samples])
        legend = []
        subparts = legend_struct.split('+')
        for sample in self._samples:
            legend.append(', '.join(
                [Experiment.get_nested(samples_description, sample,
                                       *tuple(part.split('.')))
                 for part in subparts]))
        return legend


class UniformitySEMCS(Experiment):

    def __init__(self, *samples, **ignored_kwargs):
        super().__init__(UNIFORMITYSEMCS, *samples)
        self._sample_paths = [glob.glob(sample) if sample is not None else None
                              for sample in self.get_path(*samples)]
        self._result = []

    def run(self):
        for sample_path in self._sample_paths:
            for image_file in sample_path:
                plt.show()
                self._result.append(
                    ia.get_porous_thickness(SEMZeissImage(image_file)))

    def plot(self, legend):
        plt.figure()
        plt.plot(self._result, 'o')
        plt.show()


class UvsT(Experiment):

    def __init__(self, *samples, **ignored_kwargs):
        super().__init__(UVST, *samples)
        self._sample_paths = self.get_path(*samples)
        self._result = []

    def run(self):
        for sample_path in self._sample_paths:
            plt.show()
            self._result.append(
                ec.get_U_vs_t(sample_path))

    def plot(self, legend):
        fig = plt.figure()
        for result in self._result:
            result.plot(x='Seconds', y='Voltage', ax=fig.gca())
        plt.legend(self.get_legend(legend_struct=legend))
        plt.xlabel('Time [seconds]')
        plt.ylabel('Cell voltage [V]')
        plt.show()


factory = {
        UNIFORMITYSEMCS: UniformitySEMCS,
        UVST: UvsT
    }


def get_experiment(experiment_id, *args, **kwargs):
    return factory[experiment_id](*args, **kwargs)
