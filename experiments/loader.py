# Copyright (c) 2021, Quentin Van Overmeere
# Licensed under MIT License

from pathlib import Path
import yaml
import logging

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
        return [samples_description[sample]['experiments'][self._type]['path'] for sample in args]


class UniformitySEMCS(Experiment):

    def __init__(self, *samples, **ignored_kwargs):
        super().__init__(UNIFORMITYSEMCS)
        for sample in samples:
            print(f"Sample {sample} uniformity plot generated")
            print(self.get_path(sample))

    def run(self):
        pass

    def plot(self):
        print("plotted")


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
