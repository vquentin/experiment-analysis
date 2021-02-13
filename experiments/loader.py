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

#load sample descriptions
path = Path(__file__)
samples_description = yaml.safe_load(open(path.parent / 'samples.yml'))


class Experiment(object):

    def __init__(self, experiment_id, *args, **kwargs):
        self._experiment = factory[experiment_id](*args, **kwargs)

    def run(self):
        self._experiment.run()

    def plot(self):
        self._experiment.plot()


class UniformitySEMCS(Experiment):

    def __init__(self, *samples, **ignored_kwargs):
        for sample in samples:
            print(f"Sample {sample} uniformity plot generated")

    def run(self):
        pass

    def plot(self):
        print("plotted")


factory = {
        'uniformity-SEM-CS': UniformitySEMCS
    }