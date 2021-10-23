# Copyright (c) 2021, Quentin Van Overmeere
# Licensed under MIT License

from pathlib import Path
import yaml
import logging
import glob
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import itertools
from cycler import cycler

from semimage.sem_image import SEMZeissImage
import semimage.image_analysis as ia
import voltage.electrochemistry as ec
import optical.moss as moss

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
UNIFORMITYSEMCSNORMALIZE = 'uniformity-SEM-CS-normalize'
UVST = 'u-vs-t'
MOSS = 'MOSS'


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

    def save(self):
        """Cache experiment results"""
        pass

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
            path = Path(sample_path[0])
            name = "result.csv"
            file_candidate = path.parent / name
            if file_candidate.exists():
                self._result.append(pd.read_csv(file_candidate))
            else:
                result_image = np.zeros((len(sample_path), 4))
                for i, image_file in enumerate(sample_path):
                    plt.show()
                    result_image[i, :] = ia.get_porous_thickness(SEMZeissImage(image_file))  # x, y, thickness, uncertainty
                idx = np.argmin(result_image[:,0])
                d = np.linalg.norm(result_image[:, 0:1] - result_image[idx, 0:1], axis=1)
                d = d-np.min(d)
                result = np.concatenate((result_image, d[:, np.newaxis]), axis=1)
                result = result[result[:,0].argsort()]
                df = pd.DataFrame(result, columns=['X [mm]', 'Y [mm]', 'Porous thickness [um]', 'Uncertainty [um]', 'Distance [mm]'])
                self._result.append(df)

    def plot(self, legend):
        fig = plt.figure()
        for result in self._result:
            #plt.errorbar(result[:,4], result[:,2], yerr=result[:,3])
            result.plot(x='Distance [mm]', y='Porous thickness [um]', ax=fig.gca())
        plt.legend(self.get_legend(legend_struct=legend))
        plt.xlabel('Distance [mm]')
        plt.ylabel('Porous Si thickness [um]')
        plt.show()

    def save(self):
        for i, result in enumerate(self._result):
            path = Path(self._sample_paths[i][0])
            name = "result.csv"
            result.to_csv(path.parent / name)


class UniformitySEMCSNormalize(UniformitySEMCS):
    def __init__(self, *samples, **ignored_kwargs):
        super().__init__(*samples, **ignored_kwargs)

    def run(self):
        super().run()
        for result in self._result:
            # transform points to distance from edge
            distCenterCollector = 26500  # um
            distFirstCavity = 6400  # um
            result['Distance to collector [mm]'] = abs(abs((result['Distance [mm]']+distFirstCavity/1000)-(distCenterCollector/1000))-distCenterCollector/1000)
            #TODO normalize thickness result['Ratio ']
            result['Thickness Ratio Center [-]'] = result['Porous thickness [um]']/min(result['Porous thickness [um]'])

    def plot(self, legend):
        fig = plt.figure()
        colors = itertools.cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
        #cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']))
        for result in self._result:
            #plt.errorbar(result[:,4], result[:,2], yerr=result[:,3])
            result.plot(x='Distance to collector [mm]', y='Porous thickness [um]', ax=fig.gca(), kind='scatter', logy=False, color=next(colors))
            #result.plot(x='Distance to collector [mm]', y='Thickness Ratio Center [-]', ax=fig.gca(), kind='scatter', logy=False, color=next(colors))
        plt.legend(self.get_legend(legend_struct=legend))
        plt.xlabel('Distance from current collector [mm]')
        plt.ylabel('Porous thickness ['+chr(956)+'m]')
        fig.gca().set_xlim(xmin=0)
        if fig.gca().get_yscale() is 'linear':
            fig.gca().set_ylim(ymin=0)
        #for result in self._result:
            #A, K = fit_exp_linear(result['Distance to collector [mm]'], result['Porous thickness [um]'], 0)
            #fit_y = model_func(result['Distance to collector [mm]'], A, K, 0)
            #fig.gca().plot(result['Distance to collector [mm]'], fit_y, 'blue', linewidth=1)

            #p = np.polyfit(result['Distance to collector [mm]'], result['Porous thickness [um]'], 3)
            #fig.gca().plot(result['Distance to collector [mm]'], np.polyval(p, result['Distance to collector [mm]']), 'blue', linewidth=1)

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



class MOSS(Experiment):

    def __init__(self, *samples, **ignored_kwargs):
        super().__init__(MOSS, *samples)
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
        UNIFORMITYSEMCSNORMALIZE: UniformitySEMCSNormalize,
        UVST: UvsT,
        MOSS: MOSS
    }


def get_experiment(experiment_id, *args, **kwargs):
    return factory[experiment_id](*args, **kwargs)
