# Copyright (c) 2021, Quentin Van Overmeere
# Licensed under MIT License

from experiments.loader import Experiment
import logging

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s:%(message)s')


if __name__ == '__main__':
    plot_type = 'uniformity-SEM-CS'
    samples = ['ADB1', 'WF76']
    experiment = Experiment(plot_type, *samples)
    experiment.run()
    experiment.plot()
