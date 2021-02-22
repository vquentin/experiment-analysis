# Copyright (c) 2021, Quentin Van Overmeere
# Licensed under MIT License

from experiments.loader import get_experiment
import logging

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s:%(message)s')

SEM_UNIFORMITY = 'uniformity-SEM-CS'
U_VS_T = 'u-vs-t'

if __name__ == '__main__':
    #samples = ('ADB1',)
    samples = ('UB19',)
    metal_masks = ('WF56', 'WF66', 'WF67', 'WF71', 'WF74', 'WF102', 'WF105')
    metal_mask_1 = ('WF56',)
    legend = 'name+mask.material'
    #samples = ('examples',)
    #experiment = get_experiment(sem_uniformity, *samples)
    experiment = get_experiment(U_VS_T, *metal_masks)
    experiment.run()
    experiment.plot(legend)

    experiment2 = get_experiment(SEM_UNIFORMITY, *metal_mask_1)
    experiment2.run()
    experiment2.plot(legend)
