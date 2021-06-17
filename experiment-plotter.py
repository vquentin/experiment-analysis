# Copyright (c) 2021, Quentin Van Overmeere
# Licensed under MIT License

from experiments.loader import get_experiment
import logging

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s:%(message)s')

SEM_UNIFORMITY = 'uniformity-SEM-CS'
SEM_UNIFORMITY_NORMALIZED = 'uniformity-SEM-CS-normalize'
U_VS_T = 'u-vs-t'

if __name__ == '__main__':
    #samples = ('ADB1',)
    #samples = ('UB19',)
    metal_masks = ('WF56', 'WF66', 'WF67', 'WF74', 'WF102', 'WF40')
    #metal_mask_1 = ('WF102',)
    legend = 'mask.material'
    #samples = ('examples',)
    experiment = get_experiment(SEM_UNIFORMITY, *metal_masks)
    #experiment = get_experiment(U_VS_T, *metal_masks)
    #experiment = get_experiment(SEM_UNIFORMITY_NORMALIZED, *metal_masks)
    experiment.run()
    experiment.plot(legend)

    #experiment2 = get_experiment(SEM_UNIFORMITY, *metal_masks)
    #experiment2.run()
    #experiment2.plot(legend)
    #experiment2.save()
